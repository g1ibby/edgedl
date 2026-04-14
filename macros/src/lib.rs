//! Proc-macro crate for edgedl. Owns the .fbs schema and (eventually) the
//! generated FlatBuffers bindings. Runtime remains no_std and does not depend
//! on FlatBuffers.

extern crate proc_macro;

use std::{collections::BTreeMap, fs};

use proc_macro::TokenStream;
use quote::{format_ident, quote, ToTokens};
use syn::{parse_macro_input, Item};
#[path = "Dl_generated.rs"]
#[allow(unused_imports, dead_code, clippy::all, mismatched_lifetime_syntaxes)]
mod dl_generated;
use dl_generated::dl;

mod arena_liveness;
mod args;
mod context;
mod initializers;
mod loader;
mod ops;
mod util;
mod values;

use proc_macro_error::{abort_call_site, proc_macro_error};

// All diagnostics use proc-macro-error's abort/abort_call_site macros.
use crate::loader::{parse_edl_header, resolve_and_read_model};

#[proc_macro_error]
#[proc_macro_attribute]
pub fn espdl_model(args: TokenStream, input: TokenStream) -> TokenStream {
    let item = parse_macro_input!(input as Item);

    // Parse attribute args: expect named `path = "..."` (ignore others for now).
    let args_parsed = parse_macro_input!(args as args::EspdlArgs);
    let model_path = args_parsed
        .path
        .or(args_parsed.alt_path)
        .unwrap_or_else(|| syn::LitStr::new("<unspecified>", proc_macro2::Span::call_site()));

    // Read header + payload now so we can build model constants.
    // Resolve model path relative to the consuming crate's manifest dir.
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".into());
    let raw_bytes = match resolve_and_read_model(&manifest_dir, &model_path.value()) {
        Ok(t) => t,
        Err(e) => {
            abort_call_site!("{}", e)
        }
    };
    let payload = match parse_edl_header(&raw_bytes) {
        Ok(p) => p,
        Err(msg) => {
            abort_call_site!("{}", msg)
        }
    };

    let fb_model = match dl::root_as_model(payload) {
        Ok(m) => m,
        Err(_) => {
            abort_call_site!("FlatBuffers verification failed")
        }
    };
    let fb_graph = match fb_model.graph() {
        Some(g) => g,
        None => {
            abort_call_site!("missing graph")
        }
    };

    // Build name <-> id mappings via context helper
    let (name_to_id, name_by_id): (BTreeMap<String, u16>, Vec<String>) =
        context::build_name_maps(fb_graph);

    // Collect quick-lookup metadata for validations
    let mut val_meta_map: BTreeMap<String, context::ValMeta> =
        match context::seed_val_meta_from_value_info(fb_graph) {
            Ok(m) => m,
            Err(e) => abort_call_site!("{}", e),
        };
    let mut init_info_map: BTreeMap<String, context::InitInfo> = BTreeMap::new();

    // Values + IO arrays
    let (values_tokens, inputs_ids, outputs_ids) =
        values::emit_values_and_io(fb_graph, &name_to_id, &name_by_id, &mut val_meta_map);

    // Initializers
    let init_tokens = match initializers::emit_initializers(
        fb_graph,
        &name_to_id,
        &mut init_info_map,
        &manifest_dir,
    ) {
        Ok(t) => t,
        Err(e) => {
            abort_call_site!("{}", e)
        }
    };

    // Build node plan (Conv2D, Pad, ReduceMean, Gemm)
    let (plan_tokens, ops_set, simd_scratch_max) =
        match ops::emit_nodes(fb_graph, &name_to_id, &val_meta_map, &init_info_map) {
            Ok(r) => r,
            Err(e) => {
                abort_call_site!("{}", e)
            }
        };

    // Build inputs/outputs arrays tokens
    let input_elems = inputs_ids
        .iter()
        .map(|id| quote! { #id })
        .collect::<Vec<_>>();
    let output_elems = outputs_ids
        .iter()
        .map(|id| quote! { #id })
        .collect::<Vec<_>>();
    let values_elems = values_tokens;
    // Debug: expose value names in id order for diagnostics
    let value_name_elems = name_by_id.iter().map(|s| quote! { #s }).collect::<Vec<_>>();
    let init_statics = init_tokens.statics;
    let init_array_elems = init_tokens.table;
    let plan_elems = plan_tokens;
    let ops_array = ops_set.iter().map(|s| quote! { #s }).collect::<Vec<_>>();

    // Compute zero-alloc arena layout at macro-time, reusing freed regions
    // using liveness intervals over the node execution order.
    let align: usize = 16; // keep a 16-byte alignment boundary for offsets
    let arena_plan = match arena_liveness::plan_arena_liveness(
        fb_graph,
        &name_to_id,
        &name_by_id,
        &val_meta_map,
        &init_info_map,
        &inputs_ids,
        &outputs_ids,
        align,
    ) {
        Ok(p) => p,
        Err(e) => abort_call_site!("{}", e),
    };
    let offsets_host = arena_plan.offsets;
    let size_bytes_host: usize = arena_plan.size_bytes;
    let scratch_bytes_host: usize = (simd_scratch_max + (align - 1)) & !(align - 1);
    let offsets_tokens: Vec<proc_macro2::TokenStream> =
        offsets_host.iter().map(|o| quote! { #o }).collect();
    let arena_report_lit = syn::LitStr::new(&arena_plan.report, proc_macro2::Span::call_site());

    // Determine the identifier of the annotated item to attach associated consts/methods.
    let (type_ident, item_tokens) = match &item {
        Item::Struct(s) => (s.ident.clone(), s.to_token_stream()),
        Item::Enum(e) => (e.ident.clone(), e.to_token_stream()),
        Item::Type(t) => (t.ident.clone(), t.to_token_stream()),
        _ => {
            abort_call_site!("#[espdl_model] must be applied to a struct/enum/type item")
        }
    };

    // Private per-invocation module name derived from item ident
    let mod_ident = format_ident!("__nn_{}_gen", type_ident);
    // Public arena type alias name (e.g., __ModelBindArena)
    let arena_type_ident = format_ident!("{}Arena", type_ident);

    // Expansion: encapsulate statics in a private module and attach public API to the type
    let runtime_ident = format_ident!("{}Runtime", type_ident);
    let expanded = quote! {
        #item_tokens

        // Generated code lives inside a private module to avoid symbol collisions
        #[allow(non_snake_case)]
        mod #mod_ident {
            // Generated parameter storage (aligned) and exponents tables.
            #( #init_statics )*

            // Zero-alloc arena plan (offsets and total size), computed at compile-time by the macro.
            pub(super) const ARENA_OFFSETS: &[usize] = &[ #( #offsets_tokens ),* ];
            pub(super) const ARENA_SIZE: usize = #size_bytes_host;
            pub(super) const ARENA_SCRATCH: usize = #scratch_bytes_host;
            pub(super) const PLANNED: edgedl::arena::PlannedArena = edgedl::arena::PlannedArena {
                size_bytes: ARENA_SIZE,
                offsets: ARENA_OFFSETS,
                scratch_bytes: ARENA_SCRATCH,
            };

            #[allow(unexpected_cfgs)]
            #[cfg(any(test, debug_assertions, feature = "arena-report"))]
            pub(super) const ARENA_REPORT: &str = #arena_report_lit;

            // Generated model descriptor (kept crate-private to shrink public surface)
            pub(super) const SPEC: edgedl::model::ModelSpec = edgedl::model::ModelSpec {
                alignment: 16,
                inputs: &[ #( #input_elems ),* ],
                outputs: &[ #( #output_elems ),* ],
                values: &[ #( #values_elems ),* ],
                initializers: &[ #( #init_array_elems ),* ],
                plan: &[ #( #plan_elems ),* ],
            };

            // Publicly exposed via associated consts on the annotated type
            pub(super) const MODEL_OPS: &[&str] = &[ #( #ops_array ),* ];
            // Debugging: map ValueId -> name
            pub(super) const VALUE_NAMES: &[&str] = &[ #( #value_name_elems ),* ];
        }

        impl #type_ident {
            // Public associated constants for the model and fixtures
            pub const SPEC: edgedl::model::ModelSpec = #mod_ident::SPEC;
            pub const MODEL_OPS: &'static [&'static str] = #mod_ident::MODEL_OPS;
            pub const VALUE_NAMES: &'static [&'static str] = #mod_ident::VALUE_NAMES;
            pub const ARENA_SIZE: usize = #mod_ident::ARENA_SIZE + #mod_ident::ARENA_SCRATCH;
            #[allow(unexpected_cfgs)]
            #[cfg(any(test, debug_assertions, feature = "arena-report"))]
            pub const ARENA_REPORT: &'static str = #mod_ident::ARENA_REPORT;

            /// Create a new zeroed arena suitable for this model (16-byte aligned for SIMD).
            pub const fn new_arena() -> #arena_type_ident {
                edgedl::Aligned16([0i8; Self::ARENA_SIZE])
            }

            /// Construct a reusable inference runtime bound to `arena`.
            /// No allocation; returns an error if the arena is too small.
            pub fn new<'a>(arena: &'a mut [i8]) -> edgedl::error::Result<#runtime_ident<'a>> {
                #runtime_ident::new(arena)
            }
        }

        // A reusable, allocation-free runtime bound to a caller-provided arena.
        // Minimal surface: new() and predict().
        pub struct #runtime_ident<'a> {
            eng: edgedl::engine::Engine<'static, 'static, 'a>,
        }

        impl<'a> #runtime_ident<'a> {
            pub fn new(arena: &'a mut [i8]) -> edgedl::error::Result<Self> {
                if arena.len() < #type_ident::ARENA_SIZE {
                    return Err(edgedl::error::Error::ArenaTooSmall { expected: #type_ident::ARENA_SIZE, got: arena.len() });
                }
                let eng = edgedl::engine::Engine::from_parts(&#type_ident::SPEC, &#mod_ident::PLANNED, arena);
                Ok(Self { eng })
            }

            pub fn predict(&mut self, input: &[i8], probs: &mut [f32]) -> edgedl::error::Result<()> {
                if #type_ident::SPEC.inputs.is_empty() { return Err(edgedl::error::Error::NoInputs); }
                if #type_ident::SPEC.outputs.is_empty() { return Err(edgedl::error::Error::NoOutputs); }

                let in_id = #type_ident::SPEC.inputs[0];
                let in_meta = #type_ident::SPEC.values[in_id as usize];
                let in_need = in_meta.shape.elements();
                if input.len() != in_need {
                    return Err(edgedl::error::Error::InputLenMismatch { expected: in_need, got: input.len() });
                }

                let out_id = #type_ident::SPEC.outputs[0];
                let out_meta = #type_ident::SPEC.values[out_id as usize];
                let out_need = out_meta.shape.elements();
                if probs.len() != out_need {
                    return Err(edgedl::error::Error::OutputLenMismatch { expected: out_need, got: probs.len() });
                }

                self.eng.write_input_index(0, input);
                self.eng.infer_scalar();
                let out_view = self.eng.read_output_id(out_id);
                edgedl::post::softmax_from_i8(out_view, out_meta.exp, probs);
                Ok(())
            }

            /// Predict using the SIMD-preferred execution path. Available only
            /// for Xtensa targets; not emitted for host/desktop builds.
            #[cfg(target_arch = "xtensa")]
            pub fn predict_simd(&mut self, input: &[i8], probs: &mut [f32]) -> edgedl::error::Result<()> {
                if #type_ident::SPEC.inputs.is_empty() { return Err(edgedl::error::Error::NoInputs); }
                if #type_ident::SPEC.outputs.is_empty() { return Err(edgedl::error::Error::NoOutputs); }

                let in_id = #type_ident::SPEC.inputs[0];
                let in_meta = #type_ident::SPEC.values[in_id as usize];
                let in_need = in_meta.shape.elements();
                if input.len() != in_need {
                    return Err(edgedl::error::Error::InputLenMismatch { expected: in_need, got: input.len() });
                }

                let out_id = #type_ident::SPEC.outputs[0];
                let out_meta = #type_ident::SPEC.values[out_id as usize];
                let out_need = out_meta.shape.elements();
                if probs.len() != out_need {
                    return Err(edgedl::error::Error::OutputLenMismatch { expected: out_need, got: probs.len() });
                }

                self.eng.write_input_index(0, input);
                self.eng.infer_simd();
                let out_view = self.eng.read_output_id(out_id);
                edgedl::post::softmax_from_i8(out_view, out_meta.exp, probs);
                Ok(())
            }
        }

        // Expose a stable, model-specific runtime type name derived from the annotated item
        // Usage: let mut rt = #runtime_ident::new(&mut arena)?; rt.predict(...)?;

        /// Arena type for this model - 16-byte aligned for SIMD operations.
        pub type #arena_type_ident = edgedl::Aligned16<[i8; #type_ident::ARENA_SIZE]>;
    };
    // Write expansion to target for debugging (best-effort)
    let _ = fs::write("target/edgedl-expansion.rs", expanded.to_string());

    TokenStream::from(expanded)
}
