use structmeta::StructMeta;

/// Parsed arguments for the `#[espdl_model(...)]` attribute.
///
/// Supported forms:
/// - Named path: `path = "..."`
/// - Positional path: `"..."` (captured as `alt_path`)
#[derive(StructMeta)]
pub struct EspdlArgs {
    /// Optional positional path argument, e.g. #[espdl_model("model.espdl")]
    #[struct_meta(unnamed)]
    pub alt_path: Option<syn::LitStr>,

    /// Named path argument, e.g. #[espdl_model(path = "model.espdl")]
    pub path: Option<syn::LitStr>,
    // No additional flags supported (keep args minimal and deterministic)
}
