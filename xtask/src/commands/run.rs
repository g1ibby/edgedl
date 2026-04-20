use std::path::Path;

use anyhow::Result;
use clap::Subcommand;

use super::{ExamplesArgs, TestsArgs};
use crate::{cargo::CargoAction, firmware::Metadata};

// ----------------------------------------------------------------------------
// Subcommands

#[derive(Debug, Subcommand)]
pub enum Run {
    /// Run the given example for the specified chip.
    Example(ExamplesArgs),
    /// Run all applicable tests or the specified test for a specified chip.
    Tests(TestsArgs),
    /// Run HIL tests end-to-end: flash, capture RTT tensor dumps to a temp
    /// directory, and run the matching host crosscheck (`tests/<name>_crosscheck.rs`).
    FullTests(TestsArgs),
}

// ----------------------------------------------------------------------------
// Subcommand Actions

/// Run the specified examples for the given chip.
pub fn run_examples(
    args: ExamplesArgs,
    examples: Vec<Metadata>,
    package_path: &Path,
) -> Result<()> {
    let mut examples = examples;

    // At this point, chip can never be `None`, so we can safely unwrap it.
    let chip = args.chip.unwrap();
    let target = args.package.target_triple(&chip)?;

    examples.sort_by_key(|ex| ex.tag());

    let console = console::Term::stdout();

    for example in examples {
        let mut skip = false;

        log::info!("Running example '{}'", example.output_file_name());
        if let Some(description) = example.description() {
            log::info!(
                "\n\n{}\n\nPress ENTER to run example, `s` to skip",
                description.trim()
            );
        } else {
            log::info!("\n\nPress ENTER to run example, `s` to skip");
        }

        loop {
            let key = console.read_key();

            match key {
                Ok(console::Key::Enter) => break,
                Ok(console::Key::Char('s')) => {
                    skip = true;
                    break;
                }
                _ => (),
            }
        }

        while !skip {
            let result = crate::execute_app(
                package_path,
                chip,
                &target,
                &example,
                CargoAction::Run,
                args.debug,
                args.toolchain.as_deref(),
                args.timings,
                &[],
            );

            if let Err(error) = result {
                log::error!("Failed to run example: {}", error);
                log::info!("Retry or skip? (r/s)");
                loop {
                    let key = console.read_key();

                    match key {
                        Ok(console::Key::Char('r')) => break,
                        Ok(console::Key::Char('s')) => {
                            skip = true;
                            break;
                        }
                        _ => (),
                    }
                }
            } else {
                break;
            }
        }
    }

    Ok(())
}
