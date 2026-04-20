use std::path::Path;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use clap::Args;
use esp_metadata::Chip;
use inquire::Select;

pub use self::{build::*, check_changelog::*, release::*, run::*};
use crate::{
    Package,
    cargo::{CargoAction, CargoCommandBatcher},
    supported_chips,
};
mod build;
mod check_changelog;
mod release;
mod run;

// ----------------------------------------------------------------------------
// Subcommand Arguments

/// Arguments common to commands which act on examples.
#[derive(Debug, Args)]
pub struct ExamplesArgs {
    /// Example to act on ("all" will execute every example).
    pub example: Option<String>,
    /// Chip to target.
    #[arg(value_enum, long)]
    pub chip: Option<Chip>,
    /// Package whose examples we wish to act on.
    #[arg(value_enum, long, default_value_t = Package::Examples)]
    pub package: Package,
    /// Build examples in debug mode only
    #[arg(long)]
    pub debug: bool,

    /// The toolchain used to build the examples
    #[arg(long)]
    pub toolchain: Option<String>,

    /// Emit crate build timings
    #[arg(long)]
    pub timings: bool,
}

/// Arguments common to commands which act on tests.
#[derive(Debug, Args)]
pub struct TestsArgs {
    /// Chip to target.
    #[arg(value_enum)]
    pub chip: Chip,

    /// Repeat the tests for a specific number of times.
    #[arg(long, default_value_t = 1)]
    pub repeat: usize,
    /// Optional test to act on (all tests used if omitted).
    ///
    /// The `test_suite::test_name` syntax allows running a single specific test.
    #[arg(long, short = 't')]
    pub test: Option<String>,

    /// The toolchain used to build the tests
    #[arg(long)]
    pub toolchain: Option<String>,

    /// Emit crate build timings
    #[arg(long)]
    pub timings: bool,
}

// ----------------------------------------------------------------------------
// Subcommand Actions

/// Execute the given action on the specified examples.
pub fn examples(workspace: &Path, mut args: ExamplesArgs, action: CargoAction) -> Result<()> {
    log::debug!(
        "Running examples for '{}' on '{:?}'",
        args.package,
        args.chip
    );
    if args.chip.is_none() {
        let chip_variants = supported_chips();

        let chip = Select::new("Select your target chip:", chip_variants).prompt()?;

        args.chip = Some(chip);
    }

    let chip = args.chip.unwrap();

    // Ensure that the package/chip combination provided are valid:
    args.package.validate_package_chip(&chip).with_context(|| {
        format!(
            "The package '{0}' does not support the chip '{chip:?}'",
            args.package
        )
    })?;

    // Absolute path of the package's root:
    let package_path = crate::windows_safe_path(&workspace.join(args.package.dir_name()));

    // Load all examples which support the specified chip and parse their metadata.
    //
    // The `examples` directory contains a number of individual projects, and does not rely on
    // metadata comments in the source files. As such, it needs to load its metadata differently
    // than other packages.
    let examples = if args.package == Package::Examples {
        crate::firmware::load_cargo_toml(&package_path).with_context(|| {
            format!(
                "Failed to load specified examples from {}",
                package_path.display()
            )
        })?
    } else {
        let example_path = match args.package {
            Package::HilTest => package_path.join("tests"),
            _ => package_path.join("examples"),
        };

        crate::firmware::load(&example_path)?
    };

    let mut examples = examples
        .into_iter()
        .filter(|example| example.supports_chip(chip))
        .collect::<Vec<_>>();

    // At this point, chip can never be `None`, so we can safely unwrap it.
    let chip = args.chip.unwrap();

    // Filter the examples down to only the binaries supported by the given chip
    examples.retain(|ex| ex.supports_chip(chip));

    // Sort all examples by name:
    examples.sort_by_key(|a| a.binary_name());

    let mut filtered = vec![];

    if let Some(example) = args.example.as_deref() {
        filtered.clone_from(&examples);
        if !example.eq_ignore_ascii_case("all") {
            // Only keep the example the user wants
            filtered.retain(|ex| ex.matches_name(example));

            if filtered.is_empty() {
                log::warn!(
                    "Example '{example}' not found or unsupported for the given chip. Please select one of the existing examples in the desired package."
                );

                let example_name = inquire::Select::new(
                    "Select the example:",
                    examples.iter().map(|ex| ex.binary_name()).collect(),
                )
                .prompt()?;

                if let Some(selected) = examples.iter().find(|ex| ex.binary_name() == example_name)
                {
                    filtered.push(selected.clone());
                }
            }
        }
    } else {
        let example_name = inquire::Select::new(
            "Select an example:",
            examples.iter().map(|ex| ex.binary_name()).collect(),
        )
        .prompt()?;

        if let Some(selected) = examples.iter().find(|ex| ex.binary_name() == example_name) {
            filtered.push(selected.clone());
        }
    }

    // Execute the specified action:
    match action {
        CargoAction::Build(out_path) => build_examples(
            args,
            filtered,
            &package_path,
            out_path.as_ref().map(|p| p.as_path()),
        ),
        CargoAction::Run => run_examples(args, filtered, &package_path),
    }
}

/// Execute the given action on the specified doctests.
pub fn tests(workspace: &Path, args: TestsArgs, action: CargoAction) -> Result<()> {
    let (test_arg, filter) = if let Some(test_arg) = args.test.as_deref() {
        match test_arg.split_once("::") {
            Some((test, filter)) => (Some(test), Some(filter)),
            None => (Some(test_arg), None),
        }
    } else {
        (None, None)
    };

    // Absolute path of the 'hil-test' package's root:
    let package_path = crate::windows_safe_path(&workspace.join("hil-test"));

    // Determine the appropriate build target for the given package and chip:
    let target = Package::HilTest.target_triple(&args.chip)?;

    // Load all tests which support the specified chip and parse their metadata:
    let mut tests = crate::firmware::load(&package_path.join("src").join("bin"))?
        .into_iter()
        .filter(|example| example.supports_chip(args.chip))
        .collect::<Vec<_>>();

    // Sort all tests by name:
    tests.sort_by_key(|a| a.binary_name());

    let run_test_extra_args = (action == CargoAction::Run)
        .then(|| filter.as_ref().map(|f| std::slice::from_ref(f)))
        .flatten()
        .unwrap_or(&[]);

    if let CargoAction::Build(Some(out_dir)) = &action {
        // Make sure the tmp directory has no garbage for us.
        let tmp_dir = out_dir.join("tmp");
        _ = std::fs::remove_dir_all(&tmp_dir);
        std::fs::create_dir_all(&tmp_dir).unwrap();
    }

    let mut commands = CargoCommandBatcher::new();
    // Execute the specified action:
    if tests.iter().any(|test| test.matches(test_arg.as_deref())) {
        for test in tests
            .iter()
            .filter(|test| test.matches(test_arg.as_deref()))
        {
            let command = crate::generate_build_command(
                &package_path,
                args.chip,
                &target,
                &test,
                action.clone(),
                false,
                args.toolchain.as_deref(),
                args.timings,
                run_test_extra_args,
            )?;
            commands.push(command);
        }
    } else if test_arg.is_some() {
        bail!("Test not found or unsupported for the given chip")
    } else {
        for test in tests {
            let command = crate::generate_build_command(
                &package_path,
                args.chip,
                &target,
                &test,
                action.clone(),
                false,
                args.toolchain.as_deref(),
                args.timings,
                run_test_extra_args,
            )?;
            commands.push(command);
        }
    }
    let mut failed = Vec::new();

    for c in commands.build(false) {
        let repeat = if matches!(action, CargoAction::Run) {
            args.repeat
        } else {
            1
        };

        println!(
            "Command: cargo {}",
            c.command.join(" ").replace("---", "\n    ---")
        );
        for i in 0..repeat {
            if repeat != 1 {
                log::info!("Run {}/{}", i + 1, repeat);
            }
            if c.run(false).is_err() {
                failed.push(c.artifact_name.clone());
            }
        }
    }

    move_artifacts(args.chip, &action);

    if !failed.is_empty() {
        bail!("Failed tests: {:#?}", failed);
    }

    Ok(())
}

/// Run the HIL "full" workflow: build each matching test with crosscheck
/// enabled, flash + drain RTT tensor frames into a temp dir, then invoke the
/// matching host crosscheck (`tests/<name>_crosscheck.rs`) if present.
pub fn full_tests(workspace: &Path, args: TestsArgs) -> Result<()> {
    let (test_arg, filter) = if let Some(test_arg) = args.test.as_deref() {
        match test_arg.split_once("::") {
            Some((test, filter)) => (Some(test.to_string()), Some(filter.to_string())),
            None => (Some(test_arg.to_string()), None),
        }
    } else {
        (None, None)
    };

    let package_path = crate::windows_safe_path(&workspace.join("hil-test"));
    let target = Package::HilTest.target_triple(&args.chip)?;

    let mut tests = crate::firmware::load(&package_path.join("src").join("bin"))?
        .into_iter()
        .filter(|t| t.supports_chip(args.chip))
        .collect::<Vec<_>>();
    tests.sort_by_key(|a| a.binary_name());

    let selected: Vec<_> = match test_arg.as_deref() {
        Some(arg) => tests.into_iter().filter(|t| t.matches(Some(arg))).collect(),
        None => tests,
    };

    if selected.is_empty() {
        if test_arg.is_some() {
            bail!("Test not found or unsupported for the given chip");
        } else {
            bail!("No HIL tests found");
        }
    }

    let mut failed = Vec::new();

    for test in &selected {
        log::info!(
            "full-tests: running '{}' on {}",
            test.binary_name(),
            args.chip
        );

        // `crosscheck` is required for the channel-1 tensor stream the probe
        // driver drains. Inject it here so plain `run tests` doesn't pull in
        // the binary RTT channel and garble the terminal.
        let mut test_with_dump = test.clone();
        test_with_dump.add_features(["crosscheck"]);

        // Build the test ELF. We use Build(None) so cargo lands the artifact at
        // the standard path (`target/<triple>/release/<bin>`), which we then
        // hand to the probe driver.
        let cmd = crate::generate_build_command(
            &package_path,
            args.chip,
            &target,
            &test_with_dump,
            CargoAction::Build(None),
            false,
            args.toolchain.as_deref(),
            args.timings,
            &[],
        )?;

        let built = CargoCommandBatcher::build_one_for_cargo(&cmd);
        println!(
            "Command: cargo {}",
            built.command.join(" ").replace("---", "\n    ---")
        );
        if let Err(e) = built.run(false) {
            log::error!("build failed for '{}': {e}", test.binary_name());
            failed.push(test.binary_name());
            continue;
        }

        let elf = workspace
            .join("target")
            .join(&target)
            .join("release")
            .join(test.binary_name());

        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or_default();
        let dump_dir = std::env::temp_dir().join(format!(
            "edgedl-dumps-{}-{}",
            test.binary_name(),
            ts
        ));

        if let Err(e) = crate::probe::run(&elf, filter.as_deref(), Some(&dump_dir)) {
            log::error!("probe driver failed for '{}': {e:?}", test.binary_name());
            println!("\n==> dump directory (partial): {}\n", dump_dir.display());
            failed.push(test.binary_name());
            continue;
        }

        let crosscheck_file = workspace
            .join("tests")
            .join(format!("{}_crosscheck.rs", test.binary_name()));

        if crosscheck_file.exists() {
            log::info!(
                "full-tests: running host crosscheck 'tests/{}_crosscheck.rs'",
                test.binary_name()
            );
            if let Err(e) = run_host_crosscheck(workspace, &test.binary_name(), &dump_dir) {
                log::error!("host crosscheck failed for '{}': {e}", test.binary_name());
                println!("\n==> dump directory: {}\n", dump_dir.display());
                failed.push(format!("{}_crosscheck", test.binary_name()));
                continue;
            }
            println!("\n==> dump directory: {}\n", dump_dir.display());
        } else {
            println!(
                "\n==> no sibling host crosscheck at {} — skipping host step",
                crosscheck_file.display()
            );
            println!("==> dump directory: {}\n", dump_dir.display());
        }
    }

    if !failed.is_empty() {
        bail!("Failed HIL steps: {:#?}", failed);
    }

    Ok(())
}

/// Invoke `cargo test --test <bin>_crosscheck -- --ignored --nocapture` at the
/// workspace root with `EDGEDL_DUMP_DIR` set. The crosscheck test is expected
/// to read the env var and fail if the dump can't be reconciled.
fn run_host_crosscheck(workspace: &Path, bin_name: &str, dump_dir: &Path) -> Result<()> {
    let crosscheck_target = format!("{bin_name}_crosscheck");

    let mut cmd = Command::new("cargo");
    cmd.arg("test")
        .arg("--release")
        .arg("--features=std,crosscheck")
        .arg("--test")
        .arg(&crosscheck_target)
        .arg("--")
        .arg("--ignored")
        .arg("--nocapture")
        .current_dir(workspace)
        .env("EDGEDL_DUMP_DIR", dump_dir);

    log::debug!("running host crosscheck: {cmd:?}");

    let status = cmd.status().context("failed to invoke cargo test")?;
    if !status.success() {
        bail!("host crosscheck '{crosscheck_target}' exited with {status}");
    }
    Ok(())
}

fn move_artifacts(chip: Chip, action: &CargoAction) {
    if let CargoAction::Build(Some(out_dir)) = action {
        // Move binaries
        let from = out_dir.join("tmp");
        let to = out_dir.join(chip.to_string());
        std::fs::create_dir_all(&to).unwrap();

        // Binaries are in nested folders. There is one file in each folder. The name of the
        // final binary should be the name of the source binary's parent folder.
        for dir_entry in std::fs::read_dir(&from).unwrap() {
            let dir = dir_entry.unwrap();
            let mut bin_folder = std::fs::read_dir(dir.path()).unwrap();
            let file = bin_folder
                .next()
                .expect("No binary found")
                .expect("Failed to read entry");
            assert!(
                bin_folder.next().is_none(),
                "Only one binary should be present in each folder"
            );
            let source_file = file.path();
            let dest = to.join(dir.path().file_name().unwrap().to_string_lossy().as_ref());
            std::fs::rename(source_file, dest).unwrap();
        }
        // Clean up
        std::fs::remove_dir_all(from).unwrap();
    }
}
