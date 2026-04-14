use std::{path::Path, time::Instant};

use anyhow::{Context, Result, bail};
use clap::{Args, Parser};
use esp_metadata::{Chip, Config};
use strum::IntoEnumIterator;
use xtask::{
    Package,
    cargo::{CargoAction, CargoArgsBuilder, CargoCommandBatcher},
    commands::*,
    supported_chips,
};

// ----------------------------------------------------------------------------
// Command-line Interface

#[derive(Debug, Parser)]
enum Cli {
    /// Build-related subcommands
    #[clap(subcommand)]
    Build(Build),
    /// Run-related subcommands
    #[clap(subcommand)]
    Run(Run),
    /// Release-related subcommands
    #[clap(subcommand)]
    Release(Release),

    /// Perform (parts of) the checks done in CI
    Ci(CiArgs),
    /// Format all packages in the workspace with rustfmt
    #[clap(alias = "format-packages")]
    FmtPackages(FmtPackagesArgs),
    /// Run cargo clean
    Clean(CleanArgs),
    /// Check all packages in the workspace with cargo check
    CheckPackages(CheckPackagesArgs),
    /// Lint all packages in the workspace with clippy
    LintPackages(LintPackagesArgs),
    /// Check the changelog for packages.
    CheckChangelog(CheckChangelogArgs),
    /// Run host-tests in the workspace with `cargo test`
    HostTests(HostTestsArgs),
}

#[derive(Debug, Args)]
struct CiArgs {
    /// Chip to target.
    #[arg(value_enum)]
    chip: Chip,

    /// The toolchain used to run the lints
    #[arg(long)]
    toolchain: Option<String>,

    /// Steps to run in the CI pipeline.
    #[arg(long, value_delimiter = ',')]
    steps: Vec<String>,

    /// Whether to skip running lints
    #[arg(long)]
    no_lint: bool,

    /// Whether to skip checking the crates itself
    #[arg(long)]
    no_check_crates: bool,
}

#[derive(Debug, Args)]
struct FmtPackagesArgs {
    /// Run in 'check' mode; exists with 0 if formatted correctly, 1 otherwise
    #[arg(long)]
    check: bool,

    /// Package(s) to target.
    #[arg(value_enum, default_values_t = Package::iter())]
    packages: Vec<Package>,
}

#[derive(Debug, Args)]
struct CleanArgs {
    /// Package(s) to target.
    #[arg(value_enum, default_values_t = Package::iter())]
    packages: Vec<Package>,
}

#[derive(Debug, Args)]
struct HostTestsArgs {
    /// Package(s) to target.
    #[arg(value_enum, default_values_t = Package::iter())]
    packages: Vec<Package>,
}

#[derive(Debug, Args)]
struct CheckPackagesArgs {
    /// Package(s) to target.
    #[arg(value_enum, default_values_t = Package::iter())]
    packages: Vec<Package>,

    /// Check for a specific chip
    #[arg(long, value_enum, value_delimiter = ',', default_values_t = supported_chips())]
    chips: Vec<Chip>,

    /// The toolchain used to run the checks
    #[arg(long)]
    toolchain: Option<String>,
}

#[derive(Debug, Args)]
struct LintPackagesArgs {
    /// Package(s) to target.
    #[arg(value_enum, default_values_t = Package::iter())]
    packages: Vec<Package>,

    /// Lint for a specific chip
    #[arg(long, value_enum, value_delimiter = ',', default_values_t = supported_chips())]
    chips: Vec<Chip>,

    /// Automatically apply fixes
    #[arg(long)]
    fix: bool,

    /// The toolchain used to run the lints
    #[arg(long)]
    toolchain: Option<String>,
}

#[derive(Debug, Args)]
struct CheckChangelogArgs {
    /// Package(s) to tag.
    #[arg(long, value_enum, value_delimiter = ',', default_values_t = Package::iter())]
    packages: Vec<Package>,

    /// Re-generate the changelog with consistent formatting.
    #[arg(long)]
    normalize: bool,
}

// ----------------------------------------------------------------------------
// Application

fn main() -> Result<()> {
    let mut builder =
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"));
    builder.target(env_logger::Target::Stdout);
    builder.init();

    let workspace =
        std::env::current_dir().with_context(|| format!("Failed to get the current dir!"))?;
    let target_path = workspace.join("target");

    if std::env::var("CARGO_TARGET_DIR").is_err() {
        unsafe { std::env::set_var("CARGO_TARGET_DIR", target_path.to_str().unwrap()) };
    }

    match Cli::parse() {
        // Build-related subcommands:
        Cli::Build(build) => match build {
            Build::Examples(args) => examples(&workspace, args, CargoAction::Build(None)),
            Build::Package(args) => build_package(&workspace, args),
            Build::Tests(args) => tests(
                &workspace,
                args,
                CargoAction::Build(Some(target_path.join("tests"))),
            ),
        },

        // Run-related subcommands:
        Cli::Run(run) => match run {
            Run::Example(args) => examples(&workspace, args, CargoAction::Run),
            Run::Tests(args) => tests(&workspace, args, CargoAction::Run),
        },

        // Release-related subcommands:
        Cli::Release(release) => match release {
            Release::BumpVersion(args) => bump_version(&workspace, args),
            Release::TagReleases(args) => tag_releases(&workspace, args),
            Release::Publish(args) => publish(&workspace, args),
        },

        Cli::Ci(args) => run_ci_checks(&workspace, args),
        Cli::FmtPackages(args) => fmt_packages(&workspace, args),
        Cli::Clean(args) => clean(&workspace, args),
        Cli::CheckPackages(args) => check_packages(&workspace, args),
        Cli::LintPackages(args) => lint_packages(&workspace, args),
        Cli::CheckChangelog(args) => check_changelog(&workspace, &args.packages, args.normalize),
        Cli::HostTests(args) => host_tests(&workspace, args),
    }
}

// ----------------------------------------------------------------------------
// Subcommands

fn fmt_packages(workspace: &Path, args: FmtPackagesArgs) -> Result<()> {
    let mut packages = args.packages;
    packages.sort();

    for package in packages {
        xtask::format_package(workspace, package, args.check, None)?;
    }

    // Format ymls in .github/
    xtask::format_yml(args.check, "./.github")?;

    // Format Cargo.toml files with taplo
    xtask::format_toml(workspace, args.check)?;

    Ok(())
}

fn clean(workspace: &Path, args: CleanArgs) -> Result<()> {
    let mut packages = args.packages;
    packages.sort();

    for package in packages {
        log::info!("Cleaning package: {}", package);
        let path = workspace.join(package.dir_name());

        let cargo_args = CargoArgsBuilder::default()
            .subcommand("clean")
            .arg("--target-dir")
            .arg(path.join("target").display().to_string())
            .build();

        xtask::cargo::run(&cargo_args, &path).with_context(|| {
            format!(
                "Failed to run `cargo run` with {cargo_args:?} in {}",
                path.display()
            )
        })?;
    }

    Ok(())
}

fn check_packages(workspace: &Path, args: CheckPackagesArgs) -> Result<()> {
    log::debug!("Checking packages: {:?}", args.packages);
    let mut packages = args.packages;
    packages.sort();

    let mut commands = CargoCommandBatcher::new();

    for package in packages.iter().filter(|p| p.is_published()) {
        // Unfortunately each package has its own unique requirements for
        // building, so we need to handle each individually (though there
        // is *some* overlap)
        for chip in &args.chips {
            log::debug!("  for chip: {}", chip);
            let device = Config::for_chip(chip);

            if let Err(e) = package.validate_package_chip(chip) {
                log::warn!("{e}. Skipping");
                continue;
            }

            for mut features in package.check_feature_rules(device) {
                if package.has_chip_features() {
                    features.push(device.name())
                }

                commands.push(build_check_package_command(
                    workspace,
                    *package,
                    chip,
                    &["--no-default-features"],
                    &features,
                    args.toolchain.as_deref(),
                )?);
            }
        }
    }

    for c in commands.build(false) {
        println!(
            "Command: cargo {}",
            c.command.join(" ").replace("---", "\n    ---")
        );
        c.run(false)?;
    }

    Ok(())
}

fn build_check_package_command(
    workspace: &Path,
    package: Package,
    chip: &Chip,
    args: &[&str],
    features: &[String],
    mut toolchain: Option<&str>,
) -> Result<CargoArgsBuilder> {
    log::info!(
        "Checking package: {} ({}, features: {:?})",
        package,
        chip,
        features
    );

    let path = workspace.join(package.dir_name());

    let mut builder = CargoArgsBuilder::default()
        .subcommand("check")
        .manifest_path(path.join("Cargo.toml"));

    if !package.build_on_host(features) {
        if chip.is_xtensa() {
            // In case the user doesn't specify a toolchain, make sure we use +esp
            toolchain.get_or_insert("esp");
        }
        builder = builder.target(package.target_triple(chip)?);
    }

    if let Some(toolchain) = toolchain {
        if !package.build_on_host(features) && toolchain.starts_with("esp") {
            builder = builder.config("-Zbuild-std=core,alloc");
        }
        builder = builder.toolchain(toolchain);
    }

    builder = builder.args(&args);

    if !features.is_empty() {
        builder = builder.arg(format!("--features={}", features.join(",")));
    }

    // TODO: these should come from the outside
    builder.add_env_var("CI", "1");
    builder.add_env_var("DEFMT_LOG", "trace");
    builder.add_env_var("ESP_LOG", "trace");

    Ok(builder)
}

fn lint_packages(workspace: &Path, args: LintPackagesArgs) -> Result<()> {
    log::debug!("Linting packages: {:?}", args.packages);
    let mut packages = args.packages;
    packages.sort();

    for package in packages.iter().filter(|p| p.is_published()) {
        // Unfortunately each package has its own unique requirements for
        // building, so we need to handle each individually (though there
        // is *some* overlap)
        for chip in &args.chips {
            log::debug!("  for chip: {}", chip);
            let device = Config::for_chip(chip);

            if let Err(e) = package.validate_package_chip(chip) {
                log::warn!("{e}. Skipping");
                continue;
            }

            for mut features in package.lint_feature_rules(device) {
                if package.has_chip_features() {
                    features.push(device.name())
                }

                lint_package(
                    workspace,
                    *package,
                    chip,
                    &["--no-default-features"],
                    &features,
                    args.fix,
                    args.toolchain.as_deref(),
                )?;
            }
        }
    }

    Ok(())
}

fn lint_package(
    workspace: &Path,
    package: Package,
    chip: &Chip,
    args: &[&str],
    features: &[String],
    fix: bool,
    mut toolchain: Option<&str>,
) -> Result<()> {
    log::info!(
        "Linting package: {} ({}, features: {:?})",
        package,
        chip,
        features
    );

    let path = workspace.join(package.dir_name());

    let mut builder = CargoArgsBuilder::default().subcommand("clippy");

    if !package.build_on_host(features) {
        if chip.is_xtensa() {
            // In case the user doesn't specify a toolchain, make sure we use +esp
            toolchain.get_or_insert("esp");
        }
        builder = builder.target(package.target_triple(chip)?);
    }

    if let Some(toolchain) = toolchain {
        if !package.build_on_host(features) && toolchain.starts_with("esp") {
            builder = builder.config("-Zbuild-std=core,alloc");
        }
        builder = builder.toolchain(toolchain);
    }

    for arg in args {
        builder = builder.arg(arg.to_string());
    }

    if !features.is_empty() {
        builder = builder.arg(format!("--features={}", features.join(",")));
    }

    let builder = if fix {
        builder.arg("--fix").arg("--lib").arg("--allow-dirty")
    } else {
        builder.arg("--").arg("-D").arg("warnings").arg("--no-deps")
    };

    let cargo_args = builder.build();

    xtask::cargo::run_with_env(
        &cargo_args,
        &path,
        [("CI", "1"), ("DEFMT_LOG", "trace")],
        false,
    )
    .with_context(|| {
        format!(
            "Failed to run `cargo run` with {args:?} `CI, `1`, `DEFMT_LOG`, and `trace` envs in {}",
            path.display()
        )
    })?;

    Ok(())
}

struct Runner {
    failed: Vec<&'static str>,
    started_at: Instant,
    skip_steps: Vec<String>,
    run_steps: Vec<String>,
    steps_executed: usize,
}

impl Runner {
    fn new(options: &CiArgs) -> Self {
        Self {
            failed: Vec::new(),
            started_at: Instant::now(),
            skip_steps: {
                let mut skip = vec![];
                if options.no_lint {
                    skip.push(String::from("lint"));
                }
                if options.no_check_crates {
                    skip.push(String::from("check"));
                }
                skip
            },
            run_steps: options.steps.clone(),
            steps_executed: 0,
        }
    }

    fn run(&mut self, id: &str, group: &'static str, op: impl FnOnce() -> Result<()>) {
        if self.skip_steps.iter().any(|s| s == id) {
            log::debug!("{group} skipped by user request");
            return;
        }
        if !self.run_steps.is_empty() && !self.run_steps.iter().any(|s| s == id) {
            log::debug!("{group} skipped by user request");
            return;
        }

        self.steps_executed += 1;

        // Output grouped logs
        // https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-commands#grouping-log-lines
        println!("::group::{group}");
        if let Err(e) = op() {
            log::error!("{group} failed: {e:?}");
            self.failed.push(group);
        } else {
            log::debug!("{group} succeeded");
        }
        println!("::endgroup::");
    }

    fn finish(self) -> Result<()> {
        fn write_summary(message: &str) {
            if let Some(summary_file) = std::env::var_os("GITHUB_STEP_SUMMARY") {
                std::fs::write(summary_file, message).unwrap();
            }
        }

        let expected_to_run = self
            .run_steps
            .iter()
            .filter(|s| !self.skip_steps.contains(s))
            .cloned()
            .collect::<Vec<_>>();
        if self.steps_executed == 0 && !expected_to_run.is_empty() {
            bail!(
                "The following steps were requested but not executed: {}. Perhaps they contain typos?",
                expected_to_run.join(", ")
            );
        }

        log::info!("CI checks completed in {:?}", self.started_at.elapsed());

        if !self.failed.is_empty() {
            let mut summary = String::new();
            summary.push_str("# Summary of failed CI checks\n");
            for failed_check in self.failed {
                summary.push_str(&format!("* {failed_check}\n"));
            }
            println!("{summary}");
            write_summary(&summary);
            bail!("CI checks failed");
        }

        Ok(())
    }
}

fn run_ci_checks(workspace: &Path, args: CiArgs) -> Result<()> {
    log::info!("Running CI checks for chip: {}", args.chip);
    println!("::add-matcher::.github/rust-matchers.json");

    let mut runner = Runner::new(&args);

    unsafe {
        std::env::set_var("CI", "true");
    }

    runner.run("fmt", "Check formatting", || {
        fmt_packages(
            workspace,
            FmtPackagesArgs {
                check: true,
                packages: Package::iter().collect(),
            },
        )
    });

    runner.run("check", "Check crates", || {
        check_packages(
            workspace,
            CheckPackagesArgs {
                packages: Package::iter().collect(),
                chips: vec![args.chip],
                toolchain: args.toolchain.clone(),
            },
        )
    });

    runner.run("lint", "Lint", || {
        lint_packages(
            workspace,
            LintPackagesArgs {
                packages: Package::iter().collect(),
                chips: vec![args.chip],
                fix: false,
                toolchain: args.toolchain.clone(),
            },
        )
    });

    runner.run("examples", "Build examples", || {
        examples(
            workspace,
            ExamplesArgs {
                package: Package::Examples,
                chip: Some(args.chip),
                example: Some("all".to_string()),
                debug: true,
                toolchain: args.toolchain.clone(),
                timings: false,
            },
            CargoAction::Build(None),
        )
    });

    runner.finish()
}

fn host_tests(workspace: &Path, args: HostTestsArgs) -> Result<()> {
    let mut packages = args.packages;
    packages.sort();

    for package in packages {
        log::debug!("Running host-tests for package: {}", package);
        if package.has_host_tests(workspace) {
            xtask::run_host_tests(workspace, package)?;
        }
    }

    Ok(())
}
