use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, anyhow};
use cargo::CargoAction;
use esp_metadata::{Chip, Config};
use parking_lot::{MappedMutexGuard, Mutex, MutexGuard};
use pretty_yaml::{config::FormatOptions, format_text};
use serde::{Deserialize, Serialize};
use toml_edit::Item;
use walkdir::WalkDir;

/// Chips supported by edgedl.
/// Add new chips here as support is implemented.
pub const SUPPORTED_CHIPS: &[Chip] = &[Chip::Esp32s3];

/// Returns the list of supported chips.
pub fn supported_chips() -> Vec<Chip> {
    SUPPORTED_CHIPS.to_vec()
}

use crate::{
    cargo::{CargoArgsBuilder, CargoCommandBatcher, CargoToml},
    firmware::Metadata,
};

pub mod cargo;
pub mod changelog;
pub mod commands;
pub mod firmware;
pub mod git;

#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    clap::ValueEnum,
    strum::Display,
    strum::EnumIter,
    strum::AsRefStr,
    serde::Deserialize,
    serde::Serialize,
)]
#[serde(rename_all = "kebab-case")]
#[strum(serialize_all = "kebab-case")]
/// Represents the packages in the edgedl workspace.
pub enum Package {
    /// Main edgedl runtime crate
    Edgedl,
    /// Proc-macro crate for model binding
    EdgedlMacros,
    /// Examples directory
    Examples,
    /// Hardware-in-loop tests
    HilTest,
}

impl Package {
    /// Does the package have chip-specific cargo features?
    pub fn has_chip_features(&self) -> bool {
        use strum::IntoEnumIterator;

        if *self == Package::Examples {
            return true;
        }

        let chips = Chip::iter()
            .map(|chip| chip.to_string())
            .collect::<Vec<_>>();
        let toml = self.toml();
        let Some(Item::Table(features)) = toml.manifest.get("features") else {
            return false;
        };

        features
            .iter()
            .any(|(feature, _)| chips.iter().any(|c| c == feature))
    }

    /// Does the package have inline assembly?
    pub fn has_inline_assembly(&self, workspace: &Path) -> bool {
        // edgedl uses inline assembly for SIMD kernels
        if *self == Package::Edgedl {
            return true;
        }

        let lib_rs_path = workspace.join(self.to_string()).join("src").join("lib.rs");
        let Ok(source) = std::fs::read_to_string(&lib_rs_path) else {
            return false;
        };

        source
            .lines()
            .filter(|line| line.starts_with("#!["))
            .any(|line| line.contains("asm_experimental_arch"))
    }

    /// Does the package have any host tests?
    pub fn has_host_tests(&self, workspace: &Path) -> bool {
        if *self == Package::HilTest || *self == Package::Examples {
            return false;
        }
        let package_path = workspace.join(self.dir_name());

        // Check both src/ (unit tests) and tests/ (integration tests)
        let dirs_to_check = [package_path.join("src"), package_path.join("tests")];

        dirs_to_check.iter().any(|dir| {
            dir.exists()
                && walkdir::WalkDir::new(dir)
                    .into_iter()
                    .filter_map(Result::ok)
                    .filter(|e| e.path().extension().is_some_and(|ext| ext == "rs"))
                    .any(|entry| {
                        std::fs::read_to_string(entry.path())
                            .is_ok_and(|src| src.contains("#[test]"))
                    })
        })
    }

    /// Does the package need to be built with build-std?
    pub fn needs_build_std(&self) -> bool {
        // All edgedl packages target no_std
        true
    }

    /// Do the package's chip-specific cargo features affect the public API?
    pub fn chip_features_matter(&self) -> bool {
        matches!(self, Package::Edgedl | Package::Examples | Package::HilTest)
    }

    /// Should the package be published?
    pub fn is_published(&self) -> bool {
        match self {
            Package::Edgedl | Package::EdgedlMacros => true,
            Package::Examples | Package::HilTest => false,
        }
    }

    /// Build on host (without cross-compilation)
    pub fn build_on_host(&self, _features: &[String]) -> bool {
        // edgedl can run tests on host with std feature
        matches!(self, Package::Edgedl)
    }

    /// Given a device config, return the features which should be enabled for
    /// this package.
    pub fn doc_feature_rules(&self, _config: &Config) -> Vec<String> {
        vec![]
    }

    /// Additional feature rules to test subsets of features for a package.
    pub fn check_feature_rules(&self, _config: &Config) -> Vec<Vec<String>> {
        vec![vec![]]
    }

    /// Additional feature rules to test subsets of features for a package.
    pub fn lint_feature_rules(&self, _config: &Config) -> Vec<Vec<String>> {
        vec![vec![]]
    }

    fn toml(&self) -> MappedMutexGuard<'_, CargoToml> {
        static TOML: Mutex<Option<HashMap<Package, CargoToml>>> = Mutex::new(None);

        let tomls = TOML.lock();

        MutexGuard::map(tomls, |tomls| {
            let tomls = tomls.get_or_insert_default();

            tomls.entry(*self).or_insert_with(|| {
                CargoToml::new(&std::env::current_dir().unwrap(), *self)
                    .expect("Failed to parse Cargo.toml")
            })
        })
    }

    /// Return the target triple for a given package/chip pair.
    pub fn target_triple(&self, chip: &Chip) -> Result<String> {
        Ok(chip.target())
    }

    /// Validate that the specified chip is valid for the specified package.
    pub fn validate_package_chip(&self, chip: &Chip) -> Result<()> {
        if *self == Package::Examples || *self == Package::HilTest {
            // Examples and HIL tests validate via their Cargo.toml features
            return Ok(());
        }

        // edgedl SIMD only supports ESP32-S3
        if *self == Package::Edgedl && *chip != Chip::Esp32s3 {
            return Err(anyhow!(
                "Package 'edgedl' SIMD features only support ESP32-S3"
            ));
        }

        Ok(())
    }

    /// Creates a tag string for this [`Package`] combined with a semantic version.
    pub fn tag(&self, version: &semver::Version) -> String {
        log::debug!(
            "Creating tag for package '{}' with version '{}'",
            self,
            version
        );
        format!("{self}-v{version}")
    }
}

// Custom Display for package paths that differ from enum name
impl Package {
    /// Get the directory name for this package
    pub fn dir_name(&self) -> &'static str {
        match self {
            Package::Edgedl => ".",
            Package::EdgedlMacros => "macros",
            Package::Examples => "examples",
            Package::HilTest => "hil-test",
        }
    }
}

#[derive(Debug, Clone, Copy, strum::Display, clap::ValueEnum, Serialize, Deserialize)]
#[strum(serialize_all = "lowercase")]
/// Represents the versioning scheme for a package.
pub enum Version {
    Major,
    Minor,
    Patch,
}

/// Run or build the specified test or example for the specified chip.
pub fn execute_app(
    package_path: &Path,
    chip: Chip,
    target: &str,
    app: &Metadata,
    action: CargoAction,
    debug: bool,
    toolchain: Option<&str>,
    timings: bool,
    extra_args: &[&str],
) -> Result<()> {
    let package = app.example_path().strip_prefix(package_path)?;
    log::info!("Building example '{}' for '{}'", package.display(), chip);

    let builder = generate_build_command(
        package_path,
        chip,
        target,
        app,
        action,
        debug,
        toolchain,
        timings,
        extra_args,
    )?;

    let command = CargoCommandBatcher::build_one_for_cargo(&builder);

    command.run(false)?;

    Ok(())
}

pub fn generate_build_command(
    package_path: &Path,
    chip: Chip,
    target: &str,
    app: &Metadata,
    action: CargoAction,
    debug: bool,
    toolchain: Option<&str>,
    timings: bool,
    extra_args: &[&str],
) -> Result<CargoArgsBuilder> {
    let package = app.example_path().strip_prefix(package_path)?;
    log::info!(
        "Building command: {} '{}' for '{}'",
        if matches!(action, CargoAction::Build(_)) {
            "Build"
        } else {
            "Run"
        },
        package.display(),
        chip
    );

    let mut features = app.feature_set().to_vec();
    if !features.is_empty() {
        log::info!("  Features:      {}", features.join(", "));
    }
    features.push(chip.to_string());

    let cwd = if package_path.ends_with("examples") {
        package_path.join(package).to_path_buf()
    } else {
        package_path.to_path_buf()
    };

    let mut builder = CargoArgsBuilder::new(app.output_file_name())
        .manifest_path(cwd.join("Cargo.toml"))
        .config_path(cwd.join(".cargo").join("config.toml"))
        .target(target)
        .features(&features);

    let subcommand = if matches!(action, CargoAction::Build(_)) {
        "build"
    } else {
        "run"
    };
    builder = builder.subcommand(subcommand);

    let bin_arg = if package.starts_with("src/bin") {
        Some(format!("--bin={}", app.binary_name()))
    } else if !package_path.ends_with("examples") {
        Some(format!("--example={}", app.binary_name()))
    } else {
        None
    };

    if let Some(arg) = bin_arg {
        builder.add_arg(arg);
    }

    if !app.configuration().is_empty() {
        log::info!("  Configuration: {}", app.configuration());
    }

    for config in app.cargo_config() {
        log::info!(" Cargo --config: {config}");
        builder.add_config("--config").add_config(config);
        // Some configuration requires nightly rust, so let's just assume it. May be
        // overwritten by the esp toolchain on xtensa.
        builder = builder.toolchain("nightly");
    }

    let env_vars = app.env_vars();
    for (key, value) in env_vars {
        log::info!("  esp-config:    {} = {}", key, value);
        builder.add_env_var(key, value);
    }

    if !debug {
        builder.add_arg("--release");
    }
    if timings {
        builder.add_arg("--timings");
    }

    let toolchain = match toolchain {
        // Preserve user choice
        Some(tc) => Some(tc),
        // If targeting an Xtensa device, we must use the '+esp' toolchain modifier:
        _ if target.starts_with("xtensa") => Some("esp"),
        _ => None,
    };
    if let Some(toolchain) = toolchain {
        builder = builder.toolchain(toolchain);
    }

    // Add -Zbuild-std for Xtensa targets (config.toml's [unstable] doesn't propagate correctly)
    if target.starts_with("xtensa") {
        builder.add_arg("-Zbuild-std=core,alloc");
    }

    if let CargoAction::Build(Some(out_dir)) = action {
        // We have to place the binary into a directory named after the app, because
        // we can't set the binary name.
        builder.add_arg("--artifact-dir");
        builder.add_arg(
            out_dir
                .join("tmp") // This will be deleted in one go
                .join(app.output_file_name()) // This sets the name of the binary
                .display()
                .to_string(),
        );
    }

    let builder = builder.args(extra_args);

    Ok(builder)
}

// ----------------------------------------------------------------------------
// Helper Functions

/// Copy an entire directory recursively.
// https://stackoverflow.com/a/65192210
pub fn copy_dir_all(src: impl AsRef<Path>, dst: impl AsRef<Path>) -> Result<()> {
    log::debug!(
        "Copying directory '{}' to '{}'",
        src.as_ref().display(),
        dst.as_ref().display()
    );
    fs::create_dir_all(&dst).with_context(|| "Failed to create a {dst}")?;

    for entry in fs::read_dir(src).with_context(|| "Failed to read {src}")? {
        let entry = entry?;
        let ty = entry.file_type()?;

        if ty.is_dir() {
            copy_dir_all(entry.path(), dst.as_ref().join(entry.file_name()))?;
        } else {
            fs::copy(entry.path(), dst.as_ref().join(entry.file_name()))?;
        }
    }

    Ok(())
}

/// Return a (sorted) list of paths to each valid Cargo package in the
/// workspace.
pub fn package_paths(workspace: &Path) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    for entry in fs::read_dir(workspace).context("Failed to read {workspace}")? {
        let entry = entry?;
        if entry.file_type()?.is_dir() && entry.path().join("Cargo.toml").exists() {
            paths.push(entry.path());
        }
    }

    paths.sort();

    log::debug!(
        "Found {} packages in workspace '{}':",
        paths.len(),
        workspace.display()
    );

    Ok(paths)
}

/// Parse the version from the specified package's Cargo manifest.
pub fn package_version(_workspace: &Path, package: Package) -> Result<semver::Version> {
    Ok(package.toml().package_version())
}

/// Make the path "Windows"-safe
pub fn windows_safe_path(path: &Path) -> PathBuf {
    PathBuf::from(path.to_str().unwrap().to_string().replace("\\\\?\\", ""))
}

/// Format the specified package in the workspace using `cargo fmt`.
pub fn format_package(
    workspace: &Path,
    package: Package,
    check: bool,
    format_rules: Option<&Path>,
) -> Result<()> {
    log::info!("Formatting package: {}", package);
    let package_path = workspace.join(package.dir_name());

    let paths = if package == Package::Examples {
        crate::find_packages(&package_path)?
    } else {
        vec![package_path]
    };

    for path in &paths {
        format_package_path(workspace, path, check, format_rules)?;
        format_yml(check, path)?;
    }

    Ok(())
}

/// Run the host tests for the specified package.
pub fn run_host_tests(workspace: &Path, package: Package) -> Result<()> {
    log::info!("Running host tests for package: {}", package);
    let package_path = workspace.join(package.dir_name());

    let cmd = CargoArgsBuilder::default();

    match package {
        Package::Edgedl => {
            // Run edgedl tests with std feature
            cargo::run(
                &cmd.clone()
                    .subcommand("test")
                    .features(&vec!["std".into()])
                    .build(),
                &package_path,
            )
        }
        _ => Err(anyhow!(
            "Instructions for host testing were not provided for: '{}'",
            package,
        )),
    }
}

/// Format a package directory in the workspace using `cargo fmt`.
pub fn format_package_path(
    workspace: &Path,
    package_path: &Path,
    check: bool,
    format_rules: Option<&Path>,
) -> Result<()> {
    let mut cargo_args = CargoArgsBuilder::default()
        .toolchain("nightly")
        .subcommand("fmt")
        .build();

    if check {
        cargo_args.push("--check".into());
    }

    // Find the rustfmt config file
    let mut config_file_path;
    let config_file = if let Some(rules) = format_rules {
        rules
    } else {
        config_file_path = package_path.join("rustfmt.toml");
        if !config_file_path.exists() {
            config_file_path = workspace.join("rustfmt.toml");
        }
        &config_file_path
    };

    cargo_args.push("--".into());
    cargo_args.push(format!("--config-path={}", config_file.display()));

    log::debug!("{cargo_args:#?}");

    cargo::run(&cargo_args, package_path)
}

/// Recursively format all `.yml` files in the specified directory.
pub fn format_yml<P: AsRef<Path>>(check: bool, path: P) -> Result<()> {
    WalkDir::new(path)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "yml"))
        .try_for_each(|entry| -> Result<()> {
            let path = entry.path();
            let content = fs::read_to_string(path)?;

            let formatted = format_text(&content, &FormatOptions::default())
                .context(format!("Failed to format {:?} yml!", path))?;

            if content.replace("\r\n", "\n") != formatted.replace("\r\n", "\n") {
                if check {
                    anyhow::bail!("File not formatted: {:?}", path);
                }

                log::info!("Fixing format: {:?}", path);
                fs::write(path, formatted)?;
            }

            Ok(())
        })?;

    Ok(())
}

/// Format all Cargo.toml files in the workspace using taplo.
pub fn format_toml(workspace: &Path, check: bool) -> Result<()> {
    log::info!("Formatting TOML files with taplo");

    let mut cmd = std::process::Command::new("taplo");
    cmd.arg("fmt");

    if check {
        cmd.arg("--check");
    }

    // Use workspace taplo.toml config
    let config_path = workspace.join("taplo.toml");
    if config_path.exists() {
        cmd.arg("--config").arg(&config_path);
    }

    // Format all Cargo.toml files (glob pattern as positional argument)
    cmd.arg("**/Cargo.toml");

    cmd.current_dir(workspace);

    log::debug!("Running: {:?}", cmd);

    let status = cmd
        .status()
        .context("Failed to run taplo. Is it installed? Run: cargo install taplo-cli")?;

    if !status.success() {
        if check {
            anyhow::bail!(
                "TOML files are not formatted correctly. Run `cargo xtask fmt-packages` to fix."
            );
        } else {
            anyhow::bail!("taplo formatting failed");
        }
    }

    Ok(())
}

/// Recursively find all packages in the given path that contain a `Cargo.toml` file.
pub fn find_packages(path: &Path) -> Result<Vec<PathBuf>> {
    let mut packages = Vec::new();

    for result in
        fs::read_dir(path).with_context(|| format!("Failed to read {}", path.display()))?
    {
        log::debug!("Inspecting path: {}", path.display());
        let entry = result?;
        if entry.path().is_file() {
            continue;
        }

        // Path is a directory:
        if entry.path().join("Cargo.toml").exists() {
            packages.push(entry.path());
        } else {
            packages.extend(find_packages(&entry.path())?);
        }
    }

    log::debug!(
        "Found {} packages in path '{}':",
        packages.len(),
        path.display()
    );

    Ok(packages)
}
