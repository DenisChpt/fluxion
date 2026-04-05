use std::collections::HashMap;
use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde::Deserialize;

// ── Build configuration schema ──────────────────────────────────────

#[derive(Deserialize)]
struct BuildConfig {
	compiler: CompilerConfig,
	kernels: KernelConfig,
	target: TargetConfig,
	link: LinkConfig,
}

#[derive(Deserialize)]
struct CompilerConfig {
	binary: String,
	std: String,
	opt_level: u8,
	extra_flags: Vec<String>,
	#[serde(default)]
	defines: HashMap<String, bool>,
	#[serde(default)]
	includes: HashMap<String, String>,
}

#[derive(Deserialize)]
struct KernelConfig {
	dir: String,
	extension: String,
	sources: Vec<String>,
}

#[derive(Deserialize)]
struct TargetConfig {
	fallback_arches: Vec<String>,
	detect_command: String,
	detect_prefix: String,
}

#[derive(Deserialize)]
struct LinkConfig {
	lib: String,
	rocm_default_path: String,
	#[serde(default)]
	extra_lib_dirs: Vec<String>,
}

// ── Main ────────────────────────────────────────────────────────────

fn main() {
	let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
	let config = load_config(manifest_dir);

	let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
	let kernel_dir = manifest_dir.join(&config.kernels.dir);
	let arches = resolve_arches(&config.target);

	compile_kernels(&config, &kernel_dir, &out_dir, &arches);
	emit_link_flags(&config.link);
}

// ── Config loading ──────────────────────────────────────────────────

fn load_config(manifest_dir: &Path) -> BuildConfig {
	let config_path = manifest_dir.join("build.toml");
	println!("cargo::rerun-if-changed={}", config_path.display());

	let contents = std::fs::read_to_string(&config_path).unwrap_or_else(
		|e| panic!("failed to read {}: {e}", config_path.display()),
	);

	toml::from_str(&contents).unwrap_or_else(|e| {
		panic!("invalid {}: {e}", config_path.display())
	})
}

// ── Architecture detection ──────────────────────────────────────────

fn resolve_arches(target: &TargetConfig) -> Vec<String> {
	if let Ok(env_arches) = env::var("FLUXION_HIP_ARCH") {
		return env_arches.split(',').map(String::from).collect();
	}

	if let Some(detected) = detect_gpu_arch(target) {
		return detected;
	}

	target.fallback_arches.clone()
}

fn detect_gpu_arch(target: &TargetConfig) -> Option<Vec<String>> {
	let output = Command::new(&target.detect_command)
		.output()
		.ok()
		.filter(|o| o.status.success())?;

	let text = String::from_utf8_lossy(&output.stdout);
	let mut arches: Vec<String> = text
		.lines()
		.filter_map(|line| {
			let trimmed = line.trim();
			if trimmed.starts_with("Name:")
				&& trimmed.contains(&target.detect_prefix)
			{
				trimmed.split_whitespace().last().map(String::from)
			} else {
				None
			}
		})
		.filter(|a| {
			a.starts_with(&target.detect_prefix) && !a.contains("generic")
		})
		.collect();

	arches.sort();
	arches.dedup();

	if arches.is_empty() { None } else { Some(arches) }
}

// ── Kernel compilation ──────────────────────────────────────────────

fn compile_kernels(
	config: &BuildConfig,
	kernel_dir: &Path,
	out_dir: &Path,
	arches: &[String],
) {
	let cc = &config.compiler;

	for name in &config.kernels.sources {
		let src =
			kernel_dir.join(format!("{name}{}", config.kernels.extension));
		let out = out_dir.join(format!("{name}.co"));

		let mut cmd = Command::new(&cc.binary);

		for flag in &cc.extra_flags {
			cmd.arg(flag);
		}

		for arch in arches {
			cmd.arg(format!("--offload-arch={arch}"));
		}

		cmd.arg(format!("-O{}", cc.opt_level));
		cmd.arg(format!("-std={}", cc.std));

		for (key, path) in &cc.includes {
			let p = Path::new(path);
			let abs = if p.is_absolute() {
				p.to_path_buf()
			} else {
				kernel_dir.join(p)
			};
			if abs.exists() {
				cmd.arg(format!("-I{}", abs.display()));
			} else {
				eprintln!(
					"cargo:warning=include path '{key}' not found: {}",
					abs.display()
				);
			}
		}

		for (define, enabled) in &cc.defines {
			if *enabled {
				cmd.arg(format!("-D{define}"));
			}
		}

		cmd.arg("-o").arg(&out).arg(&src);

		let status = cmd.status().unwrap_or_else(|e| {
			panic!(
				"failed to run {} — is ROCm installed? ({e})",
				cc.binary
			)
		});

		assert!(
			status.success(),
			"{} failed to compile {name}{}",
			cc.binary,
			config.kernels.extension,
		);

		println!("cargo::rerun-if-changed={}", src.display());
	}
}

// ── Link flags ──────────────────────────────────────────────────────

fn emit_link_flags(link: &LinkConfig) {
	let rocm_path = env::var("ROCM_PATH")
		.unwrap_or_else(|_| link.rocm_default_path.clone());

	let lib_dir = format!("{rocm_path}/lib");
	if Path::new(&lib_dir).exists() {
		println!("cargo::rustc-link-search={lib_dir}");
	}

	for dir in &link.extra_lib_dirs {
		if Path::new(dir).exists() {
			println!("cargo::rustc-link-search={dir}");
		}
	}

	println!("cargo::rustc-link-lib={}", link.lib);
}
