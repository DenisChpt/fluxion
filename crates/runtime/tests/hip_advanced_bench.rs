//! Advanced Fluxion HIP benchmarks:
//! - Anisotropic grids (dx != dy)
//! - Convection-diffusion (variable coeff, upwind)
//! - Naive vs LDS-tiled stencil
//! - Size sweep to GPU saturation
//! - Roofline (effective bandwidth)
//!
//! Run: cargo test --features hip --test hip_advanced_bench
//!      --release -- --nocapture

#![cfg(feature = "hip")]

use std::time::Instant;

use fluxion_core::{
	Backend, Boundaries, DType, Grid, Stencil,
};

fn hip_backend() -> &'static fluxion_backend_hip::HipBackend {
	use std::sync::OnceLock;
	static INSTANCE: OnceLock<fluxion_backend_hip::HipBackend> =
		OnceLock::new();
	INSTANCE.get_or_init(|| {
		fluxion_backend_hip::HipBackend::new().unwrap()
	})
}

fn gaussian(n: usize, h: f64) -> Vec<f64> {
	let sigma = 0.1_f64;
	(0..n * n)
		.map(|i| {
			let (r, c) = (i / n, i % n);
			let dx = c as f64 * h - 0.5;
			let dy = r as f64 * h - 0.5;
			(-dx.mul_add(dx, dy * dy)
				/ (2.0 * sigma * sigma))
			.exp()
		})
		.collect()
}

/// Force GPU sync by doing a reduction readback.
fn sync_buf(
	b: &fluxion_backend_hip::HipBackend,
	buf: &fluxion_backend_hip::HipBuffer,
) {
	let _ = b.norm_l2(buf);
}

// ── Test A: Anisotropic (dx != dy) ──────────────────

fn bench_aniso(n: usize, steps: usize) {
	let hx = 1.0 / (n - 1) as f64;
	let hy = 0.5 / (n - 1) as f64; // aspect 2:1
	let grid = Grid::new(n, n, hx, hy).unwrap();
	let stencil = Stencil::laplacian_2d_5pt(hx, hy);
	let bcs = Boundaries::zero_dirichlet();
	let alpha = 0.01_f64;
	let h_min = hx.min(hy);
	let dt = 0.4 * h_min * h_min / (4.0 * alpha);

	let b = hip_backend();
	let data = gaussian(n, hx);
	let mut u = b.upload_f64(&data).unwrap();
	let mut scratch = b.allocate(n * n, DType::F64).unwrap();

	// Warmup.
	for _ in 0..20 {
		b.apply_stencil(&u, &mut scratch, &grid, &stencil, &bcs)
			.unwrap();
		b.axpy(dt * alpha, &scratch, &mut u).unwrap();
	}
	sync_buf(b, &u);

	let start = Instant::now();
	for _ in 0..steps {
		b.apply_stencil(&u, &mut scratch, &grid, &stencil, &bcs)
			.unwrap();
		b.axpy(dt * alpha, &scratch, &mut u).unwrap();
	}
	sync_buf(b, &u);
	let elapsed = start.elapsed();

	let ms = elapsed.as_secs_f64() * 1000.0;
	println!(
		"  Fluxion Aniso   | {n:5}x{n:<5} | {steps:6} steps | \
		 {ms:8.1} ms | {:8.1} us/step",
		ms * 1000.0 / steps as f64
	);
}

// ── Test B: Convection-Diffusion ────────────────────

fn bench_conv_diff(n: usize, steps: usize) {
	let h = 1.0 / (n - 1) as f64;
	let kappa_val = 0.01_f64;
	let vx_val = 1.0_f64;
	let vy_val = 0.5_f64;
	let dt = (0.2 * h / vx_val.abs().max(vy_val.abs()))
		.min(0.4 * h * h / (4.0 * kappa_val));

	let b = hip_backend();
	let data = gaussian(n, h);
	let mut u = b.upload_f64(&data).unwrap();
	let mut output =
		b.allocate(n * n, DType::F64).unwrap();

	// Constant coefficient fields.
	let kappa_data = vec![kappa_val; n * n];
	let vx_data = vec![vx_val; n * n];
	let vy_data = vec![vy_val; n * n];
	let kappa = b.upload_f64(&kappa_data).unwrap();
	let vx = b.upload_f64(&vx_data).unwrap();
	let vy = b.upload_f64(&vy_data).unwrap();

	let inv_dx = 1.0 / h;
	let inv_dy = 1.0 / h;
	let inv_dx2 = inv_dx * inv_dx;
	let inv_dy2 = inv_dy * inv_dy;

	// Warmup.
	for _ in 0..20 {
		b.conv_diff_axpy(
			dt, &u, &kappa, &vx, &vy, &mut output,
			n as i32, n as i32,
			inv_dx, inv_dy, inv_dx2, inv_dy2,
		);
		b.copy(&output, &mut u).unwrap();
	}
	sync_buf(b, &u);

	// Use fused kernel: u += dt * L(u_copy).
	let mut u_copy =
		b.allocate(n * n, DType::F64).unwrap();

	let start = Instant::now();
	for _ in 0..steps {
		b.copy(&u, &mut u_copy).unwrap();
		b.conv_diff_axpy(
			dt, &u_copy, &kappa, &vx, &vy, &mut u,
			n as i32, n as i32,
			inv_dx, inv_dy, inv_dx2, inv_dy2,
		);
	}
	sync_buf(b, &u);
	let elapsed = start.elapsed();

	let ms = elapsed.as_secs_f64() * 1000.0;
	println!(
		"  Fluxion ConvDiff| {n:5}x{n:<5} | {steps:6} steps | \
		 {ms:8.1} ms | {:8.1} us/step",
		ms * 1000.0 / steps as f64
	);
}

// ── Test C: Naive vs tiled stencil ──────────────────

fn bench_naive_vs_tiled(n: usize, iters: usize) {
	let h = 1.0 / (n - 1) as f64;
	let inv_dx2 = 1.0 / (h * h);
	let inv_dy2 = inv_dx2;
	let grid = Grid::square(n, h).unwrap();
	let stencil = Stencil::laplacian_2d_5pt(h, h);
	let bcs = Boundaries::zero_dirichlet();

	let b = hip_backend();
	let data = gaussian(n, h);
	let input = b.upload_f64(&data).unwrap();
	let mut output =
		b.allocate(n * n, DType::F64).unwrap();

	// Warmup both.
	for _ in 0..20 {
		b.apply_stencil(
			&input, &mut output, &grid, &stencil, &bcs,
		)
		.unwrap();
	}
	for _ in 0..20 {
		b.stencil_naive(
			&input,
			&mut output,
			n as i32,
			n as i32,
			inv_dx2,
			inv_dy2,
		);
	}
	sync_buf(b, &output);

	// Tiled (LDS).
	let start = Instant::now();
	for _ in 0..iters {
		b.apply_stencil(
			&input, &mut output, &grid, &stencil, &bcs,
		)
		.unwrap();
	}
	sync_buf(b, &output);
	let tiled_us =
		start.elapsed().as_secs_f64() * 1e6 / iters as f64;
	let tiled_bw = iters as f64 * (n * n) as f64 * 48.0
		/ (start.elapsed().as_secs_f64() * 1e9);

	// Naive (global).
	let start = Instant::now();
	for _ in 0..iters {
		b.stencil_naive(
			&input,
			&mut output,
			n as i32,
			n as i32,
			inv_dx2,
			inv_dy2,
		);
	}
	sync_buf(b, &output);
	let naive_us =
		start.elapsed().as_secs_f64() * 1e6 / iters as f64;
	let naive_bw = iters as f64 * (n * n) as f64 * 48.0
		/ (start.elapsed().as_secs_f64() * 1e9);

	println!(
		"  {n:5}x{n:<5}  Tiled: {tiled_us:7.1} us \
		 ({tiled_bw:6.0} GB/s) | Naive: {naive_us:7.1} us \
		 ({naive_bw:6.0} GB/s) | \
		 speedup: {:.2}x",
		naive_us / tiled_us
	);
}

// ── Test D: Size sweep to saturation ────────────────

fn bench_size_sweep(n: usize) {
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let stencil = Stencil::laplacian_2d_5pt(h, h);
	let bcs = Boundaries::zero_dirichlet();

	let b = hip_backend();
	let data = gaussian(n, h);
	let input = b.upload_f64(&data).unwrap();
	let mut output =
		b.allocate(n * n, DType::F64).unwrap();

	let iters = (50_000_000 / (n * n)).max(10);

	for _ in 0..20 {
		b.apply_stencil(
			&input, &mut output, &grid, &stencil, &bcs,
		)
		.unwrap();
	}
	sync_buf(b, &output);

	let start = Instant::now();
	for _ in 0..iters {
		b.apply_stencil(
			&input, &mut output, &grid, &stencil, &bcs,
		)
		.unwrap();
	}
	sync_buf(b, &output);
	let elapsed = start.elapsed();

	let us =
		elapsed.as_secs_f64() * 1e6 / iters as f64;
	let gflops = iters as f64 * (n * n) as f64 * 9.0
		/ (elapsed.as_secs_f64() * 1e9);
	let bw = iters as f64 * (n * n) as f64 * 48.0
		/ (elapsed.as_secs_f64() * 1e9);
	let points = (n * n) as f64;
	// Arithmetic intensity: 9 FLOP / 48 bytes = 0.1875 FLOP/byte
	let ai = 9.0 / 48.0;

	println!(
		"  Fluxion Stencil | {n:5}x{n:<5} | {iters:6} iters | \
		 {us:8.1} us | {bw:6.0} GB/s | {gflops:5.1} GF/s | \
		 AI={ai:.3} F/B | {points:.0} pts"
	);
}

// ── Main ────────────────────────────────────────────

#[test]
fn advanced_benchmark_suite() {
	println!();
	println!("=== Fluxion Advanced Benchmark ===");

	println!(
		"\n--- A: Anisotropic (dx != dy, aspect 2:1) ---"
	);
	bench_aniso(256, 2000);
	bench_aniso(512, 1000);
	bench_aniso(1024, 200);

	println!(
		"\n--- B: Convection-Diffusion (Pe~100) ---"
	);
	bench_conv_diff(256, 2000);
	bench_conv_diff(512, 1000);
	bench_conv_diff(1024, 200);

	println!("\n--- C: Naive vs LDS-Tiled stencil ---");
	bench_naive_vs_tiled(256, 5000);
	bench_naive_vs_tiled(512, 2000);
	bench_naive_vs_tiled(1024, 1000);
	bench_naive_vs_tiled(2048, 500);

	println!(
		"\n--- D: Size sweep to GPU saturation ---"
	);
	for &n in
		&[64, 128, 256, 512, 1024, 2048, 4096]
	{
		bench_size_sweep(n);
	}

	// Roofline summary.
	println!("\n--- Roofline summary ---");
	println!(
		"  RX 7800XT: ~624 GB/s peak BW, ~37 TFLOP/s f32, ~1.2 TFLOP/s f64"
	);
	println!(
		"  5-pt stencil AI = 0.1875 F/B → memory-bound"
	);
	println!(
		"  Theoretical peak: 0.1875 * 624 = 117 GFLOP/s (f64)"
	);
	println!(
		"  Our best: see D results above"
	);

	println!();
}
