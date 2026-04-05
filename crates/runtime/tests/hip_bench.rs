//! Quick performance comparison: CPU vs HIP.
//!
//! Run with: cargo test --features hip --test hip_bench --release -- --nocapture

#![cfg(feature = "hip")]

use std::time::Instant;

use fluxion_core::{Boundaries, Grid};
use fluxion_runtime::{
	CrankNicolsonSolver, Device, DiffusionSolver, Field,
	TimeScheme,
};

fn gaussian(n: usize, h: f64) -> Vec<f64> {
	let sigma = 0.1_f64;
	let mut data = vec![0.0_f64; n * n];
	for row in 0..n {
		for col in 0..n {
			let x = col as f64 * h;
			let y = row as f64 * h;
			let dx = x - 0.5;
			let dy = y - 0.5;
			let r2 = dx.mul_add(dx, dy * dy);
			data[row * n + col] =
				(-r2 / (2.0 * sigma * sigma)).exp();
		}
	}
	data
}

fn bench_explicit(
	label: &str,
	device: Device,
	n: usize,
	steps: usize,
	scheme: TimeScheme,
) {
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let data = gaussian(n, h);

	let mut u =
		Field::from_f64(grid, &data, device).unwrap();
	let mut solver = DiffusionSolver::build(
		grid,
		0.01,
		None,
		Boundaries::zero_dirichlet(),
		scheme,
		device,
	)
	.unwrap();

	// Warmup.
	solver.step_n(&mut u, 10).unwrap();

	let start = Instant::now();
	solver.step_n(&mut u, steps).unwrap();

	// Force sync for GPU.
	let _peak = u.max().unwrap();
	let elapsed = start.elapsed();

	let us_per_step =
		elapsed.as_micros() as f64 / steps as f64;
	let total_ms = elapsed.as_millis();
	let scheme_str = format!("{scheme:?}");
	println!(
		"  {label:20} | {n:>5}x{n:<5} | {scheme_str:>6} | \
		 {steps:>6} steps | {total_ms:>6} ms | \
		 {us_per_step:>8.1} us/step"
	);
}

fn bench_implicit(
	label: &str,
	device: Device,
	n: usize,
	steps: usize,
) {
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let data = gaussian(n, h);

	let mut u =
		Field::from_f64(grid, &data, device).unwrap();
	// Large dt (no CFL constraint).
	let dt = 0.001;
	let mut solver = CrankNicolsonSolver::new(
		grid,
		0.01,
		dt,
		Boundaries::zero_dirichlet(),
		device,
		1e-6,
		500,
	)
	.unwrap();

	// Warmup.
	for _ in 0..2 {
		let _ = solver.step(&mut u, None).unwrap();
	}

	let start = Instant::now();
	for _ in 0..steps {
		let _ = solver.step(&mut u, None).unwrap();
	}
	let _peak = u.max().unwrap();
	let elapsed = start.elapsed();

	let us_per_step =
		elapsed.as_micros() as f64 / steps as f64;
	let total_ms = elapsed.as_millis();
	println!(
		"  {label:20} | {n:>5}x{n:<5} |    C-N | \
		 {steps:>6} steps | {total_ms:>6} ms | \
		 {us_per_step:>8.1} us/step"
	);
}

#[test]
fn benchmark_cpu_vs_hip() {
	println!();
	println!(
		"  {:<20} | {:>11} | {:>6} | {:>12} | \
		 {:>8} | {:>13}",
		"Backend", "Grid", "Scheme", "Steps", "Time",
		"us/step"
	);
	println!("  {:-<85}", "");

	for &n in &[64, 128, 256, 512, 1024] {
		let steps = if n <= 128 { 1000 } else { 200 };

		bench_explicit(
			"CPU Euler",
			Device::Cpu,
			n,
			steps,
			TimeScheme::Euler,
		);
		bench_explicit(
			"HIP Euler",
			Device::Hip { ordinal: 0 },
			n,
			steps,
			TimeScheme::Euler,
		);
		bench_explicit(
			"CPU RK4",
			Device::Cpu,
			n,
			steps,
			TimeScheme::Rk4,
		);
		bench_explicit(
			"HIP RK4",
			Device::Hip { ordinal: 0 },
			n,
			steps,
			TimeScheme::Rk4,
		);
		println!("  {:-<85}", "");
	}

	// Implicit solvers.
	println!();
	for &n in &[64, 128, 256, 512] {
		let steps = 20;
		bench_implicit("CPU C-N", Device::Cpu, n, steps);
		bench_implicit(
			"HIP C-N",
			Device::Hip { ordinal: 0 },
			n,
			steps,
		);
		println!("  {:-<85}", "");
	}
}
