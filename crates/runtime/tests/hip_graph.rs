//! hipGraph integration tests.
//!
//! These must run in isolation (single thread) because graph
//! capture affects the shared HIP stream singleton.
//!
//! Run: cargo test --features hip --test hip_graph -- --test-threads=1

#![cfg(feature = "hip")]

use fluxion_core::{DType, Grid};
use fluxion_runtime::{Device, DiffusionSolver, Field};

fn hip() -> Device {
	Device::Hip { ordinal: 0 }
}

fn gaussian(n: usize, h: f64) -> Vec<f64> {
	let sigma = 0.1_f64;
	(0..n * n)
		.map(|i| {
			let (row, col) = (i / n, i % n);
			let dx = col as f64 * h - 0.5;
			let dy = row as f64 * h - 0.5;
			(-dx.mul_add(dx, dy * dy) / (2.0 * sigma * sigma))
				.exp()
		})
		.collect()
}

#[test]
fn graph_euler_matches_regular() {
	let n = 64;
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let data = gaussian(n, h);

	// Reference: regular steps.
	let mut u_ref =
		Field::from_f64(grid, &data, hip()).unwrap();
	let mut solver_ref =
		DiffusionSolver::new(grid, 0.01, None, hip())
			.unwrap();
	solver_ref.step_n(&mut u_ref, 200).unwrap();

	// Graph-accelerated.
	let mut u_gr =
		Field::from_f64(grid, &data, hip()).unwrap();
	let mut solver_gr =
		DiffusionSolver::new(grid, 0.01, None, hip())
			.unwrap();
	solver_gr.enable_hip_graph();
	solver_gr.step_n(&mut u_gr, 200).unwrap();

	let d_ref = u_ref.to_vec_f64();
	let d_gr = u_gr.to_vec_f64();
	let max_diff: f64 = d_ref
		.iter()
		.zip(d_gr.iter())
		.map(|(a, b)| (a - b).abs())
		.reduce(f64::max)
		.unwrap();
	assert!(
		max_diff < 1e-10,
		"graph vs ref max diff = {max_diff}"
	);
}

#[test]
fn graph_diffusion_decreases_peak() {
	let n = 64;
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let data = gaussian(n, h);
	let initial_peak: f64 = data
		.iter()
		.copied()
		.reduce(f64::max)
		.unwrap();

	let mut u =
		Field::from_f64(grid, &data, hip()).unwrap();
	let mut solver =
		DiffusionSolver::new(grid, 0.01, None, hip())
			.unwrap();
	solver.enable_hip_graph();
	solver.step_n(&mut u, 500).unwrap();

	let final_peak = u.max().unwrap();
	assert!(
		final_peak < initial_peak,
		"peak should decrease: {final_peak} >= {initial_peak}"
	);
}
