#![allow(clippy::many_single_char_names)]
//! Grid refinement convergence tests.
//!
//! Verify that the finite difference discretization converges
//! at the expected O(h²) rate as the grid is refined.

use fluxion_core::{Boundaries, DType, Grid, Stencil};
use fluxion_runtime::{Device, DiffusionSolver, Field};

fn device() -> Device {
	Device::best()
}

/// Compute L2 error of the Laplacian on sin(πx)·sin(πy)
/// at grid size n. Returns (h, error).
fn laplacian_l2_error(n: usize) -> (f64, f64) {
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let stencil = Stencil::laplacian_2d_5pt(h, h);
	let dev = device();
	let pi = std::f64::consts::PI;
	let factor = -2.0 * pi * pi;

	let mut data = vec![0.0_f64; n * n];
	for row in 0..n {
		for col in 0..n {
			let x = col as f64 * h;
			let y = row as f64 * h;
			data[row * n + col] = (pi * x).sin() * (pi * y).sin();
		}
	}

	let input = Field::from_f64(grid, &data, dev).unwrap();
	let mut output =
		Field::zeros(grid, DType::F64, dev).unwrap();
	input.apply_stencil_into(&stencil, &Boundaries::zero_dirichlet(), &mut output).unwrap();

	let out = output.to_vec_f64();

	// L2 error over interior (skip 1 boundary row/col).
	let mut sum_sq = 0.0_f64;
	let mut count = 0usize;
	for row in 1..n - 1 {
		for col in 1..n - 1 {
			let x = col as f64 * h;
			let y = row as f64 * h;
			let exact = factor * (pi * x).sin() * (pi * y).sin();
			let err = out[row * n + col] - exact;
			sum_sq = err.mul_add(err, sum_sq);
			count += 1;
		}
	}

	let l2 = (sum_sq / count as f64).sqrt();
	(h, l2)
}

/// The 5-point Laplacian stencil must converge at O(h²).
///
/// We measure the L2 error at 4 grid sizes and verify the
/// convergence rate is approximately 2.
#[test]
fn laplacian_convergence_order_2() {
	let sizes = [32, 64, 128, 256];
	let results: Vec<(f64, f64)> =
		sizes.iter().map(|&n| laplacian_l2_error(n)).collect();

	// Compute convergence rates between successive pairs.
	for i in 1..results.len() {
		let (h1, e1) = results[i - 1];
		let (h2, e2) = results[i];
		let rate = (e1 / e2).log(h1 / h2);
		assert!(
			rate > 1.8 && rate < 2.5,
			"convergence rate between n={} and n={} is {rate:.2}, expected ~2.0",
			sizes[i - 1],
			sizes[i]
		);
	}
}

/// Compute L2 error of diffusion solver against analytical
/// solution at grid size n. Returns (h, error).
fn diffusion_l2_error(n: usize) -> (f64, f64) {
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let dev = device();
	let alpha = 0.01;
	let pi = std::f64::consts::PI;

	let mut data = vec![0.0_f64; n * n];
	for row in 0..n {
		for col in 0..n {
			let x = col as f64 * h;
			let y = row as f64 * h;
			data[row * n + col] = (pi * x).sin() * (pi * y).sin();
		}
	}

	let mut u = Field::from_f64(grid, &data, dev).unwrap();
	let mut solver =
		DiffusionSolver::new(grid, alpha, None, dev).unwrap();

	// Run 100 steps (dt is CFL-scaled to h², so total sim time
	// varies with grid size — that's expected).
	solver.step_n(&mut u, 100).unwrap();
	let t = solver.sim_time();
	let decay = (-2.0 * pi * pi * alpha * t).exp();

	let result = u.to_vec_f64();
	let mut sum_sq = 0.0_f64;
	let mut count = 0usize;
	for row in 2..n - 2 {
		for col in 2..n - 2 {
			let x = col as f64 * h;
			let y = row as f64 * h;
			let exact =
				(pi * x).sin() * (pi * y).sin() * decay;
			let err = result[row * n + col] - exact;
			sum_sq = err.mul_add(err, sum_sq);
			count += 1;
		}
	}

	let l2 = (sum_sq / count as f64).sqrt();
	(h, l2)
}

/// The diffusion solver must converge at O(h²) spatially.
///
/// Since dt scales as h² (CFL), and Euler is O(dt) in time,
/// the combined error is O(h²). We verify rate ≥ 1.5.
#[test]
fn diffusion_spatial_convergence() {
	let sizes = [32, 64, 128];
	let results: Vec<(f64, f64)> =
		sizes.iter().map(|&n| diffusion_l2_error(n)).collect();

	for i in 1..results.len() {
		let (h1, e1) = results[i - 1];
		let (h2, e2) = results[i];
		let rate = (e1 / e2).log(h1 / h2);
		assert!(
			rate > 1.5,
			"diffusion convergence rate between n={} and n={} is {rate:.2}, expected >= 1.5",
			sizes[i - 1],
			sizes[i]
		);
	}
}
