#![allow(clippy::many_single_char_names)]
//! Numerical stability and edge case tests.
//!
//! Verify solver robustness under extreme conditions.

use fluxion_core::{Boundaries, DType, Grid, Stencil};
use fluxion_runtime::{Device, DiffusionSolver, Field};

fn device() -> Device {
	Device::best()
}

// ── CFL stability ────────────────────────────────────

/// The auto-computed dt must satisfy the CFL condition.
/// Running many steps without blowup confirms stability.
#[test]
fn auto_dt_is_stable_1000_steps() {
	let n = 64;
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let dev = device();

	let mut u = Field::zeros(grid, DType::F64, dev).unwrap();
	u.fill(1.0).unwrap();

	let mut solver =
		DiffusionSolver::new(grid, 1.0, None, dev).unwrap();
	solver.step_n(&mut u, 1000).unwrap();

	let norm = u.norm_l2().unwrap();
	assert!(
		norm.is_finite(),
		"solution blew up after 1000 steps: norm = {norm}"
	);
	assert!(norm > 0.0, "solution collapsed to zero");
}

/// Very small alpha should still produce stable results.
#[test]
fn tiny_alpha_stable() {
	let n = 32;
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let dev = device();

	let data = vec![1.0_f64; n * n];
	let mut u =
		Field::from_f64(grid, &data, dev).unwrap();

	let mut solver =
		DiffusionSolver::new(grid, 1e-8, None, dev)
			.unwrap();
	solver.step_n(&mut u, 100).unwrap();

	let norm = u.norm_l2().unwrap();
	assert!(norm.is_finite());
}

/// Large alpha with auto-dt should still be stable (CFL adapts).
#[test]
fn large_alpha_auto_dt_stable() {
	let n = 32;
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let dev = device();

	let data = vec![1.0_f64; n * n];
	let mut u =
		Field::from_f64(grid, &data, dev).unwrap();

	let mut solver =
		DiffusionSolver::new(grid, 100.0, None, dev)
			.unwrap();
	solver.step_n(&mut u, 50).unwrap();

	let norm = u.norm_l2().unwrap();
	assert!(
		norm.is_finite(),
		"high-alpha blew up: norm = {norm}"
	);
}

// ── Stencil on special fields ────────────────────────

/// Laplacian of a linear field u(x,y) = ax + by must be zero
/// (5-point stencil is exact for degree ≤ 2, linear is degree 1).
#[test]
fn laplacian_of_linear_is_zero() {
	let n = 32;
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let stencil = Stencil::laplacian_2d_5pt(h, h);
	let dev = device();

	let a = 3.0_f64;
	let b = -7.0_f64;
	let mut data = vec![0.0_f64; n * n];
	for row in 0..n {
		for col in 0..n {
			let x = col as f64 * h;
			let y = row as f64 * h;
			data[row * n + col] = a.mul_add(x, b * y);
		}
	}

	let input =
		Field::from_f64(grid, &data, dev).unwrap();
	let mut output =
		Field::zeros(grid, DType::F64, dev).unwrap();
	input
		.apply_stencil_into(&stencil, &Boundaries::zero_dirichlet(), &mut output)
		.unwrap();

	let out = output.to_vec_f64();
	for row in 1..n - 1 {
		for col in 1..n - 1 {
			let val = out[row * n + col];
			assert!(
				val.abs() < 1e-10,
				"Δ(linear) at ({row},{col}) = {val}"
			);
		}
	}
}

/// Stencil on a large grid should not OOM or panic.
#[test]
fn large_grid_stencil_no_panic() {
	let n = 512;
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let stencil = Stencil::laplacian_2d_5pt(h, h);
	let dev = device();

	let data = vec![1.0_f64; n * n];
	let input =
		Field::from_f64(grid, &data, dev).unwrap();
	let mut output =
		Field::zeros(grid, DType::F64, dev).unwrap();
	input
		.apply_stencil_into(&stencil, &Boundaries::zero_dirichlet(), &mut output)
		.unwrap();

	let norm = output.norm_l2().unwrap();
	assert!(norm.is_finite());
}

// ── Field operations edge cases ──────────────────────

/// `Field::swap` must exchange contents.
#[test]
fn field_swap_exchanges_data() {
	let n = 16;
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let dev = device();

	let mut a =
		Field::zeros(grid, DType::F64, dev).unwrap();
	let mut b =
		Field::zeros(grid, DType::F64, dev).unwrap();
	a.fill(1.0).unwrap();
	b.fill(2.0).unwrap();

	Field::swap(&mut a, &mut b);

	let a_data = a.to_vec_f64();
	let b_data = b.to_vec_f64();
	assert!((a_data[0] - 2.0).abs() < 1e-14);
	assert!((b_data[0] - 1.0).abs() < 1e-14);
}

/// `norm_l2` of zero field is zero.
#[test]
fn norm_l2_of_zero_is_zero() {
	let n = 32;
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let dev = device();

	let field =
		Field::zeros(grid, DType::F64, dev).unwrap();
	let norm = field.norm_l2().unwrap();
	assert!(
		norm.abs() < 1e-15,
		"norm of zeros = {norm}"
	);
}

/// axpy with alpha=0 should not change y.
#[test]
fn axpy_zero_alpha_is_identity() {
	let n = 16;
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let dev = device();

	let x =
		Field::zeros(grid, DType::F64, dev).unwrap();
	let mut y =
		Field::zeros(grid, DType::F64, dev).unwrap();
	y.fill(42.0).unwrap();

	y.axpy(0.0, &x).unwrap();

	let data = y.to_vec_f64();
	for &v in &data {
		assert!((v - 42.0).abs() < 1e-14);
	}
}

/// Multiple consecutive axpy calls accumulate correctly.
#[test]
fn axpy_accumulation() {
	let n = 16;
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let dev = device();

	let data = vec![1.0_f64; n * n];
	let x =
		Field::from_f64(grid, &data, dev).unwrap();
	let mut y =
		Field::zeros(grid, DType::F64, dev).unwrap();

	// y += 1*x three times → y = 3
	y.axpy(1.0, &x).unwrap();
	y.axpy(1.0, &x).unwrap();
	y.axpy(1.0, &x).unwrap();

	let result = y.to_vec_f64();
	for &v in &result {
		assert!((v - 3.0).abs() < 1e-13);
	}
}
