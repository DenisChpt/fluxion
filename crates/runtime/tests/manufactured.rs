#![allow(clippy::many_single_char_names)]
//! Method of Manufactured Solutions (MMS) tests.
//!
//! Standard problems from deal.II and SUNDIALS.
//! Each test uses a known exact solution, computes the
//! discrete Laplacian, and verifies against the analytical
//! result within O(h²) tolerance.

use fluxion_core::{Boundaries, DType, Grid, Stencil};
use fluxion_runtime::{Device, DiffusionSolver, Field};

fn device() -> Device {
	Device::best()
}

// ── Poisson: u = x(1-x)y(1-y) ─────────────────────────

/// Standard Poisson test (KSP ex2).
/// u(x,y) = x(1-x)y(1-y)
/// Δu = -2[y(1-y) + x(1-x)]
#[test]
fn petsc_poisson_x1mx_y1my() {
	let n = 128;
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let stencil = Stencil::laplacian_2d_5pt(h, h);
	let dev = device();

	let mut data = vec![0.0_f64; n * n];
	for row in 0..n {
		for col in 0..n {
			let x = col as f64 * h;
			let y = row as f64 * h;
			data[row * n + col] =
				x * (1.0 - x) * y * (1.0 - y);
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
	let rel_tol = 5.0 * h * h;
	for row in 2..n - 2 {
		for col in 2..n - 2 {
			let x = col as f64 * h;
			let y = row as f64 * h;
			let expected =
				-2.0 * (y * (1.0 - y) + x * (1.0 - x));
			let got = out[row * n + col];
			let tol = rel_tol * (1.0 + expected.abs());
			assert!(
				(got - expected).abs() < tol,
				"Poisson at ({row},{col}): got {got:.8}, expected {expected:.8}"
			);
		}
	}
}

// ── deal.II Step-3: u = 1 + x² + 2y² ───────────────

/// deal.II reference Poisson problem.
/// u(x,y) = 1 + x² + 2y²
/// Δu = 2 + 4 = 6
#[test]
fn dealii_poisson_1_x2_2y2() {
	let n = 64;
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let stencil = Stencil::laplacian_2d_5pt(h, h);
	let dev = device();

	let mut data = vec![0.0_f64; n * n];
	for row in 0..n {
		for col in 0..n {
			let x = col as f64 * h;
			let y = row as f64 * h;
			data[row * n + col] =
				(2.0 * y).mul_add(y, x.mul_add(x, 1.0));
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
	// 5-point stencil is exact for quadratics.
	for row in 1..n - 1 {
		for col in 1..n - 1 {
			let got = out[row * n + col];
			assert!(
				(got - 6.0).abs() < 1e-10,
				"deal.II at ({row},{col}): got {got}, expected 6.0"
			);
		}
	}
}

// ── SUNDIALS cv_heat2D: sin²(πx)sin²(πy) ────────────

/// SUNDIALS heat equation benchmark.
/// u(0,x,y) = sin²(πx)·sin²(πy)
/// Analytical solution under pure diffusion (α=1, no source):
/// decays toward zero. We verify L2 norm decreases monotonically.
#[test]
fn sundials_heat2d_norm_decreasing() {
	let n = 64;
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let dev = device();
	let pi = std::f64::consts::PI;

	let mut data = vec![0.0_f64; n * n];
	for row in 0..n {
		for col in 0..n {
			let x = col as f64 * h;
			let y = row as f64 * h;
			data[row * n + col] =
				(pi * x).sin().powi(2)
					* (pi * y).sin().powi(2);
		}
	}

	let mut u =
		Field::from_f64(grid, &data, dev).unwrap();
	let mut solver =
		DiffusionSolver::new(grid, 0.01, None, dev)
			.unwrap();

	let mut prev_norm = u.norm_l2().unwrap();
	for _ in 0..20 {
		solver.step_n(&mut u, 50).unwrap();
		let norm = u.norm_l2().unwrap();
		assert!(
			norm <= prev_norm + 1e-12,
			"L2 norm must be non-increasing: {prev_norm} -> {norm}"
		);
		prev_norm = norm;
	}
}

// ── High-order polynomial: u = x⁴ + y⁴ ─────────────

/// u(x,y) = x⁴ + y⁴
/// Δu = 12x² + 12y²
/// Error is O(h²) since the stencil is 2nd order.
#[test]
fn laplacian_quartic_mms() {
	let n = 128;
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let stencil = Stencil::laplacian_2d_5pt(h, h);
	let dev = device();

	let mut data = vec![0.0_f64; n * n];
	for row in 0..n {
		for col in 0..n {
			let x = col as f64 * h;
			let y = row as f64 * h;
			data[row * n + col] =
				x.powi(4) + y.powi(4);
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
	let tol = 50.0 * h * h; // Quartic has larger h² coefficient.
	for row in 2..n - 2 {
		for col in 2..n - 2 {
			let x = col as f64 * h;
			let y = row as f64 * h;
			let expected =
				(12.0 * x).mul_add(x, 12.0 * y * y);
			let got = out[row * n + col];
			let abs_tol = tol * (1.0 + expected.abs());
			assert!(
				(got - expected).abs() < abs_tol,
				"Δ(x⁴+y⁴) at ({row},{col}): got {got:.6}, expected {expected:.6}"
			);
		}
	}
}

// ── Exponential: u = exp(x+y) ────────────────────────

/// u(x,y) = exp(x+y)
/// Δu = 2·exp(x+y)
#[test]
fn laplacian_exponential_mms() {
	let n = 128;
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let stencil = Stencil::laplacian_2d_5pt(h, h);
	let dev = device();

	let mut data = vec![0.0_f64; n * n];
	for row in 0..n {
		for col in 0..n {
			let x = col as f64 * h;
			let y = row as f64 * h;
			data[row * n + col] = (x + y).exp();
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
	let rel_tol = 10.0 * h * h;
	for row in 2..n - 2 {
		for col in 2..n - 2 {
			let x = col as f64 * h;
			let y = row as f64 * h;
			let expected = 2.0 * (x + y).exp();
			let got = out[row * n + col];
			let tol = rel_tol * (1.0 + expected.abs());
			assert!(
				(got - expected).abs() < tol,
				"Δexp at ({row},{col}): got {got:.6}, expected {expected:.6}"
			);
		}
	}
}

// ── Product of sinusoids: u = sin(2πx)sin(3πy) ──────

/// u(x,y) = sin(2πx)·sin(3πy)
/// Δu = -(4π² + 9π²)·sin(2πx)·sin(3πy) = -13π²·u
#[test]
fn laplacian_higher_frequency_sinusoidal() {
	let n = 256; // Needs finer grid for higher frequencies.
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let stencil = Stencil::laplacian_2d_5pt(h, h);
	let dev = device();
	let pi = std::f64::consts::PI;

	let mut data = vec![0.0_f64; n * n];
	for row in 0..n {
		for col in 0..n {
			let x = col as f64 * h;
			let y = row as f64 * h;
			data[row * n + col] =
				(2.0 * pi * x).sin()
					* (3.0 * pi * y).sin();
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
	let factor = -13.0 * pi * pi;
	let rel_tol = 20.0 * h * h; // Higher frequency → larger h² coefficient.
	for row in 3..n - 3 {
		for col in 3..n - 3 {
			let x = col as f64 * h;
			let y = row as f64 * h;
			let expected = factor
				* (2.0 * pi * x).sin()
				* (3.0 * pi * y).sin();
			let got = out[row * n + col];
			let tol = rel_tol * (1.0 + expected.abs());
			assert!(
				(got - expected).abs() < tol,
				"Δsin(2πx)sin(3πy) at ({row},{col}): got {got:.6}, expected {expected:.6}"
			);
		}
	}
}
