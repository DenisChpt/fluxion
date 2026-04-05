#![allow(clippy::many_single_char_names)]
//! Analytical solution validation tests.
//!
//! These tests verify correctness against known exact solutions,
//! independent of the backend. They run on whatever backend
//! `Device::best()` selects.

use fluxion_core::{Boundaries, DType, Grid, Stencil};
use fluxion_runtime::{
	CrankNicolsonSolver, Device, DiffusionSolver, Field,
	Multigrid, SmootherKind,
};

fn device() -> Device {
	Device::best()
}

// ── Laplacian correctness ────────────────────────────

/// u(x,y) = x² + y²  →  Δu = 4 everywhere.
#[test]
fn laplacian_quadratic_field() {
	let n = 64;
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let stencil = Stencil::laplacian_2d_5pt(h, h);

	let mut data = vec![0.0_f64; n * n];
	for row in 0..n {
		for col in 0..n {
			let x = col as f64 * h;
			let y = row as f64 * h;
			data[row * n + col] = x.mul_add(x, y * y);
		}
	}

	let input = Field::from_f64(grid, &data, device()).unwrap();
	let mut output =
		Field::zeros(grid, DType::F64, device()).unwrap();
	input.apply_stencil_into(&stencil, &Boundaries::zero_dirichlet(), &mut output).unwrap();

	let out = output.to_vec_f64();
	// Check interior (skip 2 rows/cols from boundary for stencil edge effects).
	for row in 2..n - 2 {
		for col in 2..n - 2 {
			let val = out[row * n + col];
			assert!(
				(val - 4.0).abs() < 1e-10,
				"Δ(x²+y²) at ({row},{col}) = {val}, expected 4.0"
			);
		}
	}
}

// ── 9-point stencil tests ────────────────────────────

/// 9-point stencil on u(x,y) = x²+y² must yield Δu = 4
/// (exact for quadratics — same as 5-point).
#[test]
fn laplacian_9pt_quadratic_field() {
	let n = 64;
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let stencil = Stencil::laplacian_2d_9pt(h, h);

	let mut data = vec![0.0_f64; n * n];
	for row in 0..n {
		for col in 0..n {
			let x = col as f64 * h;
			let y = row as f64 * h;
			data[row * n + col] = x.mul_add(x, y * y);
		}
	}

	let input =
		Field::from_f64(grid, &data, Device::Cpu).unwrap();
	let mut output =
		Field::zeros(grid, DType::F64, Device::Cpu).unwrap();
	input
		.apply_stencil_into(
			&stencil,
			&Boundaries::zero_dirichlet(),
			&mut output,
		)
		.unwrap();

	let out = output.to_vec_f64();
	for row in 2..n - 2 {
		for col in 2..n - 2 {
			let val = out[row * n + col];
			assert!(
				(val - 4.0).abs() < 1e-10,
				"9pt Δ(x²+y²) at ({row},{col}) = {val}"
			);
		}
	}
}

/// 9-point stencil converges at O(h²) with grid refinement,
/// same order as 5-point but with isotropic error structure.
#[test]
fn laplacian_9pt_converges_oh2() {
	let pi = std::f64::consts::PI;
	let dev = Device::Cpu;
	let bc = Boundaries::zero_dirichlet();

	let mut prev_err = f64::MAX;
	let mut prev_h = 0.0_f64;
	// Grid refinement: 32 → 64 → 128.
	for &n in &[32, 64, 128] {
		let h = 1.0 / (n - 1) as f64;
		let grid = Grid::square(n, h).unwrap();
		let stencil = Stencil::laplacian_2d_9pt(h, h);

		let mut data = vec![0.0_f64; n * n];
		for row in 0..n {
			for col in 0..n {
				let x = col as f64 * h;
				let y = row as f64 * h;
				data[row * n + col] =
					(pi * x).sin() * (pi * y).sin();
			}
		}
		let input =
			Field::from_f64(grid, &data, dev).unwrap();
		let mut output =
			Field::zeros(grid, DType::F64, dev).unwrap();
		input
			.apply_stencil_into(&stencil, &bc, &mut output)
			.unwrap();
		let out = output.to_vec_f64();

		let two_pi_sq = 2.0 * pi * pi;
		let mut err_max = 0.0_f64;
		for row in 3..n - 3 {
			for col in 3..n - 3 {
				let x = col as f64 * h;
				let y = row as f64 * h;
				let exact = -two_pi_sq
					* (pi * x).sin()
					* (pi * y).sin();
				err_max = err_max
					.max((out[row * n + col] - exact).abs());
			}
		}

		if prev_h > 0.0 {
			// Convergence rate: err ~= C·h^p
			// p = log(err_prev/err) / log(h_prev/h)
			let ratio = prev_err / err_max;
			let h_ratio = prev_h / h;
			let order =
				ratio.ln() / h_ratio.ln();
			assert!(
				order > 1.5,
				"9pt convergence order = {order:.2} \
				 (expected ~2), n={n}"
			);
		}
		prev_err = err_max;
		prev_h = h;
	}
}

/// 9-point stencil works with the fused stencil_axpy path.
#[test]
fn laplacian_9pt_stencil_axpy() {
	let n = 32;
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let dev = Device::Cpu;
	let stencil = Stencil::laplacian_2d_9pt(h, h);
	let bc = Boundaries::zero_dirichlet();

	let mut data = vec![0.0_f64; n * n];
	for row in 0..n {
		for col in 0..n {
			let x = col as f64 * h;
			let y = row as f64 * h;
			data[row * n + col] = x.mul_add(x, y * y);
		}
	}
	let x = Field::from_f64(grid, &data, dev).unwrap();

	// y = 1.0 everywhere, then y += 0.5 * Δ(x).
	// Since Δ(x²+y²) = 4, expect y = 1 + 0.5*4 = 3.
	let mut y =
		Field::zeros(grid, DType::F64, dev).unwrap();
	y.fill(1.0).unwrap();
	y.stencil_axpy(0.5, &x, &stencil, &bc).unwrap();

	let vals = y.to_vec_f64();
	for row in 2..n - 2 {
		for col in 2..n - 2 {
			let v = vals[row * n + col];
			assert!(
				(v - 3.0).abs() < 1e-10,
				"9pt axpy at ({row},{col}) = {v}, expected 3"
			);
		}
	}
}

/// u(x,y) = sin(πx)·sin(πy)  →  Δu = -2π²·sin(πx)·sin(πy).
#[test]
fn laplacian_sinusoidal_field() {
	let n = 128;
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let stencil = Stencil::laplacian_2d_5pt(h, h);

	let pi = std::f64::consts::PI;
	let expected_factor = -2.0 * pi * pi;

	let mut data = vec![0.0_f64; n * n];
	for row in 0..n {
		for col in 0..n {
			let x = col as f64 * h;
			let y = row as f64 * h;
			data[row * n + col] = (pi * x).sin() * (pi * y).sin();
		}
	}

	let input = Field::from_f64(grid, &data, device()).unwrap();
	let mut output =
		Field::zeros(grid, DType::F64, device()).unwrap();
	input.apply_stencil_into(&stencil, &Boundaries::zero_dirichlet(), &mut output).unwrap();

	let out = output.to_vec_f64();
	// FD truncation error is O(h²) relative to the value magnitude.
	let rel_tol = 10.0 * h * h;
	for row in 2..n - 2 {
		for col in 2..n - 2 {
			let x = col as f64 * h;
			let y = row as f64 * h;
			let expected =
				expected_factor * (pi * x).sin() * (pi * y).sin();
			let got = out[row * n + col];
			let tol = rel_tol * (1.0 + expected.abs());
			assert!(
				(got - expected).abs() < tol,
				"Δsin at ({row},{col}): got {got:.6}, expected {expected:.6}, tol={tol:.6}"
			);
		}
	}
}

/// u(x,y) = x³ + y³  →  Δu = 6x + 6y (MMS cubic).
#[test]
fn laplacian_cubic_mms() {
	let n = 64;
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let stencil = Stencil::laplacian_2d_5pt(h, h);

	let mut data = vec![0.0_f64; n * n];
	for row in 0..n {
		for col in 0..n {
			let x = col as f64 * h;
			let y = row as f64 * h;
			data[row * n + col] = x.powi(3) + y.powi(3);
		}
	}

	let input = Field::from_f64(grid, &data, device()).unwrap();
	let mut output =
		Field::zeros(grid, DType::F64, device()).unwrap();
	input.apply_stencil_into(&stencil, &Boundaries::zero_dirichlet(), &mut output).unwrap();

	let out = output.to_vec_f64();
	// O(h²) tolerance for the cubic term (FD is exact for quadratic,
	// so the error comes from the cubic part).
	let tol = 10.0 * h * h;
	for row in 2..n - 2 {
		for col in 2..n - 2 {
			let x = col as f64 * h;
			let y = row as f64 * h;
			let expected = 6.0f64.mul_add(x, 6.0 * y);
			let got = out[row * n + col];
			assert!(
				(got - expected).abs() < tol,
				"Δ(x³+y³) at ({row},{col}): got {got:.6}, expected {expected:.6}"
			);
		}
	}
}

// ── Diffusion solver correctness ─────────────────────

/// Analytical heat equation solution:
/// u(t,x,y) = sin(πx)·sin(πy)·exp(-2π²αt)
///
/// Verify the numerical solution matches at `t_final`.
#[test]
fn diffusion_sinusoidal_analytical() {
	let n = 64;
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let dev = device();
	let alpha = 0.01;
	let pi = std::f64::consts::PI;

	// Initial condition: sin(πx)·sin(πy).
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

	let n_steps = 200;
	solver.step_n(&mut u, n_steps).unwrap();
	let t = solver.sim_time();

	let result = u.to_vec_f64();
	let decay = (-2.0 * pi * pi * alpha * t).exp();

	// Check interior — allow O(dt) + O(h²) error.
	let dt = solver.dt();
	let tol = 5.0 * (dt + h * h);
	for row in 3..n - 3 {
		for col in 3..n - 3 {
			let x = col as f64 * h;
			let y = row as f64 * h;
			let expected =
				(pi * x).sin() * (pi * y).sin() * decay;
			let got = result[row * n + col];
			assert!(
				(got - expected).abs() < tol,
				"at ({row},{col}), t={t:.6}: got {got:.8}, expected {expected:.8}"
			);
		}
	}
}

/// Gaussian diffusion: peak must strictly decrease.
#[test]
fn diffusion_peak_decreases() {
	let n = 64;
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let dev = device();
	let alpha = 0.01;
	let sigma = 0.1_f64;
	let inv_2s2 = 1.0 / (2.0 * sigma * sigma);

	let mut data = vec![0.0_f64; n * n];
	for row in 0..n {
		for col in 0..n {
			let x = col as f64 * h;
			let y = row as f64 * h;
			let dx = x - 0.5;
			let dy = y - 0.5;
			let r2 = dx.mul_add(dx, dy * dy);
			data[row * n + col] = (-r2 * inv_2s2).exp();
		}
	}

	let mut u = Field::from_f64(grid, &data, dev).unwrap();
	let mut solver =
		DiffusionSolver::new(grid, alpha, None, dev).unwrap();

	let mut prev_peak = f64::MAX;
	for _ in 0..10 {
		solver.step_n(&mut u, 50).unwrap();
		let vals = u.to_vec_f64();
		let peak: f64 = vals
			.iter()
			.copied()
			.reduce(f64::max)
			.unwrap();
		assert!(
			peak < prev_peak,
			"peak must decrease: {prev_peak} -> {peak}"
		);
		assert!(peak > 0.0, "field must remain positive");
		prev_peak = peak;
	}
}

/// Symmetric initial condition must remain symmetric.
#[test]
fn diffusion_preserves_symmetry() {
	let n = 65; // Odd so center point exists.
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let dev = device();
	let alpha = 0.01;

	// Symmetric Gaussian centered at (0.5, 0.5).
	let sigma = 0.1_f64;
	let inv_2s2 = 1.0 / (2.0 * sigma * sigma);
	let mut data = vec![0.0_f64; n * n];
	for row in 0..n {
		for col in 0..n {
			let x = col as f64 * h;
			let y = row as f64 * h;
			let dx = x - 0.5;
			let dy = y - 0.5;
			let r2 = dx.mul_add(dx, dy * dy);
			data[row * n + col] = (-r2 * inv_2s2).exp();
		}
	}

	let mut u = Field::from_f64(grid, &data, dev).unwrap();
	let mut solver =
		DiffusionSolver::new(grid, alpha, None, dev).unwrap();
	solver.step_n(&mut u, 200).unwrap();

	let vals = u.to_vec_f64();
	let mid = n / 2;
	let tol = 1e-10;

	// Check left-right symmetry: u(row, mid-k) ≈ u(row, mid+k).
	for row in 1..n - 1 {
		for k in 1..mid {
			let left = vals[row * n + (mid - k)];
			let right = vals[row * n + (mid + k)];
			assert!(
				(left - right).abs() < tol,
				"LR symmetry broken at row={row}, k={k}: {left} vs {right}"
			);
		}
	}

	// Check top-bottom symmetry: u(mid-k, col) ≈ u(mid+k, col).
	for col in 1..n - 1 {
		for k in 1..mid {
			let top = vals[(mid - k) * n + col];
			let bot = vals[(mid + k) * n + col];
			assert!(
				(top - bot).abs() < tol,
				"TB symmetry broken at col={col}, k={k}: {top} vs {bot}"
			);
		}
	}
}

// ── Multigrid V-cycle smoother tests ────────────────

/// Build a Poisson RHS (sin·sin) and check that a V-cycle
/// with the given smoother reduces the residual.
fn multigrid_reduces_residual(smoother: SmootherKind) {
	let n = 64;
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let dev = Device::Cpu;
	let bc = Boundaries::zero_dirichlet();

	// RHS = sin(πx)·sin(πy) — smooth Poisson source.
	let pi = std::f64::consts::PI;
	let mut rhs_data = vec![0.0_f64; n * n];
	for row in 0..n {
		for col in 0..n {
			let x = col as f64 * h;
			let y = row as f64 * h;
			rhs_data[row * n + col] =
				(pi * x).sin() * (pi * y).sin();
		}
	}
	let b = Field::from_f64(grid, &rhs_data, dev).unwrap();

	let mut mg = Multigrid::build(
		grid,
		bc,
		dev,
		3,
		3,
		2.0 / 3.0,
		smoother,
	)
	.unwrap();

	assert!(mg.depth() >= 3, "need at least 3 levels");

	let mut x =
		Field::zeros(grid, DType::F64, dev).unwrap();

	// Apply several V-cycles and check residual drops.
	let stencil = Stencil::laplacian_2d_5pt(h, h);
	let mut prev_resid = f64::MAX;
	for _ in 0..10 {
		mg.v_cycle(&mut x, &b).unwrap();
		// Compute residual: r = b - Δ(x).
		let mut ax =
			Field::zeros(grid, DType::F64, dev).unwrap();
		x.apply_stencil_into(&stencil, &bc, &mut ax)
			.unwrap();
		let mut r =
			Field::zeros(grid, DType::F64, dev).unwrap();
		r.copy_from(&b).unwrap();
		r.axpy(-1.0, &ax).unwrap();
		let resid = r.dot(&r).unwrap().sqrt();
		assert!(
			resid < prev_resid,
			"residual must decrease: {prev_resid} -> {resid}"
		);
		prev_resid = resid;
	}
	// After 10 V-cycles, residual should have dropped significantly.
	assert!(
		prev_resid < 10.0,
		"residual after V-cycles too high: {prev_resid}"
	);
}

#[test]
fn multigrid_jacobi_reduces_residual() {
	multigrid_reduces_residual(SmootherKind::Jacobi);
}

#[test]
fn multigrid_chebyshev_reduces_residual() {
	multigrid_reduces_residual(SmootherKind::Chebyshev);
}

/// PCG with Chebyshev smoother converges on CN step.
#[test]
fn crank_nicolson_pcg_chebyshev() {
	let n = 64;
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let dev = Device::Cpu;
	let alpha = 0.01;
	let dt = 0.01;
	let bc = Boundaries::zero_dirichlet();

	let mut solver =
		CrankNicolsonSolver::with_multigrid_smoother(
			grid,
			alpha,
			dt,
			bc,
			dev,
			1e-10,
			200,
			3,
			3,
			2.0 / 3.0,
			SmootherKind::Chebyshev,
		)
		.unwrap();

	// Gaussian IC.
	let sigma = 0.1_f64;
	let inv_2s2 = 1.0 / (2.0 * sigma * sigma);
	let mut data = vec![0.0_f64; n * n];
	for row in 0..n {
		for col in 0..n {
			let x = col as f64 * h;
			let y = row as f64 * h;
			let dx = x - 0.5;
			let dy = y - 0.5;
			let r2 = dx.mul_add(dx, dy * dy);
			data[row * n + col] = (-r2 * inv_2s2).exp();
		}
	}
	let mut u =
		Field::from_f64(grid, &data, dev).unwrap();

	let stats = solver.step(&mut u, None).unwrap();
	assert!(
		stats.converged(),
		"PCG+Chebyshev should converge, got {:?} \
		 (resid={:.2e}, iters={})",
		stats.reason,
		stats.residual,
		stats.iterations,
	);
}
