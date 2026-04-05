//! Property-based tests using proptest.
//!
//! Verify mathematical invariants hold across random inputs.

use fluxion_core::{Boundaries, DType, Grid, Stencil};
use fluxion_runtime::{Device, Field};
use proptest::prelude::*;

fn device() -> Device {
	Device::best()
}

proptest! {
	/// Stencil application on a constant field must yield zero
	/// at all interior points (Δc = 0 for any constant c).
	#[test]
	fn laplacian_of_constant_is_zero(c in -1000.0..1000.0_f64) {
		let n = 32;
		let h = 1.0 / (n - 1) as f64;
		let grid = Grid::square(n, h).unwrap();
		let stencil = Stencil::laplacian_2d_5pt(h, h);

		let data = vec![c; n * n];
		let field = Field::from_f64(grid, &data, device()).unwrap();
		let mut output = Field::zeros(grid, DType::F64, device()).unwrap();
		field.apply_stencil_into(&stencil, &Boundaries::zero_dirichlet(), &mut output).unwrap();

		let out = output.to_vec_f64();
		for row in 1..n - 1 {
			for col in 1..n - 1 {
				let val = out[row * n + col];
				// Tolerance scales with magnitude due to FP rounding.
				let tol = 1e-10 * (1.0 + c.abs());
				prop_assert!(
					val.abs() < tol,
					"Δ({c}) at ({row},{col}) = {val}, expected 0"
				);
			}
		}
	}

	/// Stencil is linear: Δ(a·u) = a·Δ(u).
	#[test]
	fn laplacian_linearity(alpha in -100.0..100.0_f64) {
		let n = 32;
		let h = 1.0 / (n - 1) as f64;
		let grid = Grid::square(n, h).unwrap();
		let stencil = Stencil::laplacian_2d_5pt(h, h);
		let dev = device();

		// u(x,y) = x² + y²
		let mut data = vec![0.0_f64; n * n];
		for row in 0..n {
			for col in 0..n {
				let x = col as f64 * h;
				let y = row as f64 * h;
				data[row * n + col] = x.mul_add(x, y * y);
			}
		}

		// Compute Δ(u) then scale by alpha.
		let u = Field::from_f64(grid, &data, dev).unwrap();
		let mut lap_u = Field::zeros(grid, DType::F64, dev).unwrap();
		u.apply_stencil_into(&stencil, &Boundaries::zero_dirichlet(), &mut lap_u).unwrap();
		let lap_u_vals = lap_u.to_vec_f64();

		// Compute Δ(alpha·u).
		let scaled: Vec<f64> = data.iter().map(|&v| alpha * v).collect();
		let au = Field::from_f64(grid, &scaled, dev).unwrap();
		let mut lap_au = Field::zeros(grid, DType::F64, dev).unwrap();
		au.apply_stencil_into(&stencil, &Boundaries::zero_dirichlet(), &mut lap_au).unwrap();
		let lap_au_vals = lap_au.to_vec_f64();

		for row in 1..n - 1 {
			for col in 1..n - 1 {
				let idx = row * n + col;
				let expected = alpha * lap_u_vals[idx];
				let got = lap_au_vals[idx];
				let tol = 1e-8 * (1.0 + expected.abs());
				prop_assert!(
					(got - expected).abs() < tol,
					"linearity at ({row},{col}): Δ({alpha}·u)={got}, {alpha}·Δu={expected}"
				);
			}
		}
	}

	/// norm_l2 must be non-negative and match manual computation.
	#[test]
	fn norm_l2_non_negative(val in -1e6..1e6_f64) {
		let n = 16;
		let h = 1.0 / (n - 1) as f64;
		let grid = Grid::square(n, h).unwrap();

		let data = vec![val; n * n];
		let field = Field::from_f64(grid, &data, device()).unwrap();
		let norm = field.norm_l2().unwrap();

		prop_assert!(norm >= 0.0, "norm must be non-negative: {norm}");

		let expected = val.abs() * ((n * n) as f64).sqrt();
		let tol = 1e-6 * (1.0 + expected);
		prop_assert!(
			(norm - expected).abs() < tol,
			"norm of constant {val}: got {norm}, expected {expected}"
		);
	}

	/// axpy must satisfy y' = alpha*x + y.
	#[test]
	fn axpy_correctness(
		alpha in -100.0..100.0_f64,
		x_val in -100.0..100.0_f64,
		y_val in -100.0..100.0_f64,
	) {
		let n = 16;
		let h = 1.0 / (n - 1) as f64;
		let grid = Grid::square(n, h).unwrap();
		let dev = device();

		let x = Field::from_f64(grid, &vec![x_val; n * n], dev).unwrap();
		let mut y = Field::from_f64(grid, &vec![y_val; n * n], dev).unwrap();
		y.axpy(alpha, &x).unwrap();

		let result = y.to_vec_f64();
		let expected = alpha.mul_add(x_val, y_val);
		let tol = 1e-10 * (1.0 + expected.abs());

		for &v in &result {
			prop_assert!(
				(v - expected).abs() < tol,
				"axpy: got {v}, expected {expected}"
			);
		}
	}
}
