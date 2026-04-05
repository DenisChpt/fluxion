//! Error handling and validation tests.
//!
//! Verify that invalid inputs are rejected cleanly.

use fluxion_core::{Boundaries, DType, Grid, Stencil};
use fluxion_runtime::{Device, Field};

fn device() -> Device {
	Device::best()
}

#[test]
fn from_f64_rejects_wrong_length() {
	let grid = Grid::square(16, 0.1).unwrap();
	let data = vec![0.0_f64; 100]; // wrong size
	let result = Field::from_f64(grid, &data, device());
	assert!(result.is_err());
}

#[test]
fn apply_stencil_rejects_mismatched_grids() {
	let dev = device();
	let grid_a = Grid::square(16, 0.1).unwrap();
	let grid_b = Grid::square(32, 0.05).unwrap();

	let a =
		Field::zeros(grid_a, DType::F64, dev).unwrap();
	let mut b =
		Field::zeros(grid_b, DType::F64, dev).unwrap();
	let stencil = Stencil::laplacian_2d_5pt(0.1, 0.1);

	let result = a.apply_stencil_into(&stencil, &Boundaries::zero_dirichlet(), &mut b);
	assert!(
		result.is_err(),
		"should reject mismatched grid sizes"
	);
}

#[test]
fn axpy_rejects_mismatched_sizes() {
	let dev = device();
	let grid_a = Grid::square(16, 0.1).unwrap();
	let grid_b = Grid::square(32, 0.05).unwrap();

	let x =
		Field::zeros(grid_a, DType::F64, dev).unwrap();
	let mut y =
		Field::zeros(grid_b, DType::F64, dev).unwrap();

	let result = y.axpy(1.0, &x);
	assert!(
		result.is_err(),
		"should reject mismatched buffer sizes"
	);
}

#[test]
fn grid_rejects_1x1() {
	assert!(Grid::square(1, 0.1).is_err());
}

#[test]
fn grid_rejects_2x2() {
	assert!(Grid::square(2, 0.1).is_err());
}

#[test]
fn grid_accepts_3x3() {
	assert!(Grid::square(3, 0.1).is_ok());
}

#[test]
fn stencil_rejects_empty() {
	assert!(Stencil::new(vec![]).is_err());
}
