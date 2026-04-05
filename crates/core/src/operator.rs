use crate::boundary::Boundaries;
use crate::dtype::DType;
use crate::grid::Grid;
use crate::stencil::Stencil;

/// What operation to perform.
#[derive(Debug, Clone)]
pub enum OperatorKind {
	/// Discrete Laplacian (Δu).
	Laplacian,
	/// Diffusion: α·Δu.
	Diffusion { alpha: f64 },
	/// User-defined stencil operation.
	Custom(Stencil),
}

/// Full specification of a PDE operator application.
///
/// Describes WHAT to compute — the backend decides HOW.
#[derive(Debug, Clone)]
pub struct OperatorSpec {
	pub kind: OperatorKind,
	pub grid: Grid,
	pub boundaries: Boundaries,
	pub dtype: DType,
}

impl OperatorSpec {
	#[must_use]
	pub const fn new(
		kind: OperatorKind,
		grid: Grid,
		boundaries: Boundaries,
		dtype: DType,
	) -> Self {
		Self { kind, grid, boundaries, dtype }
	}

	/// Build the stencil corresponding to this operator.
	///
	/// # Panics
	/// Cannot panic — internal stencil construction always succeeds.
	#[must_use]
	pub fn stencil(&self) -> Stencil {
		match &self.kind {
			OperatorKind::Laplacian => {
				Stencil::laplacian_2d_5pt(
					self.grid.dx,
					self.grid.dy,
				)
			}
			OperatorKind::Diffusion { alpha } => {
				let base = Stencil::laplacian_2d_5pt(
					self.grid.dx,
					self.grid.dy,
				);
				let scaled = base
					.entries()
					.iter()
					.map(|e| crate::stencil::StencilEntry {
						dr: e.dr,
						dc: e.dc,
						weight: e.weight * alpha,
					})
					.collect();
				// Safe: laplacian is non-empty so scaled is too.
				Stencil::new(scaled).expect("non-empty stencil")
			}
			OperatorKind::Custom(s) => s.clone(),
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::stencil::StencilEntry;

	fn test_grid() -> Grid {
		Grid::square(16, 0.1).unwrap()
	}

	#[test]
	fn laplacian_stencil_is_5pt() {
		let spec = OperatorSpec::new(
			OperatorKind::Laplacian,
			test_grid(),
			Boundaries::zero_dirichlet(),
			DType::F64,
		);
		let s = spec.stencil();
		assert_eq!(s.len(), 5);
	}

	#[test]
	fn diffusion_scales_weights() {
		let grid = test_grid();
		let alpha = 0.5;
		let spec = OperatorSpec::new(
			OperatorKind::Diffusion { alpha },
			grid,
			Boundaries::zero_dirichlet(),
			DType::F64,
		);
		let base =
			Stencil::laplacian_2d_5pt(grid.dx, grid.dy);
		let scaled = spec.stencil();

		for (b, s) in base
			.entries()
			.iter()
			.zip(scaled.entries().iter())
		{
			assert!(
				b.weight.mul_add(-alpha, s.weight).abs()
					< 1e-14
			);
		}
	}

	#[test]
	fn custom_stencil_passthrough() {
		let entries = vec![
			StencilEntry { dr: 0, dc: 0, weight: -2.0 },
			StencilEntry { dr: 0, dc: 1, weight: 1.0 },
			StencilEntry { dr: 0, dc: -1, weight: 1.0 },
		];
		let custom =
			Stencil::new(entries).unwrap();
		let spec = OperatorSpec::new(
			OperatorKind::Custom(custom.clone()),
			test_grid(),
			Boundaries::zero_dirichlet(),
			DType::F64,
		);
		assert_eq!(spec.stencil(), custom);
	}

	#[test]
	fn diffusion_stencil_weights_sum_to_zero() {
		let spec = OperatorSpec::new(
			OperatorKind::Diffusion { alpha: 3.7 },
			test_grid(),
			Boundaries::zero_dirichlet(),
			DType::F64,
		);
		let sum: f64 = spec
			.stencil()
			.entries()
			.iter()
			.map(|e| e.weight)
			.sum();
		assert!(sum.abs() < 1e-12);
	}
}
