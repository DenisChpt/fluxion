#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BoundaryCondition {
	/// Fixed value at the boundary: `u = val`.
	Dirichlet(f64),
	/// Fixed normal derivative at the boundary: `∂u/∂n = flux`.
	Neumann(f64),
	/// Robin (mixed): `alpha·u + beta·∂u/∂n = g`.
	///
	/// Dirichlet is Robin with beta=0: `alpha·u = g`.
	/// Neumann is Robin with alpha=0: `beta·∂u/∂n = g`.
	Robin { alpha: f64, beta: f64, g: f64 },
	/// Wrap-around (periodic domain).
	Periodic,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Boundaries {
	pub top: BoundaryCondition,
	pub bottom: BoundaryCondition,
	pub left: BoundaryCondition,
	pub right: BoundaryCondition,
}

impl Boundaries {
	#[must_use]
	pub const fn uniform(bc: BoundaryCondition) -> Self {
		Self {
			top: bc,
			bottom: bc,
			left: bc,
			right: bc,
		}
	}

	#[must_use]
	pub const fn zero_dirichlet() -> Self {
		Self::uniform(BoundaryCondition::Dirichlet(0.0))
	}

	#[must_use]
	pub const fn periodic() -> Self {
		Self::uniform(BoundaryCondition::Periodic)
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn zero_dirichlet_all_sides() {
		let bc = Boundaries::zero_dirichlet();
		assert_eq!(
			bc.top,
			BoundaryCondition::Dirichlet(0.0)
		);
		assert_eq!(
			bc.bottom,
			BoundaryCondition::Dirichlet(0.0)
		);
		assert_eq!(
			bc.left,
			BoundaryCondition::Dirichlet(0.0)
		);
		assert_eq!(
			bc.right,
			BoundaryCondition::Dirichlet(0.0)
		);
	}

	#[test]
	fn periodic_all_sides() {
		let bc = Boundaries::periodic();
		assert_eq!(bc.top, BoundaryCondition::Periodic);
		assert_eq!(bc.bottom, BoundaryCondition::Periodic);
	}

	#[test]
	fn mixed_boundaries() {
		let bc = Boundaries {
			top: BoundaryCondition::Dirichlet(1.0),
			bottom: BoundaryCondition::Dirichlet(0.0),
			left: BoundaryCondition::Neumann(0.0),
			right: BoundaryCondition::Periodic,
		};
		assert_eq!(
			bc.top,
			BoundaryCondition::Dirichlet(1.0)
		);
		assert_eq!(
			bc.right,
			BoundaryCondition::Periodic
		);
	}

	#[test]
	fn copy_semantics() {
		let a =
			BoundaryCondition::Dirichlet(42.0);
		let b = a;
		assert_eq!(a, b);
	}

	#[test]
	fn robin_boundary() {
		let bc = Boundaries {
			top: BoundaryCondition::Robin {
				alpha: 1.0,
				beta: 1.0,
				g: 0.0,
			},
			bottom: BoundaryCondition::Dirichlet(0.0),
			left: BoundaryCondition::Dirichlet(0.0),
			right: BoundaryCondition::Dirichlet(0.0),
		};
		assert!(matches!(
			bc.top,
			BoundaryCondition::Robin { .. }
		));
	}
}
