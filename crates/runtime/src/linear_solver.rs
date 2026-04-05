use fluxion_core::{Boundaries, Stencil};

use crate::cg::CgStats;
use crate::error::Result;
use crate::field::Field;

/// Unified interface for iterative linear solvers.
///
/// All Krylov methods (CG, Pipelined CG, BiCGSTAB) implement
/// this trait.
///
/// The operator is always matrix-free: `A·v = v + coeff·Δ(v)`.
pub trait LinearSolver {
	/// Solve `A·x = b` where `A·v = v + coeff·Δ(v)`.
	///
	/// `x` is the initial guess (modified in place).
	/// Returns convergence statistics.
	///
	/// # Errors
	/// Returns an error on backend failure.
	fn solve(
		&mut self,
		x: &mut Field,
		b: &Field,
		coeff: f64,
		stencil: &Stencil,
		boundaries: &Boundaries,
	) -> Result<CgStats>;

	/// Human-readable solver name (for diagnostics).
	fn name(&self) -> &str;
}

// ── Implementations for existing solvers ────────────

impl LinearSolver for crate::cg::CgSolver {
	fn solve(
		&mut self,
		x: &mut Field,
		b: &Field,
		coeff: f64,
		stencil: &Stencil,
		boundaries: &Boundaries,
	) -> Result<CgStats> {
		self.solve(x, b, coeff, stencil, boundaries)
	}

	fn name(&self) -> &str {
		"CG"
	}
}

impl LinearSolver for crate::pipelined_cg::PipelinedCgSolver {
	fn solve(
		&mut self,
		x: &mut Field,
		b: &Field,
		coeff: f64,
		stencil: &Stencil,
		boundaries: &Boundaries,
	) -> Result<CgStats> {
		self.solve(x, b, coeff, stencil, boundaries)
	}

	fn name(&self) -> &str {
		"PipelinedCG"
	}
}

impl LinearSolver for crate::bicgstab::BiCgStabSolver {
	fn solve(
		&mut self,
		x: &mut Field,
		b: &Field,
		coeff: f64,
		stencil: &Stencil,
		boundaries: &Boundaries,
	) -> Result<CgStats> {
		self.solve(x, b, coeff, stencil, boundaries)
	}

	fn name(&self) -> &str {
		"BiCGSTAB"
	}
}

impl LinearSolver for crate::gmres::GmresSolver {
	fn solve(
		&mut self,
		x: &mut Field,
		b: &Field,
		coeff: f64,
		stencil: &Stencil,
		boundaries: &Boundaries,
	) -> Result<CgStats> {
		self.solve(x, b, coeff, stencil, boundaries)
	}

	fn name(&self) -> &str {
		"GMRES"
	}
}
