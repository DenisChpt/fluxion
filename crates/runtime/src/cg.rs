use fluxion_core::{Boundaries, Grid, Stencil};

use crate::device::Device;
use crate::error::Result;
use crate::field::Field;
use crate::preconditioner::Preconditioner;
use fluxion_core::DType;

/// Why an iterative solver stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvergenceReason {
	/// Residual norm dropped below tolerance.
	Converged,
	/// Hit the maximum iteration count.
	MaxIterations,
	/// Scalar denominator became near-zero (breakdown).
	Breakdown,
}

/// Statistics from an iterative solve.
#[derive(Debug, Clone, Copy)]
pub struct SolveStats {
	/// Number of iterations performed.
	pub iterations: usize,
	/// Final residual norm.
	pub residual: f64,
	/// Why the solver stopped.
	pub reason: ConvergenceReason,
}

impl SolveStats {
	/// Whether the solver converged within tolerance.
	#[inline]
	#[must_use]
	pub const fn converged(&self) -> bool {
		matches!(self.reason, ConvergenceReason::Converged)
	}
}

/// Backward-compatible alias.
pub type CgStats = SolveStats;

/// Matrix-free Conjugate Gradient solver.
///
/// Solves `A·x = b` where `A·v = v + coeff·Δ(v)`.
/// No matrix is ever stored — the operator is applied via
/// the stencil.
///
/// All scratch buffers are pre-allocated at construction.
/// Supports optional preconditioning via `solve_preconditioned`.
#[derive(Debug)]
pub struct CgSolver {
	/// Residual.
	r: Field,
	/// Preconditioned residual (PCG).
	z: Field,
	/// Search direction.
	p: Field,
	/// A·p result.
	ap: Field,
	/// Convergence tolerance (on residual norm).
	pub tol: f64,
	/// Maximum iterations.
	pub max_iter: usize,
}

impl CgSolver {
	/// Create a CG solver for the given grid.
	///
	/// # Errors
	/// Returns an error if buffer allocation fails.
	pub fn new(
		grid: Grid,
		device: Device,
		tol: f64,
		max_iter: usize,
	) -> Result<Self> {
		Ok(Self {
			r: Field::zeros(grid, DType::F64, device)?,
			z: Field::zeros(grid, DType::F64, device)?,
			p: Field::zeros(grid, DType::F64, device)?,
			ap: Field::zeros(grid, DType::F64, device)?,
			tol,
			max_iter,
		})
	}

	/// Solve `A·x = b` where `A·v = v + coeff·Δ(v)`.
	///
	/// Unpreconditioned CG. `x` is the initial guess
	/// (modified in place). `b` is the right-hand side.
	/// `coeff` is typically `-dt/2·α` for Crank-Nicolson.
	///
	/// # Errors
	/// Returns an error on backend failure.
	pub fn solve(
		&mut self,
		x: &mut Field,
		b: &Field,
		coeff: f64,
		stencil: &Stencil,
		boundaries: &Boundaries,
	) -> Result<CgStats> {
		// r = b - A·x  where A·v = v + coeff·Δ(v)
		// ap = x, then ap += coeff·Δ(x).
		self.ap.copy_from(x)?;
		self.ap.stencil_axpy(
			coeff, x, stencil, boundaries,
		)?;
		self.r.copy_from(b)?;
		self.r.axpy(-1.0, &self.ap)?;

		// p = r
		self.p.copy_from(&self.r)?;

		let mut rs_old = self.r.dot(&self.r)?;

		for i in 0..self.max_iter {
			if rs_old.sqrt() < self.tol {
				return Ok(CgStats {
					iterations: i,
					residual: rs_old.sqrt(),
					reason: ConvergenceReason::Converged,
				});
			}

			// ap = A·p = p + coeff·Δ(p)
			self.ap.copy_from(&self.p)?;
			self.ap.stencil_axpy(
				coeff, &self.p, stencil, boundaries,
			)?;

			// alpha = rs_old / (p·ap)
			let p_ap = self.p.dot(&self.ap)?;
			if p_ap.abs() < 1e-30 {
				return Ok(CgStats {
					iterations: i,
					residual: rs_old.sqrt(),
					reason: ConvergenceReason::Breakdown,
				});
			}
			let alpha = rs_old / p_ap;

			// x += alpha·p
			x.axpy(alpha, &self.p)?;
			// r -= alpha·ap
			self.r.axpy(-alpha, &self.ap)?;

			let rs_new = self.r.dot(&self.r)?;
			let beta = rs_new / rs_old;

			// p = r + beta·p
			self.p.scale(beta)?;
			self.p.axpy(1.0, &self.r)?;

			rs_old = rs_new;
		}

		Ok(CgStats {
			iterations: self.max_iter,
			residual: rs_old.sqrt(),
			reason: ConvergenceReason::MaxIterations,
		})
	}

	/// Solve `A·x = b` with preconditioning (PCG).
	///
	/// The preconditioner `pc` approximately solves `M·z = r`.
	/// This typically converges in far fewer iterations than
	/// unpreconditioned CG (e.g. multigrid-preconditioned CG
	/// reduces iterations from ~10 to ~2-3).
	///
	/// Generic over `P: Preconditioner` — no vtable overhead,
	/// the compiler monomorphises each variant.
	///
	/// # Errors
	/// Returns an error on backend failure.
	pub fn solve_preconditioned<P: Preconditioner>(
		&mut self,
		x: &mut Field,
		b: &Field,
		coeff: f64,
		stencil: &Stencil,
		boundaries: &Boundaries,
		pc: &mut P,
	) -> Result<CgStats> {
		// r = b - A·x  where A·v = v + coeff·Δ(v)
		self.ap.copy_from(x)?;
		self.ap.stencil_axpy(
			coeff, x, stencil, boundaries,
		)?;
		self.r.copy_from(b)?;
		self.r.axpy(-1.0, &self.ap)?;

		// z = M⁻¹·r  (zero initial guess for V-cycle)
		self.z.fill(0.0)?;
		pc.apply(&mut self.z, &self.r)?;

		// p = z
		self.p.copy_from(&self.z)?;

		let mut rz_old = self.r.dot(&self.z)?;

		for i in 0..self.max_iter {
			let r_norm = self.r.dot(&self.r)?.sqrt();
			if r_norm < self.tol {
				return Ok(CgStats {
					iterations: i,
					residual: r_norm,
					reason: ConvergenceReason::Converged,
				});
			}

			// ap = A·p = p + coeff·Δ(p)
			self.ap.copy_from(&self.p)?;
			self.ap.stencil_axpy(
				coeff, &self.p, stencil, boundaries,
			)?;

			// alpha = rz_old / (p·ap)
			let p_ap = self.p.dot(&self.ap)?;
			if p_ap.abs() < 1e-30 {
				return Ok(CgStats {
					iterations: i,
					residual: r_norm,
					reason: ConvergenceReason::Breakdown,
				});
			}
			let alpha = rz_old / p_ap;

			// x += alpha·p
			x.axpy(alpha, &self.p)?;
			// r -= alpha·ap
			self.r.axpy(-alpha, &self.ap)?;

			// z = M⁻¹·r  (zero initial guess for V-cycle)
			self.z.fill(0.0)?;
			pc.apply(&mut self.z, &self.r)?;

			let rz_new = self.r.dot(&self.z)?;
			let beta = rz_new / rz_old;

			// p = z + beta·p
			self.p.scale(beta)?;
			self.p.axpy(1.0, &self.z)?;

			rz_old = rz_new;
		}

		let residual = self.r.dot(&self.r)?.sqrt();
		Ok(CgStats {
			iterations: self.max_iter,
			residual,
			reason: ConvergenceReason::MaxIterations,
		})
	}
}
