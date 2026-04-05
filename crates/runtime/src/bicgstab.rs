use fluxion_core::{Boundaries, Grid, Stencil};

use crate::cg::{CgStats, ConvergenceReason};
use crate::device::Device;
use crate::error::Result;
use crate::field::Field;
use crate::preconditioner::Preconditioner;
use fluxion_core::DType;

/// Matrix-free BiCGSTAB solver.
///
/// Solves `A·x = b` where `A·v = v + coeff·Δ(v)`.
/// Works for **non-symmetric** systems (convection-diffusion).
/// For symmetric SPD systems, use `CgSolver` instead (faster).
///
/// All scratch buffers are pre-allocated at construction.
/// Supports optional preconditioning via `solve_preconditioned`.
#[derive(Debug)]
pub struct BiCgStabSolver {
	/// Residual.
	r: Field,
	/// Shadow residual (r̂₀, fixed after init).
	r_hat: Field,
	/// Search direction.
	p: Field,
	/// A·p result.
	v: Field,
	/// Intermediate residual (s = r - α·v).
	s: Field,
	/// A·s result.
	t: Field,
	/// Preconditioned p / scratch.
	ph: Field,
	/// Preconditioned s / scratch.
	sh: Field,
	/// Scratch for stencil input copy.
	tmp: Field,
	/// Convergence tolerance (on residual norm).
	pub tol: f64,
	/// Maximum iterations.
	pub max_iter: usize,
}

/// Apply operator: `out = inp + coeff·Δ(inp)`.
/// Inlined as a macro to avoid borrow checker issues with
/// `self.tmp` — each call site borrows disjoint fields.
macro_rules! apply_op {
	($out:expr, $inp:expr, $tmp:expr,
	 $coeff:expr, $stencil:expr, $bc:expr) => {{
		$out.copy_from($inp)?;
		$tmp.copy_from($inp)?;
		$out.stencil_axpy($coeff, &$tmp, $stencil, $bc)?;
	}};
}

impl BiCgStabSolver {
	/// Create a BiCGSTAB solver for the given grid.
	///
	/// # Errors
	/// Returns an error if buffer allocation fails.
	pub fn new(
		grid: Grid,
		device: Device,
		tol: f64,
		max_iter: usize,
	) -> Result<Self> {
		let a =
			|_| Field::zeros(grid, DType::F64, device);
		Ok(Self {
			r: a(())?,
			r_hat: a(())?,
			p: a(())?,
			v: a(())?,
			s: a(())?,
			t: a(())?,
			ph: a(())?,
			sh: a(())?,
			tmp: a(())?,
			tol,
			max_iter,
		})
	}

	/// Solve `A·x = b` (unpreconditioned BiCGSTAB).
	///
	/// `coeff` is the stencil coefficient (e.g. `-dt/2·α`).
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
		// r = b - A·x
		apply_op!(
			self.v, x, self.tmp, coeff, stencil, boundaries
		);
		self.r.copy_from(b)?;
		self.r.axpy(-1.0, &self.v)?;

		// r_hat = r (shadow residual, fixed).
		self.r_hat.copy_from(&self.r)?;

		let mut rho = 1.0_f64;
		let mut alpha = 1.0_f64;
		let mut omega = 1.0_f64;

		self.v.fill(0.0)?;
		self.p.fill(0.0)?;

		for i in 0..self.max_iter {
			let r_norm = self.r.dot(&self.r)?.sqrt();
			if r_norm < self.tol {
				return Ok(CgStats {
					iterations: i,
					residual: r_norm,
					reason: ConvergenceReason::Converged,
				});
			}

			let rho_new = self.r_hat.dot(&self.r)?;
			if rho_new.abs() < 1e-30 {
				return Ok(CgStats {
					iterations: i,
					residual: r_norm,
					reason: ConvergenceReason::Breakdown,
				});
			}

			let beta =
				(rho_new / rho) * (alpha / omega);

			// p = r + beta·(p - omega·v)
			self.p.axpy(-omega, &self.v)?;
			self.p.scale(beta)?;
			self.p.axpy(1.0, &self.r)?;

			// v = A·p
			apply_op!(
				self.v, &self.p, self.tmp,
				coeff, stencil, boundaries
			);

			// alpha = rho_new / dot(r_hat, v)
			let rv = self.r_hat.dot(&self.v)?;
			if rv.abs() < 1e-30 {
				return Ok(CgStats {
					iterations: i,
					residual: r_norm,
					reason: ConvergenceReason::Breakdown,
				});
			}
			alpha = rho_new / rv;

			// s = r - alpha·v
			self.s.copy_from(&self.r)?;
			self.s.axpy(-alpha, &self.v)?;

			// Early convergence check on s.
			let s_norm = self.s.dot(&self.s)?.sqrt();
			if s_norm < self.tol {
				x.axpy(alpha, &self.p)?;
				return Ok(CgStats {
					iterations: i,
					residual: s_norm,
					reason: ConvergenceReason::Converged,
				});
			}

			// t = A·s
			apply_op!(
				self.t, &self.s, self.tmp,
				coeff, stencil, boundaries
			);

			// omega = dot(t, s) / dot(t, t)
			let ts = self.t.dot(&self.s)?;
			let tt = self.t.dot(&self.t)?;
			if tt.abs() < 1e-30 {
				x.axpy(alpha, &self.p)?;
				return Ok(CgStats {
					iterations: i,
					residual: s_norm,
					reason: ConvergenceReason::Breakdown,
				});
			}
			omega = ts / tt;

			// x += alpha·p + omega·s
			x.axpy(alpha, &self.p)?;
			x.axpy(omega, &self.s)?;

			// r = s - omega·t
			self.r.copy_from(&self.s)?;
			self.r.axpy(-omega, &self.t)?;

			rho = rho_new;
		}

		let residual = self.r.dot(&self.r)?.sqrt();
		Ok(CgStats {
			iterations: self.max_iter,
			residual,
			reason: ConvergenceReason::MaxIterations,
		})
	}

	/// Solve `A·x = b` with preconditioning.
	///
	/// The preconditioner `pc` approximately solves `M·z = r`.
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
		// r = b - A·x
		apply_op!(
			self.v, x, self.tmp, coeff, stencil, boundaries
		);
		self.r.copy_from(b)?;
		self.r.axpy(-1.0, &self.v)?;

		// r_hat = r
		self.r_hat.copy_from(&self.r)?;

		let mut rho = 1.0_f64;
		let mut alpha = 1.0_f64;
		let mut omega = 1.0_f64;

		self.v.fill(0.0)?;
		self.p.fill(0.0)?;

		for i in 0..self.max_iter {
			let r_norm = self.r.dot(&self.r)?.sqrt();
			if r_norm < self.tol {
				return Ok(CgStats {
					iterations: i,
					residual: r_norm,
					reason: ConvergenceReason::Converged,
				});
			}

			let rho_new = self.r_hat.dot(&self.r)?;
			if rho_new.abs() < 1e-30 {
				return Ok(CgStats {
					iterations: i,
					residual: r_norm,
					reason: ConvergenceReason::Breakdown,
				});
			}

			let beta =
				(rho_new / rho) * (alpha / omega);

			// p = r + beta·(p - omega·v)
			self.p.axpy(-omega, &self.v)?;
			self.p.scale(beta)?;
			self.p.axpy(1.0, &self.r)?;

			// p_hat = M⁻¹·p
			self.ph.fill(0.0)?;
			pc.apply(&mut self.ph, &self.p)?;

			// v = A·p_hat
			apply_op!(
				self.v, &self.ph, self.tmp,
				coeff, stencil, boundaries
			);

			// alpha = rho_new / dot(r_hat, v)
			let rv = self.r_hat.dot(&self.v)?;
			if rv.abs() < 1e-30 {
				return Ok(CgStats {
					iterations: i,
					residual: r_norm,
					reason: ConvergenceReason::Breakdown,
				});
			}
			alpha = rho_new / rv;

			// s = r - alpha·v
			self.s.copy_from(&self.r)?;
			self.s.axpy(-alpha, &self.v)?;

			let s_norm = self.s.dot(&self.s)?.sqrt();
			if s_norm < self.tol {
				x.axpy(alpha, &self.ph)?;
				return Ok(CgStats {
					iterations: i,
					residual: s_norm,
					reason: ConvergenceReason::Converged,
				});
			}

			// s_hat = M⁻¹·s
			self.sh.fill(0.0)?;
			pc.apply(&mut self.sh, &self.s)?;

			// t = A·s_hat
			apply_op!(
				self.t, &self.sh, self.tmp,
				coeff, stencil, boundaries
			);

			// omega = dot(t, s) / dot(t, t)
			let ts = self.t.dot(&self.s)?;
			let tt = self.t.dot(&self.t)?;
			if tt.abs() < 1e-30 {
				x.axpy(alpha, &self.ph)?;
				return Ok(CgStats {
					iterations: i,
					residual: s_norm,
					reason: ConvergenceReason::Breakdown,
				});
			}
			omega = ts / tt;

			// x += alpha·p_hat + omega·s_hat
			x.axpy(alpha, &self.ph)?;
			x.axpy(omega, &self.sh)?;

			// r = s - omega·t
			self.r.copy_from(&self.s)?;
			self.r.axpy(-omega, &self.t)?;

			rho = rho_new;
		}

		let residual = self.r.dot(&self.r)?.sqrt();
		Ok(CgStats {
			iterations: self.max_iter,
			residual,
			reason: ConvergenceReason::MaxIterations,
		})
	}
}
