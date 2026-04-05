use fluxion_core::{Boundaries, Grid, Stencil};

use crate::cg::{CgStats, ConvergenceReason};
use crate::device::Device;
use crate::error::Result;
use crate::field::Field;
use fluxion_core::DType;

/// Pipelined Conjugate Gradient solver (Ghysels & Vanroose).
///
/// Restructures the CG data dependencies to enable:
/// - **Fused 6-way vector update** in a single kernel pass
///   (z, t, p, x, r, w updated together → fewer launches)
/// - **Fused dual dot product** via `dot2` (gamma + delta
///   in one kernel → 1 sync instead of 2)
/// - **Overlap** of operator application with convergence
///   check (operator runs while scalar copy completes)
///
/// Trade-off: convergence is checked with a 1-iteration lag,
/// and 3 extra buffers are needed (w, z, q).
///
/// Per iteration: ~5 kernel launches + 1 sync
/// vs standard CG: ~9 launches + 2 syncs.
#[derive(Debug)]
pub struct PipelinedCgSolver {
	/// Residual.
	r: Field,
	/// Search direction (recurrence-maintained).
	p: Field,
	/// w = A·r (maintained by recurrence).
	w: Field,
	/// t ≈ A·p (maintained by recurrence).
	t: Field,
	/// z ≈ A·t (maintained by recurrence).
	z: Field,
	/// q = A·w (recomputed each iteration).
	q: Field,
	/// Scratch for operator application.
	tmp: Field,
	/// Convergence tolerance.
	pub tol: f64,
	/// Maximum iterations.
	pub max_iter: usize,
}

impl PipelinedCgSolver {
	/// Create a pipelined CG solver.
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
			p: a(())?,
			w: a(())?,
			t: a(())?,
			z: a(())?,
			q: a(())?,
			tmp: a(())?,
			tol,
			max_iter,
		})
	}

	/// Apply operator: `out = inp + coeff·Δ(inp)`.
	/// Uses `self.tmp` as scratch — caller must ensure `inp`
	/// is not `self.tmp`.
	fn apply_op(
		out: &mut Field,
		inp: &Field,
		tmp: &mut Field,
		coeff: f64,
		stencil: &Stencil,
		boundaries: &Boundaries,
	) -> Result<()> {
		out.copy_from(inp)?;
		tmp.copy_from(inp)?;
		out.stencil_axpy(coeff, tmp, stencil, boundaries)
	}

	/// Solve `A·x = b` where `A·v = v + coeff·Δ(v)`.
	///
	/// Uses the pipelined CG algorithm. On HIP devices, the
	/// 6-way fused update kernel is used automatically.
	/// On CPU/wgpu, falls back to separate operations.
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
		// ── Initialization (matches aCG exactly) ────

		// r = b - A·x
		Self::apply_op(
			&mut self.q, x, &mut self.tmp,
			coeff, stencil, boundaries,
		)?;
		self.r.copy_from(b)?;
		self.r.axpy(-1.0, &self.q)?;

		// w = A·r
		Self::apply_op(
			&mut self.w, &self.r, &mut self.tmp,
			coeff, stencil, boundaries,
		)?;

		// p = t = z = 0 (aCG init: vectors start at zero)
		self.p.fill(0.0)?;
		self.t.fill(0.0)?;
		self.z.fill(0.0)?;

		// Scalar lag: gamma_prev = alpha_prev = INF
		// so that first iteration gives beta = 0.
		let mut gamma_prev = f64::INFINITY;
		let mut alpha_prev = f64::INFINITY;

		for k in 0..self.max_iter {
			// gamma = dot(r,r), delta = dot(r,w)
			let (gamma, delta) =
				self.r.dot2(&self.r, &self.r, &self.w)?;

			if gamma.sqrt() < self.tol {
				return Ok(CgStats {
					iterations: k,
					residual: gamma.sqrt(),
					reason: ConvergenceReason::Converged,
				});
			}

			// q = A·w
			Self::apply_op(
				&mut self.q, &self.w, &mut self.tmp,
				coeff, stencil, boundaries,
			)?;

			// Compute alpha and beta.
			let beta = gamma / gamma_prev;
			let alpha = gamma
				/ (delta - beta * gamma / alpha_prev);

			// Fused 6-way update.
			self.fused_update(alpha, beta, x)?;

			gamma_prev = gamma;
			alpha_prev = alpha;
		}

		let residual = self.r.dot(&self.r)?.sqrt();
		Ok(CgStats {
			iterations: self.max_iter,
			residual,
			reason: ConvergenceReason::MaxIterations,
		})
	}

	/// Fused 6-way update. Uses the HIP fused kernel when
	/// available, falls back to separate ops otherwise.
	#[inline]
	fn fused_update(
		&mut self,
		alpha: f64,
		beta: f64,
		x: &mut Field,
	) -> Result<()> {
		#[cfg(feature = "hip")]
		if matches!(x.device(), Device::Hip { .. }) {
			return self.fused_update_hip(alpha, beta, x);
		}

		// Fallback: separate operations.
		// z = q + beta*z
		self.z.scale(beta)?;
		self.z.axpy(1.0, &self.q)?;
		// t = w + beta*t
		self.t.scale(beta)?;
		self.t.axpy(1.0, &self.w)?;
		// p = r + beta*p
		self.p.scale(beta)?;
		self.p.axpy(1.0, &self.r)?;
		// x += alpha*p (updated p)
		x.axpy(alpha, &self.p)?;
		// r -= alpha*t (updated t)
		self.r.axpy(-alpha, &self.t)?;
		// w -= alpha*z (updated z)
		self.w.axpy(-alpha, &self.z)?;
		Ok(())
	}

	/// HIP fused 6-way update via single kernel.
	#[cfg(feature = "hip")]
	fn fused_update_hip(
		&mut self,
		alpha: f64,
		beta: f64,
		x: &mut Field,
	) -> Result<()> {
		use crate::storage::BufferStorage;

		let (
			BufferStorage::Hip(z),
			BufferStorage::Hip(t),
			BufferStorage::Hip(p),
			BufferStorage::Hip(x_buf),
			BufferStorage::Hip(r),
			BufferStorage::Hip(w),
			BufferStorage::Hip(q),
		) = (
			&mut self.z.storage,
			&mut self.t.storage,
			&mut self.p.storage,
			&mut x.storage,
			&mut self.r.storage,
			&mut self.w.storage,
			&self.q.storage,
		)
		else {
			unreachable!("device mismatch in hip path");
		};

		crate::storage::hip_backend().pipelined_cg_fused(
			alpha, beta, z, t, p, x_buf, r, w, q,
		);
		Ok(())
	}
}
