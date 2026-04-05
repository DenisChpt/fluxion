use fluxion_core::{Boundaries, Grid, Stencil};

use crate::cg::{CgStats, ConvergenceReason};
use crate::device::Device;
use crate::error::Result;
use crate::field::Field;
use crate::preconditioner::Preconditioner;
use fluxion_core::DType;

/// Restarted GMRES(m) solver.
///
/// Solves `A·x = b` where `A·v = v + coeff·Δ(v)`.
/// Works for **any** system (non-symmetric, indefinite).
/// CG and BiCGSTAB are faster when applicable; GMRES is the
/// most robust fallback.
///
/// All buffers are pre-allocated at construction:
/// - `m+1` Krylov basis vectors (dominant memory cost)
/// - Hessenberg matrix `(m+1) × m` (tiny)
/// - Givens rotation coefficients
///
/// No allocations in the hot path.
#[derive(Debug)]
pub struct GmresSolver {
	/// Krylov basis vectors v₀..v_m.
	vv: Vec<Field>,
	/// Work vector for operator application.
	w: Field,
	/// Hessenberg matrix H (column-major, (m+1) rows × m cols).
	hh: Vec<f64>,
	/// Givens cosines.
	cc: Vec<f64>,
	/// Givens sines.
	ss: Vec<f64>,
	/// Right-hand side of the least-squares problem.
	grs: Vec<f64>,
	/// Restart parameter.
	restart: usize,
	/// Convergence tolerance.
	pub tol: f64,
	/// Maximum total iterations (across restarts).
	pub max_iter: usize,
}

impl GmresSolver {
	/// Create a GMRES(m) solver.
	///
	/// `restart` is `m` — the Krylov subspace dimension before
	/// restart. Typical values: 30 for well-conditioned, 100+
	/// for ill-conditioned systems.
	///
	/// Memory: `(m+1)` fields + O(m²) scalars.
	///
	/// # Errors
	/// Returns an error if buffer allocation fails.
	pub fn new(
		grid: Grid,
		device: Device,
		restart: usize,
		tol: f64,
		max_iter: usize,
	) -> Result<Self> {
		let m = restart;
		let mut vv = Vec::with_capacity(m + 1);
		for _ in 0..=m {
			vv.push(Field::zeros(grid, DType::F64, device)?);
		}
		let w = Field::zeros(grid, DType::F64, device)?;

		Ok(Self {
			vv,
			w,
			hh: vec![0.0; (m + 1) * m],
			cc: vec![0.0; m],
			ss: vec![0.0; m],
			grs: vec![0.0; m + 1],
			restart: m,
			tol,
			max_iter,
		})
	}

	/// Access Hessenberg element H(i, j).
	/// Column-major: index = j * (m+1) + i.
	#[inline]
	fn hh(&self, i: usize, j: usize) -> f64 {
		self.hh[j * (self.restart + 1) + i]
	}

	#[inline]
	fn hh_set(
		&mut self,
		i: usize,
		j: usize,
		val: f64,
	) {
		self.hh[j * (self.restart + 1) + i] = val;
	}

	/// Apply operator: `out = inp + coeff·Δ(inp)`.
	fn apply_op(
		out: &mut Field,
		inp: &Field,
		coeff: f64,
		stencil: &Stencil,
		boundaries: &Boundaries,
	) -> Result<()> {
		out.copy_from(inp)?;
		out.stencil_axpy(coeff, inp, stencil, boundaries)
	}

	/// Solve `A·x = b` (unpreconditioned GMRES(m)).
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
		let m = self.restart;
		let mut total_its = 0_usize;

		loop {
			// ── Compute initial residual ────────────
			// v₀ = b - A·x
			Self::apply_op(
				&mut self.vv[0],
				x,
				coeff,
				stencil,
				boundaries,
			)?;
			self.vv[0].scale(-1.0)?;
			self.vv[0].axpy(1.0, b)?;

			let beta = self.vv[0].dot(&self.vv[0])?.sqrt();
			if beta < self.tol {
				return Ok(CgStats {
					iterations: total_its,
					residual: beta,
					reason: ConvergenceReason::Converged,
				});
			}

			// Normalize v₀.
			self.vv[0].scale(1.0 / beta)?;

			// Initialize least-squares RHS.
			self.grs.fill(0.0);
			self.grs[0] = beta;

			// Clear Hessenberg.
			self.hh.fill(0.0);

			let mut cur_resid = beta;
			let mut k = 0;

			// ── Arnoldi iteration ───────────────────
			while k < m {
				if total_its >= self.max_iter {
					// Build best solution so far.
					self.build_solution(x, k)?;
					return Ok(CgStats {
						iterations: total_its,
						residual: cur_resid,
						reason:
							ConvergenceReason::MaxIterations,
					});
				}

				// w = A·v_k
				Self::apply_op(
					&mut self.w,
					&self.vv[k],
					coeff,
					stencil,
					boundaries,
				)?;

				// Modified Gram-Schmidt orthogonalization.
				for j in 0..=k {
					let h = self.w.dot(&self.vv[j])?;
					self.hh_set(j, k, h);
					self.w.axpy(-h, &self.vv[j])?;
				}

				let h_kp1_k =
					self.w.dot(&self.w)?.sqrt();
				self.hh_set(k + 1, k, h_kp1_k);

				// Check for happy breakdown.
				if h_kp1_k > 1e-30 {
					// v_{k+1} = w / h_{k+1,k}
					self.vv[k + 1].copy_from(&self.w)?;
					self.vv[k + 1]
						.scale(1.0 / h_kp1_k)?;
				}

				// Apply previous Givens rotations
				// to column k.
				for j in 0..k {
					let h_j =
						self.hh(j, k);
					let h_jp1 =
						self.hh(j + 1, k);
					let cj = self.cc[j];
					let sj = self.ss[j];
					self.hh_set(
						j,
						k,
						cj.mul_add(h_j, sj * h_jp1),
					);
					self.hh_set(
						j + 1,
						k,
						cj.mul_add(h_jp1, -sj * h_j),
					);
				}

				// Compute new Givens rotation for row k.
				let a = self.hh(k, k);
				let b_val = self.hh(k + 1, k);
				let r = a.hypot(b_val);
				if r.abs() < 1e-30 {
					// Breakdown.
					self.build_solution(x, k)?;
					return Ok(CgStats {
						iterations: total_its,
						residual: cur_resid,
						reason:
							ConvergenceReason::Breakdown,
					});
				}
				self.cc[k] = a / r;
				self.ss[k] = b_val / r;

				// Apply to Hessenberg.
				self.hh_set(k, k, r);
				self.hh_set(k + 1, k, 0.0);

				// Apply to RHS.
				let g_k = self.grs[k];
				self.grs[k] = self.cc[k] * g_k;
				self.grs[k + 1] = -self.ss[k] * g_k;

				cur_resid = self.grs[k + 1].abs();
				total_its += 1;
				k += 1;

				if cur_resid < self.tol {
					break;
				}

				if h_kp1_k <= 1e-30 {
					// Happy breakdown: Krylov subspace
					// contains the solution.
					break;
				}
			}

			// Build solution from Krylov basis.
			self.build_solution(x, k)?;

			if cur_resid < self.tol {
				return Ok(CgStats {
					iterations: total_its,
					residual: cur_resid,
					reason: ConvergenceReason::Converged,
				});
			}

			if total_its >= self.max_iter {
				return Ok(CgStats {
					iterations: total_its,
					residual: cur_resid,
					reason: ConvergenceReason::MaxIterations,
				});
			}

			// Not converged — restart with updated x.
		}
	}

	/// Solve with left preconditioning.
	///
	/// Solves `M⁻¹·A·x = M⁻¹·b`.
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
		let m = self.restart;
		let mut total_its = 0_usize;

		loop {
			// v₀ = M⁻¹·(b - A·x)
			Self::apply_op(
				&mut self.w,
				x,
				coeff,
				stencil,
				boundaries,
			)?;
			self.w.scale(-1.0)?;
			self.w.axpy(1.0, b)?;
			// w = b - A·x (unpreconditioned residual)
			self.vv[0].fill(0.0)?;
			pc.apply(&mut self.vv[0], &self.w)?;
			// vv[0] = M⁻¹·r

			let beta =
				self.vv[0].dot(&self.vv[0])?.sqrt();
			if beta < self.tol {
				return Ok(CgStats {
					iterations: total_its,
					residual: beta,
					reason: ConvergenceReason::Converged,
				});
			}

			self.vv[0].scale(1.0 / beta)?;
			self.grs.fill(0.0);
			self.grs[0] = beta;
			self.hh.fill(0.0);

			let mut cur_resid = beta;
			let mut k = 0;

			while k < m {
				if total_its >= self.max_iter {
					self.build_solution(x, k)?;
					return Ok(CgStats {
						iterations: total_its,
						residual: cur_resid,
						reason:
							ConvergenceReason::MaxIterations,
					});
				}

				// w = A·v_k
				Self::apply_op(
					&mut self.w,
					&self.vv[k],
					coeff,
					stencil,
					boundaries,
				)?;
				// w = M⁻¹·A·v_k
				// Use vv[k+1] as temp to avoid extra buffer.
				self.vv[k + 1].fill(0.0)?;
				pc.apply(&mut self.vv[k + 1], &self.w)?;
				// vv[k+1] = M⁻¹·w  (will be overwritten
				// after MGS if not the final basis vector)

				// Copy preconditioned result to w for MGS.
				self.w.copy_from(&self.vv[k + 1])?;

				// Modified Gram-Schmidt.
				for j in 0..=k {
					let h = self.w.dot(&self.vv[j])?;
					self.hh_set(j, k, h);
					self.w.axpy(-h, &self.vv[j])?;
				}

				let h_kp1_k =
					self.w.dot(&self.w)?.sqrt();
				self.hh_set(k + 1, k, h_kp1_k);

				if h_kp1_k > 1e-30 {
					self.vv[k + 1].copy_from(&self.w)?;
					self.vv[k + 1]
						.scale(1.0 / h_kp1_k)?;
				}

				// Givens rotations (same as unpreconditioned).
				for j in 0..k {
					let h_j = self.hh(j, k);
					let h_jp1 = self.hh(j + 1, k);
					let cj = self.cc[j];
					let sj = self.ss[j];
					self.hh_set(
						j,
						k,
						cj.mul_add(h_j, sj * h_jp1),
					);
					self.hh_set(
						j + 1,
						k,
						cj.mul_add(h_jp1, -sj * h_j),
					);
				}

				let a = self.hh(k, k);
				let b_val = self.hh(k + 1, k);
				let r = a.hypot(b_val);
				if r.abs() < 1e-30 {
					self.build_solution(x, k)?;
					return Ok(CgStats {
						iterations: total_its,
						residual: cur_resid,
						reason:
							ConvergenceReason::Breakdown,
					});
				}
				self.cc[k] = a / r;
				self.ss[k] = b_val / r;
				self.hh_set(k, k, r);
				self.hh_set(k + 1, k, 0.0);

				let g_k = self.grs[k];
				self.grs[k] = self.cc[k] * g_k;
				self.grs[k + 1] = -self.ss[k] * g_k;

				cur_resid = self.grs[k + 1].abs();
				total_its += 1;
				k += 1;

				if cur_resid < self.tol {
					break;
				}
				if h_kp1_k <= 1e-30 {
					break;
				}
			}

			self.build_solution(x, k)?;

			if cur_resid < self.tol {
				return Ok(CgStats {
					iterations: total_its,
					residual: cur_resid,
					reason: ConvergenceReason::Converged,
				});
			}
			if total_its >= self.max_iter {
				return Ok(CgStats {
					iterations: total_its,
					residual: cur_resid,
					reason: ConvergenceReason::MaxIterations,
				});
			}
		}
	}

	/// Back-substitution on the upper-triangular Hessenberg
	/// matrix, then update x from Krylov basis vectors.
	///
	/// x += y₀·v₀ + y₁·v₁ + ... + y_{k-1}·v_{k-1}
	/// where H·y = grs (upper triangular solve).
	fn build_solution(
		&self,
		x: &mut Field,
		k: usize,
	) -> Result<()> {
		if k == 0 {
			return Ok(());
		}

		// Back-substitution: H(0..k, 0..k)·y = grs(0..k).
		let mp1 = self.restart + 1;
		let mut y = vec![0.0_f64; k];
		for i in (0..k).rev() {
			let mut s = self.grs[i];
			for j in (i + 1)..k {
				s -= self.hh[j * mp1 + i] * y[j];
			}
			let diag = self.hh[i * mp1 + i];
			if diag.abs() < 1e-30 {
				break;
			}
			y[i] = s / diag;
		}

		// x += sum(y_i · v_i)
		for i in 0..k {
			x.axpy(y[i], &self.vv[i])?;
		}
		Ok(())
	}
}
