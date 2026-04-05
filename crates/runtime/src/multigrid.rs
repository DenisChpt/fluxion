use fluxion_core::{
	Backend, Boundaries, DType, Grid, Stencil,
};

use crate::device::Device;
use crate::error::Result;
use crate::field::Field;
use crate::preconditioner::Preconditioner;
use crate::storage::BufferStorage;

/// Smoother variant for multigrid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SmootherKind {
	/// Weighted Jacobi (ω typically 2/3). Simple but slow.
	Jacobi,
	/// Chebyshev polynomial smoother. 2-3x faster convergence
	/// than Jacobi for the same iteration count. No reductions
	/// → pure GPU pipelining. Default in algebraic multigrid.
	Chebyshev,
}

/// Buffers for a single multigrid level.
#[derive(Debug)]
struct GridLevel {
	grid: Grid,
	stencil: Stencil,
	/// Solution / correction at this level.
	u: Field,
	/// Right-hand side / residual at this level.
	rhs: Field,
	/// Scratch for residual computation.
	tmp: Field,
	/// Chebyshev: work buffers for 3-term recurrence.
	/// p_km1, p_k, p_kp1 rotate each iteration.
	cheby_w0: Field,
	cheby_w1: Field,
}

/// Geometric multigrid V-cycle preconditioner.
///
/// Uses full-weighting restriction / injection prolongation
/// on a hierarchy of grids from `N×N` down to `4×4`.
///
/// Smoother can be weighted Jacobi or Chebyshev polynomial.
/// All buffers are pre-allocated at construction.
#[derive(Debug)]
pub struct Multigrid {
	levels: Vec<GridLevel>,
	boundaries: Boundaries,
	/// Pre-smoothing iterations.
	n_pre: usize,
	/// Post-smoothing iterations.
	n_post: usize,
	/// Jacobi relaxation weight (typically 2/3).
	omega: f64,
	/// Smoother variant.
	smoother: SmootherKind,
}

impl Multigrid {
	/// Build a multigrid hierarchy.
	///
	/// The finest grid is `grid`. Coarsening halves each
	/// dimension until we reach ≤ 4 in any direction.
	///
	/// # Errors
	/// Returns an error if buffer allocation fails.
	/// Build with weighted Jacobi smoother (default).
	pub fn new(
		grid: Grid,
		boundaries: Boundaries,
		device: Device,
		n_pre: usize,
		n_post: usize,
		omega: f64,
	) -> Result<Self> {
		Self::build(
			grid,
			boundaries,
			device,
			n_pre,
			n_post,
			omega,
			SmootherKind::Jacobi,
		)
	}

	/// Build with a specific smoother kind.
	///
	/// # Errors
	/// Returns an error if buffer allocation fails.
	pub fn build(
		grid: Grid,
		boundaries: Boundaries,
		device: Device,
		n_pre: usize,
		n_post: usize,
		omega: f64,
		smoother: SmootherKind,
	) -> Result<Self> {
		let mut levels = Vec::new();

		let mut g = grid;
		loop {
			let stencil =
				Stencil::laplacian_2d_5pt(g.dx, g.dy);
			levels.push(GridLevel {
				grid: g,
				stencil,
				u: Field::zeros(g, DType::F64, device)?,
				rhs: Field::zeros(g, DType::F64, device)?,
				tmp: Field::zeros(g, DType::F64, device)?,
				cheby_w0: Field::zeros(
					g, DType::F64, device,
				)?,
				cheby_w1: Field::zeros(
					g, DType::F64, device,
				)?,
			});

			if g.rows <= 4 || g.cols <= 4 {
				break;
			}

			let cr = g.rows / 2;
			let cc = g.cols / 2;
			if cr < 3 || cc < 3 {
				break;
			}
			g = Grid::new(
				cr,
				cc,
				g.dx * 2.0,
				g.dy * 2.0,
			)?;
		}

		Ok(Self {
			levels,
			boundaries,
			n_pre,
			n_post,
			omega,
			smoother,
		})
	}

	/// Borrow 3 disjoint buffers (2 shared + 1 mutable)
	/// from the level's u/w0/w1 fields.
	///
	/// idx 0=u, 1=cheby_w0, 2=cheby_w1.
	#[allow(clippy::type_complexity)]
	fn get_three_bufs<'a>(
		u: &'a mut Field,
		w0: &'a mut Field,
		w1: &'a mut Field,
		i_km1: u8,
		i_k: u8,
		i_kp1: u8,
	) -> (&'a Field, &'a Field, &'a mut Field) {
		// Map indices to the 3 field refs.
		// We need i_kp1's field mutably, the others shared.
		match (i_km1, i_k, i_kp1) {
			(0, 1, 2) => (u, w0, w1),
			(1, 2, 0) => (w0, w1, u),
			(2, 0, 1) => (w1, u, w0),
			(0, 2, 1) => (u, w1, w0),
			(1, 0, 2) => (w0, u, w1),
			(2, 1, 0) => (w1, w0, u),
			_ => unreachable!(),
		}
	}

	/// Number of multigrid levels.
	#[must_use]
	pub const fn depth(&self) -> usize {
		self.levels.len()
	}

	/// Apply one V-cycle as a preconditioner.
	///
	/// Solves `A·x ≈ b` approximately where `A` is the
	/// Laplacian operator. `x` is the initial guess (modified
	/// in place), `b` is the RHS on the finest grid.
	///
	/// # Errors
	/// Returns an error on backend failure.
	pub fn v_cycle(
		&mut self,
		x: &mut Field,
		b: &Field,
	) -> Result<()> {
		// Copy b into level 0 rhs, x into level 0 u.
		self.levels[0].rhs.copy_from(b)?;
		self.levels[0].u.copy_from(x)?;

		let n_levels = self.levels.len();

		// ── Downward sweep ──────────────────────────
		for lvl in 0..n_levels - 1 {
			// Pre-smooth.
			self.smooth(lvl, self.n_pre)?;

			// Compute residual: r = rhs - A·u.
			self.compute_residual(lvl)?;

			// Restrict residual to coarser level.
			self.restrict_level(lvl)?;

			// Zero the coarse solution.
			self.levels[lvl + 1].u.fill(0.0)?;
		}

		// ── Coarsest level: smooth heavily ──────────
		let coarsest = n_levels - 1;
		self.smooth(coarsest, self.n_pre + self.n_post + 10)?;

		// ── Upward sweep ────────────────────────────
		for lvl in (0..n_levels - 1).rev() {
			// Prolong correction from coarse to fine.
			self.prolong_level(lvl)?;

			// Post-smooth.
			self.smooth(lvl, self.n_post)?;
		}

		// Copy result back to x.
		x.copy_from(&self.levels[0].u)?;
		Ok(())
	}

	/// Run `n` smoothing steps at `lvl`.
	fn smooth(&mut self, lvl: usize, n: usize) -> Result<()> {
		match self.smoother {
			SmootherKind::Jacobi => {
				self.smooth_jacobi(lvl, n)
			}
			SmootherKind::Chebyshev => {
				self.smooth_chebyshev(lvl, n)
			}
		}
	}

	/// Weighted Jacobi smoother.
	fn smooth_jacobi(
		&mut self,
		lvl: usize,
		n: usize,
	) -> Result<()> {
		let level = &mut self.levels[lvl];
		for _ in 0..n {
			match (
				&mut level.u.storage,
				&level.rhs.storage,
			) {
				#[cfg(feature = "cpu")]
				(
					BufferStorage::Cpu(x),
					BufferStorage::Cpu(b),
				) => {
					crate::storage::cpu_backend()
						.weighted_jacobi(
							x,
							b,
							self.omega,
							&level.grid,
							&level.stencil,
							&self.boundaries,
						)?;
				}
				#[cfg(feature = "wgpu")]
				(
					BufferStorage::Wgpu(x),
					BufferStorage::Wgpu(b),
				) => {
					crate::storage::wgpu_backend()
						.weighted_jacobi(
							x,
							b,
							self.omega,
							&level.grid,
							&level.stencil,
							&self.boundaries,
						)?;
				}
				#[cfg(feature = "hip")]
				(
					BufferStorage::Hip(x),
					BufferStorage::Hip(b),
				) => {
					crate::storage::hip_backend()
						.weighted_jacobi(
							x,
							b,
							self.omega,
							&level.grid,
							&level.stencil,
							&self.boundaries,
						)?;
				}
				#[allow(unreachable_patterns)]
				_ => {}
			}
		}
		Ok(())
	}

	/// First-kind Chebyshev polynomial smoother.
	///
	/// No dot products → no GPU sync → pure pipelining.
	/// Jacobi-preconditioned: eigenvalue bounds are for
	/// `D⁻¹A` where `D = diag(A)`.
	///
	/// Convention: `emin = 0.1·emax`, `emax ≈ 2`
	/// for D⁻¹·(5-pt Laplacian). Three-term recurrence
	/// on 3 rotating solution vectors.
	fn smooth_chebyshev(
		&mut self,
		lvl: usize,
		n: usize,
	) -> Result<()> {
		if n == 0 {
			return Ok(());
		}

		let level = &mut self.levels[lvl];
		let grid = level.grid;

		// Eigenvalue bounds for D⁻¹A (Jacobi-preconditioned).
		// For 5-pt Laplacian: eigenvalues of D⁻¹A ∈ [0, 2].
		// Standard transform: emin = 0.1·emax_est,
		// emax = 1.1·emax_est.
		let emax = 2.0_f64;
		let emin = 0.1 * emax;

		// Chebyshev recurrence parameters.
		let scale = 2.0 / (emax + emin);
		let alpha = 1.0 - scale * emin;
		let mu = 1.0 / alpha;
		let omegaprod = 2.0 / alpha;

		// Diagonal of 5-pt Laplacian = -(2/dx² + 2/dy²).
		let inv_dx2 = 1.0 / (grid.dx * grid.dx);
		let inv_dy2 = 1.0 / (grid.dy * grid.dy);
		let inv_diag =
			-1.0 / (2.0 * inv_dx2 + 2.0 * inv_dy2);

		// Chebyshev coefficient recurrence: c[km1], c[k].
		let mut c_km1 = 1.0_f64;
		let mut c_k = mu;

		// We use 3 rotating solution vectors.
		// p_km1 = u (initial solution, in-place).
		// p_k   = cheby_w0 (work buffer).
		// p_kp1 = cheby_w1 (work buffer).
		// At the end, copy the current solution back to u.

		// ── Iteration 0 ────────────────────────────────
		// r = rhs - A·p[km1]
		level.u.apply_stencil_into(
			&level.stencil,
			&self.boundaries,
			&mut level.tmp,
		)?;
		// tmp = A·u. r = rhs - tmp.
		// B⁻¹·r = D⁻¹·r = inv_diag · (rhs - tmp).
		// p[k] = scale · B⁻¹·r + p[km1]
		//       = scale · inv_diag · (rhs - tmp) + u
		level.cheby_w0.copy_from(&level.rhs)?;
		level.cheby_w0.axpy(-1.0, &level.tmp)?;
		// cheby_w0 = r = rhs - A·u
		level.cheby_w0.scale(scale * inv_diag)?;
		// cheby_w0 = scale · D⁻¹ · r
		level.cheby_w0.axpy(1.0, &level.u)?;
		// cheby_w0 = scale·D⁻¹·r + u = p[k]
		// Enforce zero Dirichlet BCs (global ops corrupt
		// boundary values that stencils read).
		level.cheby_w0.zero_boundaries()?;

		// For n=1 we're done: copy p[k] back to u.
		if n == 1 {
			level.u.copy_from(&level.cheby_w0)?;
			return Ok(());
		}

		// ── Iterations 1..n-1 ───────────────────────────
		// p[km1] is in u, p[k] is in cheby_w0.
		// We rotate: km1=u, k=w0, kp1=w1 → km1=w0, k=w1,
		// kp1=u → etc.
		// Instead of pointer rotation, we'll just swap refs.

		// Buffers: 0=u, 1=cheby_w0, 2=cheby_w1.
		// idx_km1=0, idx_k=1, idx_kp1=2 initially.
		let mut idx_km1 = 0_u8;
		let mut idx_k = 1_u8;
		let mut idx_kp1 = 2_u8;

		for _i in 1..n {
			// r = rhs - A·p[k]
			// Get p[k] into tmp via stencil.
			let pk = match idx_k {
				0 => &level.u,
				1 => &level.cheby_w0,
				_ => &level.cheby_w1,
			};
			pk.apply_stencil_into(
				&level.stencil,
				&self.boundaries,
				&mut level.tmp,
			)?;
			// tmp = A·p[k]. Now r = rhs - tmp.
			// B⁻¹·r = inv_diag · (rhs - tmp).
			// Store B⁻¹·r in p[kp1].
			{
				let pkp1 = match idx_kp1 {
					0 => &mut level.u,
					1 => &mut level.cheby_w0,
					_ => &mut level.cheby_w1,
				};
				pkp1.copy_from(&level.rhs)?;
				pkp1.axpy(-1.0, &level.tmp)?;
				pkp1.scale(inv_diag)?;
			}
			// p[kp1] = D⁻¹·r = B⁻¹·r

			// Chebyshev coefficient update.
			let c_kp1 = 2.0 * mu * c_k - c_km1;
			let omega = omegaprod * c_k / c_kp1;

			// p[kp1] = (1-ω)·p[km1] + ω·p[k]
			//        + ω·Γ·scale·B⁻¹·r
			// where Γ=1, and B⁻¹·r is currently in p[kp1].
			// = ω·scale·p[kp1] + ω·p[k] + (1-ω)·p[km1]
			// kp1 = (1-ω)·km1 + ω·k + ω·scale·kp1
			{
				let (pkm1_s, pk_s, pkp1_m) =
					Self::get_three_bufs(
						&mut level.u,
						&mut level.cheby_w0,
						&mut level.cheby_w1,
						idx_km1,
						idx_k,
						idx_kp1,
					);
				// pkp1 currently = D⁻¹·r.
				// pkp1 = ω·scale·pkp1 + ω·pk + (1-ω)·pkm1
				pkp1_m.scale(omega * scale)?;
				pkp1_m.axpy(omega, pk_s)?;
				pkp1_m.axpy(1.0 - omega, pkm1_s)?;
				// Enforce BCs on new solution vector.
				pkp1_m.zero_boundaries()?;
			}

			c_km1 = c_k;
			c_k = c_kp1;

			// Rotate indices.
			let tmp_idx = idx_km1;
			idx_km1 = idx_k;
			idx_k = idx_kp1;
			idx_kp1 = tmp_idx;
		}

		// Copy result (p[k]) back to u if it's not already
		// there.
		if idx_k == 1 {
			level.u.copy_from(&level.cheby_w0)?;
		} else if idx_k == 2 {
			level.u.copy_from(&level.cheby_w1)?;
		}

		Ok(())
	}

	/// Compute residual at `lvl`: `tmp = rhs - Δ(u)`.
	fn compute_residual(
		&mut self,
		lvl: usize,
	) -> Result<()> {
		let level = &mut self.levels[lvl];
		// tmp = Δ(u)
		level.u.apply_stencil_into(
			&level.stencil,
			&self.boundaries,
			&mut level.tmp,
		)?;
		// tmp = -Δ(u) + rhs = rhs - Δ(u)
		level.tmp.scale(-1.0)?;
		level.tmp.axpy(1.0, &level.rhs)?;
		Ok(())
	}

	/// Restrict residual from `lvl` to `lvl+1`.
	fn restrict_level(&mut self, lvl: usize) -> Result<()> {
		let (fine_levels, coarse_levels) =
			self.levels.split_at_mut(lvl + 1);
		let fine = &fine_levels[lvl];
		let coarse = &mut coarse_levels[0];

		match (
			&fine.tmp.storage,
			&mut coarse.rhs.storage,
		) {
			#[cfg(feature = "cpu")]
			(
				BufferStorage::Cpu(f),
				BufferStorage::Cpu(c),
			) => {
				crate::storage::cpu_backend().restrict(
					f,
					c,
					&fine.grid,
					&coarse.grid,
				)?;
			}
			#[cfg(feature = "wgpu")]
			(
				BufferStorage::Wgpu(f),
				BufferStorage::Wgpu(c),
			) => {
				crate::storage::wgpu_backend().restrict(
					f,
					c,
					&fine.grid,
					&coarse.grid,
				)?;
			}
			#[cfg(feature = "hip")]
			(
				BufferStorage::Hip(f),
				BufferStorage::Hip(c),
			) => {
				crate::storage::hip_backend().restrict(
					f,
					c,
					&fine.grid,
					&coarse.grid,
				)?;
			}
			#[allow(unreachable_patterns)]
			_ => {}
		}
		Ok(())
	}

	/// Prolong correction from `lvl+1` to `lvl`.
	fn prolong_level(&mut self, lvl: usize) -> Result<()> {
		let (fine_levels, coarse_levels) =
			self.levels.split_at_mut(lvl + 1);
		let fine = &mut fine_levels[lvl];
		let coarse = &coarse_levels[0];

		match (
			&coarse.u.storage,
			&mut fine.u.storage,
		) {
			#[cfg(feature = "cpu")]
			(
				BufferStorage::Cpu(c),
				BufferStorage::Cpu(f),
			) => {
				crate::storage::cpu_backend().prolong(
					c,
					f,
					&coarse.grid,
					&fine.grid,
				)?;
			}
			#[cfg(feature = "wgpu")]
			(
				BufferStorage::Wgpu(c),
				BufferStorage::Wgpu(f),
			) => {
				crate::storage::wgpu_backend().prolong(
					c,
					f,
					&coarse.grid,
					&fine.grid,
				)?;
			}
			#[cfg(feature = "hip")]
			(
				BufferStorage::Hip(c),
				BufferStorage::Hip(f),
			) => {
				crate::storage::hip_backend().prolong(
					c,
					f,
					&coarse.grid,
					&fine.grid,
				)?;
			}
			#[allow(unreachable_patterns)]
			_ => {}
		}
		Ok(())
	}
}

impl Preconditioner for Multigrid {
	fn apply(
		&mut self,
		z: &mut Field,
		r: &Field,
	) -> Result<()> {
		self.v_cycle(z, r)
	}
}
