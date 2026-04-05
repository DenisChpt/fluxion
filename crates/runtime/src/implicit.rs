use fluxion_core::{Boundaries, DType, Grid, Stencil};

use crate::cg::{CgSolver, CgStats};
use crate::device::Device;
use crate::error::Result;
use crate::field::Field;
use crate::multigrid::{Multigrid, SmootherKind};

/// Implicit diffusion solver using Crank-Nicolson + CG.
///
/// Solves `∂u/∂t = α·Δu + source` with the scheme:
///
/// ```text
/// (I - dt/2·α·Δ) u^{n+1} = (I + dt/2·α·Δ) u^n + dt·source
/// ```
///
/// The linear system is solved matrix-free via Conjugate
/// Gradient. Time step is **not CFL-limited** — dt can be
/// orders of magnitude larger than explicit methods.
///
/// When a `Multigrid` preconditioner is provided, the solver
/// uses PCG which converges in ~2-3 iterations instead of
/// ~10+, giving a ~3-5x speedup on the linear solve.
#[derive(Debug)]
pub struct CrankNicolsonSolver {
	stencil: Stencil,
	boundaries: Boundaries,
	cg: CgSolver,
	/// Multigrid preconditioner (optional).
	mg: Option<Multigrid>,
	/// Right-hand side buffer.
	rhs: Field,
	/// Scratch for stencil input copy.
	tmp: Field,
	dt: f64,
	alpha: f64,
	steps_done: usize,
}

impl CrankNicolsonSolver {
	/// Create a Crank-Nicolson solver (unpreconditioned CG).
	///
	/// # Errors
	/// Returns an error if allocation fails.
	pub fn new(
		grid: Grid,
		alpha: f64,
		dt: f64,
		boundaries: Boundaries,
		device: Device,
		cg_tol: f64,
		cg_max_iter: usize,
	) -> Result<Self> {
		let stencil =
			Stencil::laplacian_2d_5pt(grid.dx, grid.dy);
		let cg =
			CgSolver::new(grid, device, cg_tol, cg_max_iter)?;
		let rhs =
			Field::zeros(grid, DType::F64, device)?;
		let tmp =
			Field::zeros(grid, DType::F64, device)?;
		Ok(Self {
			stencil,
			boundaries,
			cg,
			mg: None,
			rhs,
			tmp,
			dt,
			alpha,
			steps_done: 0,
		})
	}

	/// Create a Crank-Nicolson solver with multigrid-
	/// preconditioned CG (PCG).
	///
	/// Converges in ~2-3 CG iterations instead of ~10+.
	///
	/// # Errors
	/// Returns an error if allocation fails.
	pub fn with_multigrid(
		grid: Grid,
		alpha: f64,
		dt: f64,
		boundaries: Boundaries,
		device: Device,
		cg_tol: f64,
		cg_max_iter: usize,
		mg_pre: usize,
		mg_post: usize,
		mg_omega: f64,
	) -> Result<Self> {
		Self::with_multigrid_smoother(
			grid,
			alpha,
			dt,
			boundaries,
			device,
			cg_tol,
			cg_max_iter,
			mg_pre,
			mg_post,
			mg_omega,
			SmootherKind::Jacobi,
		)
	}

	/// Like `with_multigrid`, but allows choosing the smoother.
	///
	/// # Errors
	/// Returns an error if allocation fails.
	pub fn with_multigrid_smoother(
		grid: Grid,
		alpha: f64,
		dt: f64,
		boundaries: Boundaries,
		device: Device,
		cg_tol: f64,
		cg_max_iter: usize,
		mg_pre: usize,
		mg_post: usize,
		mg_omega: f64,
		smoother: SmootherKind,
	) -> Result<Self> {
		let stencil =
			Stencil::laplacian_2d_5pt(grid.dx, grid.dy);
		let cg =
			CgSolver::new(grid, device, cg_tol, cg_max_iter)?;
		let mg = Multigrid::build(
			grid,
			boundaries.clone(),
			device,
			mg_pre,
			mg_post,
			mg_omega,
			smoother,
		)?;
		let rhs =
			Field::zeros(grid, DType::F64, device)?;
		let tmp =
			Field::zeros(grid, DType::F64, device)?;
		Ok(Self {
			stencil,
			boundaries,
			cg,
			mg: Some(mg),
			rhs,
			tmp,
			dt,
			alpha,
			steps_done: 0,
		})
	}

	#[inline]
	#[must_use]
	pub const fn dt(&self) -> f64 {
		self.dt
	}

	#[inline]
	#[must_use]
	pub const fn steps_done(&self) -> usize {
		self.steps_done
	}

	#[inline]
	#[must_use]
	pub fn sim_time(&self) -> f64 {
		self.steps_done as f64 * self.dt
	}

	/// Advance one time step. Returns CG stats.
	///
	/// # Errors
	/// Returns an error on backend failure.
	pub fn step(
		&mut self,
		u: &mut Field,
		source: Option<&Field>,
	) -> Result<CgStats> {
		let half_dt_alpha = 0.5 * self.dt * self.alpha;

		// RHS = (I + dt/2·α·Δ) u^n + dt·source
		// = u^n + half_dt_alpha·Δ(u^n) + dt·source
		self.rhs.copy_from(u)?;
		self.tmp.copy_from(u)?;
		self.rhs.stencil_axpy(
			half_dt_alpha,
			&self.tmp,
			&self.stencil,
			&self.boundaries,
		)?;
		if let Some(src) = source {
			self.rhs.axpy(self.dt, src)?;
		}

		// Solve (I - dt/2·α·Δ) u^{n+1} = rhs
		// coeff = -half_dt_alpha (negative!)
		let stats = if let Some(ref mut mg) = self.mg {
			self.cg.solve_preconditioned(
				u,
				&self.rhs,
				-half_dt_alpha,
				&self.stencil,
				&self.boundaries,
				mg,
			)?
		} else {
			self.cg.solve(
				u,
				&self.rhs,
				-half_dt_alpha,
				&self.stencil,
				&self.boundaries,
			)?
		};

		self.steps_done += 1;
		Ok(stats)
	}

	/// Advance `n` time steps.
	///
	/// # Errors
	/// Returns an error on backend failure.
	pub fn step_n(
		&mut self,
		u: &mut Field,
		n: usize,
	) -> Result<()> {
		for _ in 0..n {
			self.step(u, None)?;
		}
		Ok(())
	}
}
