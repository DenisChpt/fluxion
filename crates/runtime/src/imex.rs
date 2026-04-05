use fluxion_core::{Backend, Boundaries, DType, Grid, Stencil};

use crate::cg::{CgSolver, CgStats};
use crate::device::Device;
use crate::error::Result;
use crate::field::Field;
use crate::multigrid::Multigrid;
use crate::storage::BufferStorage;

/// IMEX (Implicit-Explicit) convection-diffusion solver.
///
/// Solves `∂u/∂t = κ·Δu + v·∇u` where:
/// - Diffusion `κ·Δu` is treated **implicitly** (Crank-Nicolson)
///   → no CFL constraint from diffusion
/// - Convection `v·∇u` is treated **explicitly** (forward Euler)
///   → CFL constraint only from advection: `dt < dx/|v|`
///
/// This is IMEX-Euler(1,1):
/// ```text
/// rhs = u^n + dt·(v·∇u^n) + (dt/2)·κ·Δ(u^n)
/// solve (I - (dt/2)·κ·Δ) u^{n+1} = rhs
/// ```
///
/// All buffers pre-allocated. Zero allocation in hot path.
#[derive(Debug)]
pub struct ImexSolver {
	stencil: Stencil,
	boundaries: Boundaries,
	cg: CgSolver,
	mg: Option<Multigrid>,
	/// RHS for the implicit solve.
	rhs: Field,
	/// Convection result: v·∇u.
	conv: Field,
	/// Zero kappa field (for pure convection computation).
	kappa_zero: Field,
	/// Velocity fields.
	vx: Field,
	vy: Field,
	dt: f64,
	kappa: f64,
	steps_done: usize,
}

impl ImexSolver {
	/// Create an IMEX solver for convection-diffusion.
	///
	/// `kappa` — scalar diffusion coefficient.
	/// `vx`, `vy` — uniform velocity components.
	///
	/// # Errors
	/// Returns an error if allocation fails.
	#[allow(clippy::too_many_arguments)]
	pub fn new(
		grid: Grid,
		kappa: f64,
		vx_val: f64,
		vy_val: f64,
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
		let conv =
			Field::zeros(grid, DType::F64, device)?;
		let kappa_zero =
			Field::zeros(grid, DType::F64, device)?;

		let mut vx =
			Field::zeros(grid, DType::F64, device)?;
		vx.fill(vx_val)?;
		let mut vy =
			Field::zeros(grid, DType::F64, device)?;
		vy.fill(vy_val)?;

		Ok(Self {
			stencil,
			boundaries,
			cg,
			mg: None,
			rhs,
			conv,
			kappa_zero,
			vx,
			vy,
			dt,
			kappa,
			steps_done: 0,
		})
	}

	/// Create with multigrid-preconditioned CG.
	///
	/// # Errors
	/// Returns an error if allocation fails.
	#[allow(clippy::too_many_arguments)]
	pub fn with_multigrid(
		grid: Grid,
		kappa: f64,
		vx_val: f64,
		vy_val: f64,
		dt: f64,
		boundaries: Boundaries,
		device: Device,
		cg_tol: f64,
		cg_max_iter: usize,
		mg_pre: usize,
		mg_post: usize,
		mg_omega: f64,
	) -> Result<Self> {
		let mut s = Self::new(
			grid, kappa, vx_val, vy_val, dt,
			boundaries.clone(), device, cg_tol,
			cg_max_iter,
		)?;
		s.mg = Some(Multigrid::new(
			grid, boundaries, device, mg_pre, mg_post,
			mg_omega,
		)?);
		Ok(s)
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
	) -> Result<CgStats> {
		let half_dt_kappa = 0.5 * self.dt * self.kappa;

		// ── Explicit convection: conv = v·∇u ────────
		// Use apply_conv_diff with kappa=0 to get pure
		// convection (upwind differencing).
		self.compute_convection(u)?;

		// ── Build RHS ───────────────────────────────
		// rhs = u^n + half_dt_kappa·Δ(u^n) + dt·conv
		self.rhs.copy_from(u)?;
		self.rhs.stencil_axpy(
			half_dt_kappa,
			u,
			&self.stencil,
			&self.boundaries,
		)?;
		self.rhs.axpy(self.dt, &self.conv)?;

		// ── Implicit diffusion solve ────────────────
		// (I - half_dt_kappa·Δ) u^{n+1} = rhs
		let coeff = -half_dt_kappa;
		let stats = if let Some(ref mut mg) = self.mg {
			self.cg.solve_preconditioned(
				u,
				&self.rhs,
				coeff,
				&self.stencil,
				&self.boundaries,
				mg,
			)?
		} else {
			self.cg.solve(
				u,
				&self.rhs,
				coeff,
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
			self.step(u)?;
		}
		Ok(())
	}

	/// Compute pure convection: conv = v·∇u.
	///
	/// Uses `apply_conv_diff` with kappa=0 (the `kappa_zero`
	/// field), so only the advection term is computed.
	fn compute_convection(
		&mut self,
		u: &Field,
	) -> Result<()> {
		let grid = *u.grid();
		match (
			&u.storage,
			&mut self.conv.storage,
			&self.kappa_zero.storage,
			&self.vx.storage,
			&self.vy.storage,
		) {
			#[cfg(feature = "cpu")]
			(
				BufferStorage::Cpu(u_buf),
				BufferStorage::Cpu(c_buf),
				BufferStorage::Cpu(k_buf),
				BufferStorage::Cpu(vx_buf),
				BufferStorage::Cpu(vy_buf),
			) => {
				crate::storage::cpu_backend()
					.apply_conv_diff(
						u_buf,
						c_buf,
						k_buf,
						vx_buf,
						vy_buf,
						&grid,
						&self.boundaries,
					)?;
			}
			#[cfg(feature = "hip")]
			(
				BufferStorage::Hip(u_buf),
				BufferStorage::Hip(c_buf),
				BufferStorage::Hip(k_buf),
				BufferStorage::Hip(vx_buf),
				BufferStorage::Hip(vy_buf),
			) => {
				crate::storage::hip_backend()
					.apply_conv_diff(
						u_buf,
						c_buf,
						k_buf,
						vx_buf,
						vy_buf,
						&grid,
						&self.boundaries,
					)?;
			}
			#[allow(unreachable_patterns)]
			_ => {}
		}
		Ok(())
	}
}
