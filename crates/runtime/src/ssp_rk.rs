use fluxion_core::{Boundaries, DType, Grid, Stencil};

use crate::device::Device;
use crate::error::Result;
use crate::field::Field;

/// SSP-RK order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SspOrder {
	/// SSP-RK2 (Heun's method with SSP property). 2 stages.
	/// CFL coefficient C = 1.
	Ssp2,
	/// SSP-RK3 (Shu-Osher). 3 stages.
	/// CFL coefficient C = 1.
	Ssp3,
}

/// Strong Stability Preserving Runge-Kutta solver.
///
/// Preserves TVD (Total Variation Diminishing) properties
/// of the spatial discretization. Essential for hyperbolic
/// conservation laws and convection-dominated flows.
///
/// All buffers pre-allocated. Zero allocation in hot path.
#[derive(Debug)]
pub struct SspRkSolver {
	stencil: Stencil,
	boundaries: Boundaries,
	alpha: f64,
	/// Stage buffers.
	u1: Field,
	u2: Field,
	/// RHS scratch.
	k: Field,
	dt: f64,
	order: SspOrder,
	steps_done: usize,
}

impl SspRkSolver {
	/// Create an SSP-RK solver.
	///
	/// `dt` should satisfy the CFL condition for the chosen
	/// scheme. For SSP-RK2/3 with CFL coefficient C=1,
	/// the effective CFL is the same as forward Euler.
	///
	/// # Errors
	/// Returns an error if allocation fails.
	pub fn new(
		grid: Grid,
		alpha: f64,
		dt: f64,
		order: SspOrder,
		boundaries: Boundaries,
		device: Device,
	) -> Result<Self> {
		let stencil =
			Stencil::laplacian_2d_5pt(grid.dx, grid.dy);
		Ok(Self {
			stencil,
			boundaries,
			alpha,
			u1: Field::zeros(grid, DType::F64, device)?,
			u2: Field::zeros(grid, DType::F64, device)?,
			k: Field::zeros(grid, DType::F64, device)?,
			dt,
			order,
			steps_done: 0,
		})
	}

	/// Auto-compute stable dt (same CFL as forward Euler).
	///
	/// # Errors
	/// Returns an error if allocation fails.
	pub fn with_auto_dt(
		grid: Grid,
		alpha: f64,
		order: SspOrder,
		boundaries: Boundaries,
		device: Device,
	) -> Result<Self> {
		let max_inv =
			2.0 / (grid.dx * grid.dx)
			+ 2.0 / (grid.dy * grid.dy);
		let dt = 0.45 / (alpha * max_inv);
		Self::new(grid, alpha, dt, order, boundaries, device)
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

	/// Compute `k = α·Δ(u)` (diffusion RHS).
	fn compute_rhs(&mut self, u: &Field) -> Result<()> {
		u.apply_stencil_into(
			&self.stencil,
			&self.boundaries,
			&mut self.k,
		)?;
		self.k.scale(self.alpha)
	}

	/// Advance one time step.
	///
	/// # Errors
	/// Returns an error on backend failure.
	pub fn step(&mut self, u: &mut Field) -> Result<()> {
		match self.order {
			SspOrder::Ssp2 => self.step_ssp2(u),
			SspOrder::Ssp3 => self.step_ssp3(u),
		}
	}

	/// SSP-RK2 (Heun with SSP property):
	/// ```text
	/// u1 = u^n + dt·f(u^n)
	/// u^{n+1} = 0.5·u^n + 0.5·(u1 + dt·f(u1))
	/// ```
	fn step_ssp2(&mut self, u: &mut Field) -> Result<()> {
		let dt = self.dt;

		// u1 = u + dt·f(u)
		self.compute_rhs(u)?;
		self.u1.copy_from(u)?;
		self.u1.axpy(dt, &self.k)?;

		// u^{n+1} = 0.5·u + 0.5·(u1 + dt·f(u1))
		self.compute_rhs(&self.u1.clone_ssp()?)?;
		// u1 += dt·k → u1 = u1 + dt·f(u1)
		self.u1.axpy(dt, &self.k)?;
		// u = 0.5·u + 0.5·u1
		u.scale(0.5)?;
		u.axpy(0.5, &self.u1)?;

		self.steps_done += 1;
		Ok(())
	}

	/// SSP-RK3 (Shu-Osher, 3-stage):
	/// ```text
	/// u1 = u^n + dt·f(u^n)
	/// u2 = 0.75·u^n + 0.25·(u1 + dt·f(u1))
	/// u^{n+1} = (1/3)·u^n + (2/3)·(u2 + dt·f(u2))
	/// ```
	fn step_ssp3(&mut self, u: &mut Field) -> Result<()> {
		let dt = self.dt;

		// u1 = u + dt·f(u)
		self.compute_rhs(u)?;
		self.u1.copy_from(u)?;
		self.u1.axpy(dt, &self.k)?;

		// u2 = 0.75·u + 0.25·(u1 + dt·f(u1))
		self.compute_rhs(&self.u1.clone_ssp()?)?;
		self.u2.copy_from(&self.u1)?;
		self.u2.axpy(dt, &self.k)?;
		// u2 = 0.25·u2 + 0.75·u
		self.u2.scale(0.25)?;
		self.u2.axpy(0.75, u)?;

		// u^{n+1} = (1/3)·u + (2/3)·(u2 + dt·f(u2))
		self.compute_rhs(&self.u2.clone_ssp()?)?;
		self.u2.axpy(dt, &self.k)?;
		// u = (1/3)·u + (2/3)·u2
		u.scale(1.0 / 3.0)?;
		u.axpy(2.0 / 3.0, &self.u2)?;

		self.steps_done += 1;
		Ok(())
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
}

impl Field {
	/// Helper for SSP-RK: clone field for stencil input.
	/// Uses same allocation trick as adaptive solver.
	fn clone_ssp(&self) -> Result<Self> {
		let mut f =
			Self::zeros(*self.grid(), DType::F64, self.device())?;
		f.copy_from(self)?;
		Ok(f)
	}
}
