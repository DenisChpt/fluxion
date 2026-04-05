use fluxion_core::{Boundaries, DType, Grid, Stencil};

use crate::cg::{CgSolver, CgStats};
use crate::device::Device;
use crate::error::Result;
use crate::field::Field;
use crate::multigrid::Multigrid;

/// BDF (Backward Differentiation Formulas) solver for stiff
/// diffusion problems.
///
/// Solves `∂u/∂t = α·Δu` using implicit multi-step methods
/// of order 1–4. Higher order = more accurate but needs more
/// history steps.
///
/// BDF1 = Backward Euler: `(u^{n+1} - u^n)/dt = α·Δu^{n+1}`
/// BDF2: `(3u^{n+1} - 4u^n + u^{n-1})/(2dt) = α·Δu^{n+1}`
/// BDF3, BDF4 follow the same pattern with more history.
///
/// Each step requires solving a linear system via CG.
/// All history buffers are pre-allocated.
#[derive(Debug)]
pub struct BdfSolver {
	stencil: Stencil,
	boundaries: Boundaries,
	cg: CgSolver,
	mg: Option<Multigrid>,
	/// History buffers: u^n, u^{n-1}, u^{n-2}, u^{n-3}.
	history: Vec<Field>,
	/// RHS scratch.
	rhs: Field,
	dt: f64,
	alpha: f64,
	/// BDF order (1–4).
	order: usize,
	steps_done: usize,
}

/// BDF coefficients: a_0·u^{n+1} + a_1·u^n + ... = dt·f^{n+1}
/// The coefficients for BDF1–BDF4.
fn bdf_coeffs(order: usize) -> (&'static [f64], f64) {
	// Returns (history_coeffs, lhs_coeff) where:
	//   lhs_coeff · u^{n+1} = dt·α·Δu^{n+1} - sum(a_i · u^{n+1-i})
	// Rearranged: (lhs_coeff·I - dt·α·Δ) u^{n+1} = -sum(a_i · u^{n+1-i})
	match order {
		// BDF1: u^{n+1} - u^n = dt·f^{n+1}
		1 => (&[-1.0], 1.0),
		// BDF2: (3/2)u^{n+1} - 2u^n + (1/2)u^{n-1} = dt·f
		2 => (&[-2.0, 0.5], 1.5),
		// BDF3: (11/6)u^{n+1} - 3u^n + (3/2)u^{n-1} - (1/3)u^{n-2}
		3 => (&[-3.0, 1.5, -1.0 / 3.0], 11.0 / 6.0),
		// BDF4: (25/12)u^{n+1} - 4u^n + 3u^{n-1} - (4/3)u^{n-2} + (1/4)u^{n-3}
		4 => (&[-4.0, 3.0, -4.0 / 3.0, 0.25], 25.0 / 12.0),
		_ => unreachable!(),
	}
}

impl BdfSolver {
	/// Create a BDF solver of the given order (1–4).
	///
	/// # Panics
	/// Panics if `order` is not in 1..=4.
	///
	/// # Errors
	/// Returns an error if allocation fails.
	#[allow(clippy::too_many_arguments)]
	pub fn new(
		grid: Grid,
		alpha: f64,
		dt: f64,
		order: usize,
		boundaries: Boundaries,
		device: Device,
		cg_tol: f64,
		cg_max_iter: usize,
	) -> Result<Self> {
		assert!(
			(1..=4).contains(&order),
			"BDF order must be 1–4, got {order}"
		);
		let stencil =
			Stencil::laplacian_2d_5pt(grid.dx, grid.dy);
		let cg =
			CgSolver::new(grid, device, cg_tol, cg_max_iter)?;
		let rhs =
			Field::zeros(grid, DType::F64, device)?;

		// Need `order` history buffers.
		let mut history = Vec::with_capacity(order);
		for _ in 0..order {
			history.push(
				Field::zeros(grid, DType::F64, device)?,
			);
		}

		Ok(Self {
			stencil,
			boundaries,
			cg,
			mg: None,
			history,
			rhs,
			dt,
			alpha,
			order,
			steps_done: 0,
		})
	}

	/// Create with multigrid preconditioner.
	///
	/// # Errors
	/// Returns an error if allocation fails.
	#[allow(clippy::too_many_arguments)]
	pub fn with_multigrid(
		grid: Grid,
		alpha: f64,
		dt: f64,
		order: usize,
		boundaries: Boundaries,
		device: Device,
		cg_tol: f64,
		cg_max_iter: usize,
		mg_pre: usize,
		mg_post: usize,
		mg_omega: f64,
	) -> Result<Self> {
		let mut s = Self::new(
			grid, alpha, dt, order, boundaries.clone(),
			device, cg_tol, cg_max_iter,
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
	pub const fn order(&self) -> usize {
		self.order
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
	/// For the first `order-1` steps, automatically falls back
	/// to lower-order BDF (BDF1 for step 1, BDF2 for step 2,
	/// etc.) to bootstrap the history.
	///
	/// # Errors
	/// Returns an error on backend failure.
	pub fn step(
		&mut self,
		u: &mut Field,
	) -> Result<CgStats> {
		// Effective order: min(requested, steps_done + 1).
		let eff_order =
			self.order.min(self.steps_done + 1);
		let (hist_coeffs, lhs_coeff) =
			bdf_coeffs(eff_order);

		// Build RHS = -sum(a_i · u^{n+1-i})
		// history[0] = u^n, history[1] = u^{n-1}, etc.
		// But first, shift history: push u into history[0],
		// shift others down.
		// Actually, we build RHS from current u and history,
		// THEN update history after the solve.

		// rhs = -hist_coeffs[0] · u^n
		self.rhs.copy_from(u)?;
		self.rhs.scale(-hist_coeffs[0])?;

		// rhs -= hist_coeffs[i] · history[i-1] for i >= 1
		for (i, &c) in hist_coeffs.iter().enumerate().skip(1)
		{
			self.rhs.axpy(-c, &self.history[i - 1])?;
		}

		// Solve: (lhs_coeff·I - dt·α·Δ) u^{n+1} = rhs
		// This is: A·v = v + coeff·Δ(v) with:
		//   A = lhs_coeff · I - dt·α·Δ
		// Rewrite as: lhs_coeff · (I - (dt·α/lhs_coeff)·Δ) u = rhs
		// → (I + coeff·Δ) u = rhs/lhs_coeff
		// where coeff = -dt·α/lhs_coeff
		let coeff = -self.dt * self.alpha / lhs_coeff;
		self.rhs.scale(1.0 / lhs_coeff)?;

		// Shift history before solving (u is about to be overwritten).
		for i in (1..self.history.len()).rev() {
			// history[i] = history[i-1]
			let (left, right) =
				self.history.split_at_mut(i);
			right[0].copy_from(&left[i - 1])?;
		}
		if !self.history.is_empty() {
			self.history[0].copy_from(u)?;
		}

		// Solve.
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
}
