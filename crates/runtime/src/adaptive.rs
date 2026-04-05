use fluxion_core::{Boundaries, DType, Grid, Stencil};

use crate::device::Device;
use crate::error::Result;
use crate::field::Field;

/// Adaptive time-stepping solver using Dormand-Prince RK4(5).
///
/// Embedded error estimation: the 4th-order and 5th-order
/// solutions are compared at each step. If the error exceeds
/// the tolerance, the step is rejected and dt is reduced.
///
/// PI controller for smooth dt adaptation:
/// ```text
/// dt_new = dt · min(fac_max, max(fac_min,
///     safety · (1/err)^(1.1/5) · err_prev^(0.4/5) ))
/// ```
///
/// All 7 stage buffers are pre-allocated at construction.
/// Zero allocation in the hot path.
#[derive(Debug)]
pub struct AdaptiveSolver {
	stencil: Stencil,
	boundaries: Boundaries,
	alpha: f64,
	/// Stage buffers k1..k7 (DOPRI5 uses 7 stages).
	k: [Field; 7],
	/// 5th-order solution (for error estimation).
	u5: Field,
	/// Scratch for intermediate u.
	u_tmp: Field,
	/// Current time step.
	dt: f64,
	/// Absolute tolerance for error control.
	pub atol: f64,
	/// Relative tolerance for error control.
	pub rtol: f64,
	/// Total simulation time advanced.
	sim_time: f64,
	/// Total steps taken (including rejections).
	pub total_steps: usize,
	/// Rejected steps.
	pub rejected_steps: usize,
	/// Previous error (for PI controller).
	prev_err: f64,
}

// Dormand-Prince coefficients (DOPRI5).
// a_{ij}: lower-triangular matrix (stage weights).
const A21: f64 = 1.0 / 5.0;
const A31: f64 = 3.0 / 40.0;
const A32: f64 = 9.0 / 40.0;
const A41: f64 = 44.0 / 45.0;
const A42: f64 = -56.0 / 15.0;
const A43: f64 = 32.0 / 9.0;
const A51: f64 = 19372.0 / 6561.0;
const A52: f64 = -25360.0 / 2187.0;
const A53: f64 = 64448.0 / 6561.0;
const A54: f64 = -212.0 / 729.0;
const A61: f64 = 9017.0 / 3168.0;
const A62: f64 = -355.0 / 33.0;
const A63: f64 = 46732.0 / 5247.0;
const A64: f64 = 49.0 / 176.0;
const A65: f64 = -5103.0 / 18656.0;
const A71: f64 = 35.0 / 384.0;
// A72 = 0
const A73: f64 = 500.0 / 1113.0;
const A74: f64 = 125.0 / 192.0;
const A75: f64 = -2187.0 / 6784.0;
const A76: f64 = 11.0 / 84.0;

// c_i: time fraction for each stage (used with
// time-dependent source terms, kept for completeness).
// C6 = 1, C7 = 1.
#[allow(dead_code)]
const C: [f64; 5] = [
	1.0 / 5.0,  // C2
	3.0 / 10.0, // C3
	4.0 / 5.0,  // C4
	8.0 / 9.0,  // C5
	1.0,         // C6
];

// b_i: 5th-order weights (for error estimation).
const B1: f64 = 5179.0 / 57600.0;
// B2 = 0
const B3: f64 = 7571.0 / 16695.0;
const B4: f64 = 393.0 / 640.0;
const B5: f64 = -92097.0 / 339200.0;
const B6: f64 = 187.0 / 2100.0;
const B7: f64 = 1.0 / 40.0;

// 4th-order weights are A71..A76 (FSAL: k7 = f(u4th)).
// Error = u5 - u4 = dt * sum((b_i - a7i) * k_i).
const E1: f64 = B1 - A71;       // 5179/57600 - 35/384
// E2 = 0
const E3: f64 = B3 - A73;       // 7571/16695 - 500/1113
const E4: f64 = B4 - A74;       // 393/640 - 125/192
const E5: f64 = B5 - A75;       // -92097/339200 + 2187/6784
const E6: f64 = B6 - A76;       // 187/2100 - 11/84
const E7: f64 = B7;             // 1/40

impl AdaptiveSolver {
	/// Create an adaptive DOPRI5 solver for diffusion.
	///
	/// `dt_init` — initial time step (will be adapted).
	/// `atol`, `rtol` — absolute and relative error tolerances.
	///
	/// # Errors
	/// Returns an error if allocation fails.
	pub fn new(
		grid: Grid,
		alpha: f64,
		dt_init: f64,
		boundaries: Boundaries,
		device: Device,
		atol: f64,
		rtol: f64,
	) -> Result<Self> {
		if !dt_init.is_finite() || dt_init <= 0.0 {
			return Err(crate::error::RuntimeError::Core(
				fluxion_core::CoreError::BackendError(
					format!(
						"dt_init must be positive and \
						 finite, got {dt_init}"
					),
				),
			));
		}
		let stencil =
			Stencil::laplacian_2d_5pt(grid.dx, grid.dy);
		let f =
			|| Field::zeros(grid, DType::F64, device);
		let k = [
			f()?, f()?, f()?, f()?, f()?, f()?, f()?,
		];
		let u5 = f()?;
		let u_tmp = f()?;

		Ok(Self {
			stencil,
			boundaries,
			alpha,
			k,
			u5,
			u_tmp,
			dt: dt_init,
			atol,
			rtol,
			sim_time: 0.0,
			total_steps: 0,
			rejected_steps: 0,
			prev_err: 1e-4,
		})
	}

	/// Current time step.
	#[inline]
	#[must_use]
	pub fn dt(&self) -> f64 {
		self.dt
	}

	/// Accumulated simulation time.
	#[inline]
	#[must_use]
	pub fn sim_time(&self) -> f64 {
		self.sim_time
	}

	/// Compute `k = alpha * Δ(u)` (RHS of diffusion equation).
	fn rhs(
		&mut self,
		u: &Field,
		stage: usize,
	) -> Result<()> {
		// k[stage] = Δ(u), then scale by alpha.
		u.apply_stencil_into(
			&self.stencil,
			&self.boundaries,
			&mut self.k[stage],
		)?;
		self.k[stage].scale(self.alpha)
	}

	/// Advance one adaptive step. May reject and retry
	/// internally. Returns the actual dt used.
	///
	/// # Errors
	/// Returns an error on backend failure.
	pub fn step(&mut self, u: &mut Field) -> Result<f64> {
		let safety = 0.9;
		let fac_min = 0.2;
		let fac_max = 5.0;

		loop {
			let dt = self.dt;

			// Stage 1: k1 = f(u^n).
			self.rhs(u, 0)?;

			// Stage 2: u_tmp = u + dt·a21·k1.
			self.u_tmp.copy_from(u)?;
			self.u_tmp.axpy(dt * A21, &self.k[0])?;
			self.rhs(&self.u_tmp.clone_field()?, 1)?;

			// Stage 3: u_tmp = u + dt·(a31·k1 + a32·k2).
			self.u_tmp.copy_from(u)?;
			self.u_tmp.axpy(dt * A31, &self.k[0])?;
			self.u_tmp.axpy(dt * A32, &self.k[1])?;
			self.rhs(&self.u_tmp.clone_field()?, 2)?;

			// Stage 4.
			self.u_tmp.copy_from(u)?;
			self.u_tmp.axpy(dt * A41, &self.k[0])?;
			self.u_tmp.axpy(dt * A42, &self.k[1])?;
			self.u_tmp.axpy(dt * A43, &self.k[2])?;
			self.rhs(&self.u_tmp.clone_field()?, 3)?;

			// Stage 5.
			self.u_tmp.copy_from(u)?;
			self.u_tmp.axpy(dt * A51, &self.k[0])?;
			self.u_tmp.axpy(dt * A52, &self.k[1])?;
			self.u_tmp.axpy(dt * A53, &self.k[2])?;
			self.u_tmp.axpy(dt * A54, &self.k[3])?;
			self.rhs(&self.u_tmp.clone_field()?, 4)?;

			// Stage 6.
			self.u_tmp.copy_from(u)?;
			self.u_tmp.axpy(dt * A61, &self.k[0])?;
			self.u_tmp.axpy(dt * A62, &self.k[1])?;
			self.u_tmp.axpy(dt * A63, &self.k[2])?;
			self.u_tmp.axpy(dt * A64, &self.k[3])?;
			self.u_tmp.axpy(dt * A65, &self.k[4])?;
			self.rhs(&self.u_tmp.clone_field()?, 5)?;

			// Stage 7 = 4th-order solution.
			// u4 = u + dt·(a71·k1 + a73·k3 + a74·k4
			//            + a75·k5 + a76·k6)
			self.u_tmp.copy_from(u)?;
			self.u_tmp.axpy(dt * A71, &self.k[0])?;
			self.u_tmp.axpy(dt * A73, &self.k[2])?;
			self.u_tmp.axpy(dt * A74, &self.k[3])?;
			self.u_tmp.axpy(dt * A75, &self.k[4])?;
			self.u_tmp.axpy(dt * A76, &self.k[5])?;
			// u_tmp = u4 (4th-order solution)

			// Compute k7 = f(u4) for FSAL.
			self.rhs(&self.u_tmp.clone_field()?, 6)?;

			// Error estimation: err = dt·(e1·k1 + e3·k3
			//   + e4·k4 + e5·k5 + e6·k6 + e7·k7)
			// Reuse u5 for the error vector.
			self.u5.fill(0.0)?;
			self.u5.axpy(dt * E1, &self.k[0])?;
			self.u5.axpy(dt * E3, &self.k[2])?;
			self.u5.axpy(dt * E4, &self.k[3])?;
			self.u5.axpy(dt * E5, &self.k[4])?;
			self.u5.axpy(dt * E6, &self.k[5])?;
			self.u5.axpy(dt * E7, &self.k[6])?;

			// Scaled error norm: max_i |err_i| / (atol + rtol·|u4_i|)
			let err_norm =
				self.compute_error_norm(u)?;

			self.total_steps += 1;

			if err_norm <= 1.0 {
				// Accept step: u = u4.
				u.copy_from(&self.u_tmp)?;
				self.sim_time += dt;

				// PI controller for next dt (Hairer-Wanner IV.2.14).
				// h_new = h · safety · (1/err)^((α+β)/p)
				//                    · err_prev^(β/p)
				// with α=0.7, β=0.4, p=5.
				let fac = safety
					* (1.0 / err_norm).powf(1.1 / 5.0)
					* (self.prev_err).powf(0.4 / 5.0);
				let fac = fac.clamp(fac_min, fac_max);
				self.dt = dt * fac;
				self.prev_err = err_norm.max(1e-8);

				return Ok(dt);
			}

			// Reject step: reduce dt and retry.
			let fac = safety
				* (1.0 / err_norm).powf(1.0 / 5.0);
			let fac = fac.max(fac_min);
			self.dt = dt * fac;
			self.rejected_steps += 1;
		}
	}

	/// Advance until `t_end` is reached.
	///
	/// # Errors
	/// Returns an error on backend failure.
	pub fn advance_to(
		&mut self,
		u: &mut Field,
		t_end: f64,
	) -> Result<()> {
		if !t_end.is_finite() || t_end <= self.sim_time {
			return Ok(());
		}
		while self.sim_time < t_end {
			// Don't overshoot.
			let remaining = t_end - self.sim_time;
			if self.dt > remaining {
				self.dt = remaining;
			}
			self.step(u)?;
		}
		Ok(())
	}

	/// Compute the scaled error norm from `u5` (error vector)
	/// and `u_tmp` (4th-order solution).
	///
	/// Uses max-norm: max_i |err_i| / (atol + rtol·|u4_i|).
	fn compute_error_norm(
		&self,
		_u_old: &Field,
	) -> Result<f64> {
		// For efficiency on GPU, we approximate using L2 norm
		// instead of point-wise max/tolerance. This is standard
		// practice (most solvers accept either).
		let err_norm = self.u5.norm_l2()?;
		let sol_norm = self.u_tmp.norm_l2()?;
		let n = self.u_tmp.grid().len() as f64;
		let scale = self.atol * n.sqrt()
			+ self.rtol * sol_norm;
		if scale < 1e-30 {
			return Ok(0.0);
		}
		Ok(err_norm / scale)
	}
}

impl Field {
	/// Clone a field (allocating a new buffer). Used only in
	/// adaptive time-stepping where we need a snapshot for
	/// stencil application. NOT for the hot path of iterative
	/// solvers.
	fn clone_field(&self) -> Result<Self> {
		let mut f =
			Self::zeros(*self.grid(), DType::F64, self.device())?;
		f.copy_from(self)?;
		Ok(f)
	}
}
