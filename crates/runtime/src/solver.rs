use fluxion_core::{Boundaries, DType, Grid, Stencil};

use crate::device::Device;
use crate::error::Result;
use crate::field::Field;

/// Cached hipGraph for repeated time steps.
#[cfg(feature = "hip")]
#[derive(Debug)]
struct CachedHipGraph {
	graph: fluxion_backend_hip::HipGraph,
}

/// Time integration scheme.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeScheme {
	/// Forward Euler (1st order). CFL safety 0.4.
	Euler,
	/// Heun's method (2nd order). CFL safety 0.8.
	Rk2,
	/// Classical Runge-Kutta (4th order). CFL safety 1.4.
	Rk4,
}

/// Explicit diffusion stepper.
///
/// Solves `∂u/∂t = α·Δu + source`. All scratch buffers
/// are pre-allocated — **zero allocation** in the hot path.
#[derive(Debug)]
pub struct DiffusionSolver {
	stencil: Stencil,
	boundaries: Boundaries,
	scheme: TimeScheme,
	// scratch[0] always available (k1 / lap).
	// scratch[1..] used by RK stages.
	// Last element is always u_save for RK methods.
	scratch: Vec<Field>,
	dt: f64,
	alpha: f64,
	steps_done: usize,
	/// Cached hipGraph for repeated no-source steps.
	#[cfg(feature = "hip")]
	hip_graph: Option<CachedHipGraph>,
	/// Enable hipGraph acceleration for `step_n`.
	/// Off by default — call `enable_hip_graph()` to activate.
	#[cfg(feature = "hip")]
	use_hip_graph: bool,
}

impl DiffusionSolver {
	/// Euler solver with zero-Dirichlet BCs.
	///
	/// # Errors
	/// Returns an error if allocation fails.
	pub fn new(
		grid: Grid,
		alpha: f64,
		dt: Option<f64>,
		device: Device,
	) -> Result<Self> {
		Self::build(
			grid,
			alpha,
			dt,
			Boundaries::zero_dirichlet(),
			TimeScheme::Euler,
			device,
		)
	}

	/// Euler solver with custom BCs.
	///
	/// # Errors
	/// Returns an error if allocation fails.
	pub fn with_boundaries(
		grid: Grid,
		alpha: f64,
		dt: Option<f64>,
		boundaries: Boundaries,
		device: Device,
	) -> Result<Self> {
		Self::build(
			grid,
			alpha,
			dt,
			boundaries,
			TimeScheme::Euler,
			device,
		)
	}

	/// Full constructor.
	///
	/// # Errors
	/// Returns an error if allocation fails.
	pub fn build(
		grid: Grid,
		alpha: f64,
		dt: Option<f64>,
		boundaries: Boundaries,
		scheme: TimeScheme,
		device: Device,
	) -> Result<Self> {
		let h_min = grid.dx.min(grid.dy);
		let safety = match scheme {
			TimeScheme::Euler => 0.4,
			TimeScheme::Rk2 => 0.8,
			TimeScheme::Rk4 => 1.4,
		};
		let dt = dt.unwrap_or_else(|| {
			safety * h_min * h_min / (4.0 * alpha)
		});
		let stencil =
			Stencil::laplacian_2d_5pt(grid.dx, grid.dy);

		// Scratch buffers:
		// Euler: 1 (lap)
		// RK2: 2 (lap, u_save)
		// RK4: 5 (k1, k2, k3, k4, u_save)
		let n_scratch = match scheme {
			TimeScheme::Euler => 1,
			TimeScheme::Rk2 => 2,
			TimeScheme::Rk4 => 5,
		};
		let mut scratch = Vec::with_capacity(n_scratch);
		for _ in 0..n_scratch {
			scratch.push(Field::zeros(
				grid,
				DType::F64,
				device,
			)?);
		}

		Ok(Self {
			stencil,
			boundaries,
			scheme,
			scratch,
			dt,
			alpha,
			steps_done: 0,
			#[cfg(feature = "hip")]
			hip_graph: None,
			#[cfg(feature = "hip")]
			use_hip_graph: false,
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

	#[inline]
	#[must_use]
	pub const fn scheme(&self) -> TimeScheme {
		self.scheme
	}

	/// Enable hipGraph acceleration for `step_n`.
	///
	/// When enabled, the first call to `step_n` captures one
	/// time step into a hipGraph, then replays it for all
	/// subsequent steps. This eliminates kernel launch overhead
	/// and gives 1.5-2x speedup on small grids.
	///
	/// Only effective on HIP devices. No-op on CPU/wgpu.
	#[cfg(feature = "hip")]
	pub fn enable_hip_graph(&mut self) {
		self.use_hip_graph = true;
	}

	/// Advance one time step.
	///
	/// Automatically uses GPU command batching on wgpu
	/// devices (single `queue.submit` per step).
	///
	/// # Errors
	/// Returns an error on backend failure.
	#[inline]
	pub fn step(
		&mut self,
		u: &mut Field,
		source: Option<&Field>,
	) -> Result<()> {
		// GPU-batched path: encode all ops in one submit.
		#[cfg(feature = "wgpu")]
		if matches!(u.device(), Device::Wgpu { .. }) {
			return match self.scheme {
				TimeScheme::Euler => {
					self.step_euler_batched(u, source)
				}
				TimeScheme::Rk2 => {
					self.step_rk2(u, source)
				}
				TimeScheme::Rk4 => {
					self.step_rk4_batched(u, source)
				}
			};
		}

		match self.scheme {
			TimeScheme::Euler => {
				self.step_euler(u, source)
			}
			TimeScheme::Rk2 => self.step_rk2(u, source),
			TimeScheme::Rk4 => self.step_rk4(u, source),
		}
	}

	/// Advance `n` time steps (no source term).
	///
	/// On HIP devices, uses hipGraph to capture the step once
	/// and replay it N times with minimal launch overhead.
	///
	/// # Errors
	/// Returns an error on backend failure.
	pub fn step_n(
		&mut self,
		u: &mut Field,
		n: usize,
	) -> Result<()> {
		if n == 0 {
			return Ok(());
		}

		// hipGraph-accelerated path for HIP devices.
		#[cfg(feature = "hip")]
		if self.use_hip_graph
			&& matches!(u.device(), Device::Hip { .. })
		{
			return self.step_n_hip_graph(u, n);
		}

		for _ in 0..n {
			self.step(u, None)?;
		}
		Ok(())
	}

	/// hipGraph-accelerated step_n for HIP devices.
	///
	/// First call captures one step into a graph, then
	/// replays it N times. Subsequent calls reuse the graph.
	#[cfg(feature = "hip")]
	fn step_n_hip_graph(
		&mut self,
		u: &mut Field,
		n: usize,
	) -> Result<()> {
		use fluxion_backend_hip::HipGraph;

		// Capture graph on first call.
		if self.hip_graph.is_none() {
			let backend = crate::storage::hip_backend();
			let stream = backend.stream();

			HipGraph::begin_capture(stream)
				.map_err(fluxion_core::CoreError::BackendError)?;

			// Record one step (not executed, just captured).
			match self.scheme {
				TimeScheme::Euler => {
					self.step_euler(u, None)?;
					// Undo the steps_done increment from
					// step_euler — it wasn't really executed.
					self.steps_done -= 1;
				}
				TimeScheme::Rk2 => {
					self.step_rk2(u, None)?;
					self.steps_done -= 1;
				}
				TimeScheme::Rk4 => {
					self.step_rk4(u, None)?;
					self.steps_done -= 1;
				}
			}

			let graph = HipGraph::end_capture(stream)
				.map_err(fluxion_core::CoreError::BackendError)?;

			self.hip_graph =
				Some(CachedHipGraph { graph });
		}

		let cached = self.hip_graph.as_ref().unwrap();
		let backend = crate::storage::hip_backend();
		let stream = backend.stream();

		// Replay the graph n times.
		// No sync needed — the graph launches are stream-ordered.
		// Any subsequent D2H readback (max, to_vec, norm_l2)
		// syncs implicitly via hipMemcpy or sync_stream.
		for _ in 0..n {
			cached
				.graph
				.launch(stream)
				.map_err(fluxion_core::CoreError::BackendError)?;
			self.steps_done += 1;
		}

		Ok(())
	}

	/// Advance `n` time steps with a constant source.
	///
	/// # Errors
	/// Returns an error on backend failure.
	pub fn step_n_with_source(
		&mut self,
		u: &mut Field,
		n: usize,
		source: &Field,
	) -> Result<()> {
		for _ in 0..n {
			self.step(u, Some(source))?;
		}
		Ok(())
	}

	// ── Euler ────────────────────────────────────────

	/// `u += dt·α·Δ(u) + dt·source`
	///
	/// Uses fused `stencil_axpy`: 1 dispatch instead of 2.
	fn step_euler(
		&mut self,
		u: &mut Field,
		source: Option<&Field>,
	) -> Result<()> {
		let coeff = self.alpha * self.dt;
		// Copy u → scratch so stencil reads from scratch
		// while axpy writes to u. Single fused dispatch.
		self.scratch[0].copy_from(u)?;
		u.stencil_axpy(
			coeff,
			&self.scratch[0],
			&self.stencil,
			&self.boundaries,
		)?;
		if let Some(src) = source {
			u.axpy(self.dt, src)?;
		}
		self.steps_done += 1;
		Ok(())
	}

	// ── RK2 (Heun) ──────────────────────────────────

	/// Heun's method (2nd order, Shu-Osher form).
	///
	/// Scratch layout: `scratch[0]` = lap, `scratch[1]` = `u_old`.
	fn step_rk2(
		&mut self,
		u: &mut Field,
		source: Option<&Field>,
	) -> Result<()> {
		let dt = self.dt;
		let coeff = dt * self.alpha;

		// Save u_old.
		self.scratch[1].copy_from(u)?;

		// u* = u + dt·α·Δ(u) (fused).
		self.scratch[0].copy_from(u)?;
		u.stencil_axpy(
			coeff,
			&self.scratch[0],
			&self.stencil,
			&self.boundaries,
		)?;
		if let Some(src) = source {
			u.axpy(dt, src)?;
		}

		// u* += dt·α·Δ(u*) (fused).
		self.scratch[0].copy_from(u)?;
		u.stencil_axpy(
			coeff,
			&self.scratch[0],
			&self.stencil,
			&self.boundaries,
		)?;
		if let Some(src) = source {
			u.axpy(dt, src)?;
		}

		// u = 0.5·u_old + 0.5·u*.
		u.axpy(1.0, &self.scratch[1])?;
		u.scale(0.5)?;

		self.steps_done += 1;
		Ok(())
	}

	// ── RK4 ──────────────────────────────────────────

	/// Classical RK4, using fused `stencil_axpy` to build
	/// intermediate stages (saves 3 separate axpy dispatches
	/// for stage setup).
	///
	/// scratch layout: `[k1, k2, k3, k4, u_old]`
	fn step_rk4(
		&mut self,
		u: &mut Field,
		source: Option<&Field>,
	) -> Result<()> {
		let dt = self.dt;
		let alpha = self.alpha;

		// Save u_old in scratch[4].
		self.scratch[4].copy_from(u)?;

		// k1 = Δ(u_old), into scratch[0].
		u.apply_stencil_into(
			&self.stencil,
			&self.boundaries,
			&mut self.scratch[0],
		)?;

		// u = u_old + 0.5·dt·α·k1  (fused: copy + stencil_axpy).
		u.copy_from(&self.scratch[4])?;
		u.stencil_axpy(
			0.5 * dt * alpha,
			&self.scratch[4],
			&self.stencil,
			&self.boundaries,
		)?;

		// k2 = Δ(u), into scratch[1].
		u.apply_stencil_into(
			&self.stencil,
			&self.boundaries,
			&mut self.scratch[1],
		)?;

		// u = u_old + 0.5·dt·α·k2.
		u.copy_from(&self.scratch[4])?;
		u.axpy(0.5 * dt * alpha, &self.scratch[1])?;
		u.zero_boundaries()?;

		// k3 = Δ(u), into scratch[2].
		u.apply_stencil_into(
			&self.stencil,
			&self.boundaries,
			&mut self.scratch[2],
		)?;

		// u = u_old + dt·α·k3.
		u.copy_from(&self.scratch[4])?;
		u.axpy(dt * alpha, &self.scratch[2])?;
		u.zero_boundaries()?;

		// k4 = Δ(u), into scratch[3].
		u.apply_stencil_into(
			&self.stencil,
			&self.boundaries,
			&mut self.scratch[3],
		)?;

		// u = u_old + dt·α/6·(k1 + 2k2 + 2k3 + k4) + dt·source.
		u.copy_from(&self.scratch[4])?;
		let c = dt * alpha / 6.0;
		u.axpy(c, &self.scratch[0])?;
		u.axpy(2.0 * c, &self.scratch[1])?;
		u.axpy(2.0 * c, &self.scratch[2])?;
		u.axpy(c, &self.scratch[3])?;
		if let Some(src) = source {
			u.axpy(dt, src)?;
		}

		self.steps_done += 1;
		Ok(())
	}
	// ── GPU-batched solvers (wgpu only) ─────────────

	/// Euler step with a single GPU submit.
	#[cfg(feature = "wgpu")]
	fn step_euler_batched(
		&mut self,
		u: &mut Field,
		source: Option<&Field>,
	) -> Result<()> {
		use crate::storage::BufferStorage;

		let coeff = self.alpha * self.dt;
		let backend = crate::storage::wgpu_backend();
		let grid = *u.grid();

		// Extract wgpu buffers.
		let (
			BufferStorage::Wgpu(u_buf),
			BufferStorage::Wgpu(s0_buf),
		) = (
			&mut u.storage,
			&mut self.scratch[0].storage,
		)
		else {
			return self.step_euler(u, source);
		};

		let mut enc = backend.begin_batch();

		// copy u → scratch[0]
		backend.encode_copy(&mut enc, u_buf, s0_buf);
		// u += coeff * Δ(scratch[0])
		backend.encode_stencil_axpy(
			&mut enc, coeff, s0_buf, u_buf, &grid,
		);

		// source term
		if let Some(src) = source
			&& let BufferStorage::Wgpu(src_buf) =
				&src.storage
		{
			backend.encode_axpy(
				&mut enc, self.dt, src_buf, u_buf,
			);
		}

		backend.submit_batch(enc);
		self.steps_done += 1;
		Ok(())
	}

	/// RK4 step with a single GPU submit.
	#[cfg(feature = "wgpu")]
	#[allow(clippy::too_many_lines)]
	fn step_rk4_batched(
		&mut self,
		u: &mut Field,
		source: Option<&Field>,
	) -> Result<()> {
		use crate::storage::BufferStorage;

		let dt = self.dt;
		let alpha = self.alpha;
		let backend = crate::storage::wgpu_backend();
		let grid = *u.grid();

		// We need mutable access to u and all 5 scratch
		// buffers simultaneously. Extract raw wgpu buffers.
		let [s0, s1, s2, s3, s4] =
			self.scratch.as_mut_slice()
		else {
			return self.step_rk4(u, source);
		};

		let (
			BufferStorage::Wgpu(u_buf),
			BufferStorage::Wgpu(s0_buf),
			BufferStorage::Wgpu(s1_buf),
			BufferStorage::Wgpu(s2_buf),
			BufferStorage::Wgpu(s3_buf),
			BufferStorage::Wgpu(s4_buf),
		) = (
			&mut u.storage,
			&mut s0.storage,
			&mut s1.storage,
			&mut s2.storage,
			&mut s3.storage,
			&mut s4.storage,
		)
		else {
			return self.step_rk4(u, source);
		};

		let mut enc = backend.begin_batch();

		// Save u_old in scratch[4].
		backend.encode_copy(&mut enc, u_buf, s4_buf);

		// k1 = Δ(u), into scratch[0].
		backend.encode_stencil(
			&mut enc, u_buf, s0_buf, &grid,
		);

		// u = u_old + 0.5·dt·α·Δ(u_old)
		// (u is still u_old at this point, and s4 = u_old)
		backend.encode_stencil_axpy(
			&mut enc,
			0.5 * dt * alpha,
			s4_buf,
			u_buf,
			&grid,
		);

		// k2 = Δ(u), into scratch[1].
		backend.encode_stencil(
			&mut enc, u_buf, s1_buf, &grid,
		);

		// u = u_old + 0.5·dt·α·k2.
		backend.encode_copy(&mut enc, s4_buf, u_buf);
		backend.encode_axpy(
			&mut enc,
			0.5 * dt * alpha,
			s1_buf,
			u_buf,
		);

		// k3 = Δ(u), into scratch[2].
		backend.encode_stencil(
			&mut enc, u_buf, s2_buf, &grid,
		);

		// u = u_old + dt·α·k3.
		backend.encode_copy(&mut enc, s4_buf, u_buf);
		backend.encode_axpy(
			&mut enc, dt * alpha, s2_buf, u_buf,
		);

		// k4 = Δ(u), into scratch[3].
		backend.encode_stencil(
			&mut enc, u_buf, s3_buf, &grid,
		);

		// u = u_old + dt·α/6·(k1 + 2k2 + 2k3 + k4).
		backend.encode_copy(&mut enc, s4_buf, u_buf);
		let c = dt * alpha / 6.0;
		backend.encode_axpy(&mut enc, c, s0_buf, u_buf);
		backend.encode_axpy(
			&mut enc,
			2.0 * c,
			s1_buf,
			u_buf,
		);
		backend.encode_axpy(
			&mut enc,
			2.0 * c,
			s2_buf,
			u_buf,
		);
		backend.encode_axpy(&mut enc, c, s3_buf, u_buf);

		if let Some(src) = source
			&& let BufferStorage::Wgpu(src_buf) =
				&src.storage
		{
			backend.encode_axpy(
				&mut enc, dt, src_buf, u_buf,
			);
		}

		backend.submit_batch(enc);
		self.steps_done += 1;
		Ok(())
	}
}

#[cfg(test)]
mod tests {
	use fluxion_core::Grid;

	use super::*;

	#[test]
	fn diffusion_decreases_peak() {
		let n = 64;
		let h = 1.0 / (n - 1) as f64;
		let grid = Grid::square(n, h).unwrap();
		let device = Device::Cpu;
		let alpha = 0.01;
		let sigma = 0.1_f64;

		let mut data = vec![0.0_f64; n * n];
		for row in 0..n {
			for col in 0..n {
				let x = col as f64 * h;
				let y = row as f64 * h;
				let dx = x - 0.5;
				let dy = y - 0.5;
				let r2 = dx.mul_add(dx, dy * dy);
				data[row * n + col] =
					(-r2 / (2.0 * sigma * sigma)).exp();
			}
		}

		let initial_peak: f64 = data
			.iter()
			.copied()
			.reduce(f64::max)
			.unwrap();

		let mut u =
			Field::from_f64(grid, &data, device).unwrap();
		let mut solver =
			DiffusionSolver::new(grid, alpha, None, device)
				.unwrap();

		solver.step_n(&mut u, 100).unwrap();
		let final_peak = u.max().unwrap();

		assert!(final_peak < initial_peak);
		assert!(final_peak > 0.0);
	}

	#[test]
	fn step_n_is_incremental() {
		let n = 32;
		let h = 1.0 / (n - 1) as f64;
		let grid = Grid::square(n, h).unwrap();
		let device = Device::Cpu;

		let mut u = Field::from_f64(
			grid,
			&vec![1.0; n * n],
			device,
		)
		.unwrap();
		let mut solver =
			DiffusionSolver::new(grid, 0.01, None, device)
				.unwrap();

		solver.step_n(&mut u, 10).unwrap();
		assert_eq!(solver.steps_done(), 10);
		solver.step_n(&mut u, 5).unwrap();
		assert_eq!(solver.steps_done(), 15);
	}

	#[test]
	fn source_term_heats_field() {
		let n = 32;
		let h = 1.0 / (n - 1) as f64;
		let grid = Grid::square(n, h).unwrap();
		let device = Device::Cpu;

		let mut u =
			Field::zeros(grid, DType::F64, device).unwrap();
		let source = Field::from_f64(
			grid,
			&vec![100.0; n * n],
			device,
		)
		.unwrap();

		let mut solver =
			DiffusionSolver::new(grid, 0.01, None, device)
				.unwrap();
		solver
			.step_n_with_source(&mut u, 50, &source)
			.unwrap();

		let center = u.max().unwrap();
		assert!(center > 0.0);
	}

	#[test]
	fn rk4_stable_and_finite() {
		let n = 32;
		let h = 1.0 / (n - 1) as f64;
		let grid = Grid::square(n, h).unwrap();
		let device = Device::Cpu;
		let pi = std::f64::consts::PI;

		let mut data = vec![0.0_f64; n * n];
		for row in 0..n {
			for col in 0..n {
				let x = col as f64 * h;
				let y = row as f64 * h;
				data[row * n + col] =
					(pi * x).sin() * (pi * y).sin();
			}
		}

		let mut u =
			Field::from_f64(grid, &data, device).unwrap();
		let mut solver = DiffusionSolver::build(
			grid,
			0.01,
			None,
			Boundaries::zero_dirichlet(),
			TimeScheme::Rk4,
			device,
		)
		.unwrap();

		solver.step_n(&mut u, 200).unwrap();
		let norm = u.norm_l2().unwrap();
		assert!(norm.is_finite());
		assert!(norm > 0.0);
	}
}
