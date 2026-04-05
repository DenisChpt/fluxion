use std::ffi::{c_void, CString};
use std::ptr;

use fluxion_core::{
	Backend, BackendBuffer, Boundaries, CoreError, DType, Grid,
	Result, Stencil,
};

use crate::buffer::HipBuffer;
use crate::error::HipError;
use crate::ffi;

/// Pre-loaded HIP kernel functions.
struct Kernels {
	stencil_2d: ffi::hipFunction_t,
	stencil_9pt: ffi::hipFunction_t,
	fused_stencil_axpy: ffi::hipFunction_t,
	fill: ffi::hipFunction_t,
	scale: ffi::hipFunction_t,
	axpy: ffi::hipFunction_t,
	dot2: ffi::hipFunction_t,
	reduce_sum_sq: ffi::hipFunction_t,
	reduce_sum: ffi::hipFunction_t,
	reduce_max: ffi::hipFunction_t,
	reduce_min: ffi::hipFunction_t,
	dot: ffi::hipFunction_t,
	restrict: ffi::hipFunction_t,
	prolong: ffi::hipFunction_t,
	weighted_jacobi: ffi::hipFunction_t,
	// Extended vector operations.
	pointwise_mult: ffi::hipFunction_t,
	pointwise_div: ffi::hipFunction_t,
	waxpy: ffi::hipFunction_t,
	aypx: ffi::hipFunction_t,
	reciprocal: ffi::hipFunction_t,
	abs_val: ffi::hipFunction_t,
	// Pipelined CG fused kernel.
	pipelined_cg_fused: ffi::hipFunction_t,
	pointwise_max: ffi::hipFunction_t,
	pointwise_min: ffi::hipFunction_t,
	// Boundary enforcement.
	zero_boundaries: ffi::hipFunction_t,
	// Advanced kernels.
	conv_diff_axpy: ffi::hipFunction_t,
	conv_diff: ffi::hipFunction_t,
	stencil_naive: ffi::hipFunction_t,
}

impl std::fmt::Debug for Kernels {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.debug_struct("Kernels").finish_non_exhaustive()
	}
}

/// HIP/ROCm backend for AMD GPUs.
///
/// Loads pre-compiled kernel code objects and dispatches via
/// `hipModuleLaunchKernel`. Native f64 (CDNA has full f64
/// throughput). All kernels use LDS tiling for stencil ops.
#[derive(Debug)]
pub struct HipBackend {
	_module_stencil: ffi::hipModule_t,
	_module_fused: ffi::hipModule_t,
	_module_elem: ffi::hipModule_t,
	_module_reduce: ffi::hipModule_t,
	_module_mg: ffi::hipModule_t,
	_module_cd: ffi::hipModule_t,
	stream: ffi::hipStream_t,
	kernels: Kernels,
	/// Detected GPU architecture (RDNA vs CDNA).
	pub arch: ffi::GpuArch,
	/// Number of blocks for reductions (scaled to CU count).
	reduce_blocks: u32,
	/// Pre-allocated reduction scratch (grow-only).
	reduce_buf: std::sync::RwLock<ReduceScratch>,
	/// True if async operations are pending on the stream
	/// since the last sync. Avoids redundant sync_stream().
	stream_dirty: std::sync::atomic::AtomicBool,
}

// SAFETY: HIP runtime is thread-safe when using streams.
unsafe impl Send for HipBackend {}
unsafe impl Sync for HipBackend {}

#[derive(Debug)]
struct ReduceScratch {
	ptr: ffi::hipDeviceptr_t,
	capacity: usize, // in elements (f64)
}

// SAFETY: Device pointer, thread-safe via stream ordering.
unsafe impl Send for ReduceScratch {}
unsafe impl Sync for ReduceScratch {}

impl HipBackend {
	/// Initialize the HIP backend.
	///
	/// Loads kernel code objects from embedded byte arrays
	/// (compiled by build.rs). Creates a compute stream.
	///
	/// # Errors
	/// Returns an error if no HIP device is available.
	pub fn new() -> std::result::Result<Self, HipError> {
		let mut count = 0i32;
		ffi::check(unsafe {
			ffi::hipGetDeviceCount(&mut count)
		})
		.map_err(HipError::Runtime)?;
		if count == 0 {
			return Err(HipError::NoDevice);
		}
		ffi::check(unsafe { ffi::hipSetDevice(0) })
			.map_err(HipError::Runtime)?;

		let mut stream = ptr::null_mut();
		ffi::check(unsafe {
			ffi::hipStreamCreate(&mut stream)
		})
		.map_err(HipError::Runtime)?;

		// Load kernel modules from compiled code objects.
		let m_stencil = load_module(include_bytes!(
			concat!(env!("OUT_DIR"), "/stencil_2d.co")
		))?;
		let m_fused = load_module(include_bytes!(concat!(
			env!("OUT_DIR"),
			"/fused_stencil_axpy.co"
		)))?;
		let m_elem = load_module(include_bytes!(concat!(
			env!("OUT_DIR"),
			"/elementwise.co"
		)))?;
		let m_reduce = load_module(include_bytes!(
			concat!(env!("OUT_DIR"), "/reduce.co")
		))?;
		let m_mg = load_module(include_bytes!(concat!(
			env!("OUT_DIR"),
			"/multigrid.co"
		)))?;
		let m_cd = load_module(include_bytes!(concat!(
			env!("OUT_DIR"),
			"/convection_diffusion.co"
		)))?;

		let kernels = Kernels {
			stencil_2d: get_fn(m_stencil, "stencil_2d_f64")?,
			stencil_9pt: get_fn(
				m_stencil,
				"stencil_9pt_f64",
			)?,
			fused_stencil_axpy: get_fn(
				m_fused,
				"fused_stencil_axpy_f64",
			)?,
			fill: get_fn(m_elem, "fill_f64")?,
			scale: get_fn(m_elem, "scale_f64")?,
			axpy: get_fn(m_elem, "axpy_f64")?,
			dot2: get_fn(m_reduce, "dot2_f64")?,
			reduce_sum_sq: get_fn(
				m_reduce,
				"reduce_sum_sq_f64",
			)?,
			reduce_sum: get_fn(m_reduce, "reduce_sum_f64")?,
			reduce_max: get_fn(m_reduce, "reduce_max_f64")?,
			reduce_min: get_fn(m_reduce, "reduce_min_f64")?,
			dot: get_fn(m_reduce, "dot_f64")?,
			restrict: get_fn(m_mg, "restrict_f64")?,
			prolong: get_fn(m_mg, "prolong_f64")?,
			weighted_jacobi: get_fn(
				m_mg,
				"weighted_jacobi_f64",
			)?,
			pointwise_mult: get_fn(
				m_elem,
				"pointwise_mult_f64",
			)?,
			pointwise_div: get_fn(
				m_elem,
				"pointwise_div_f64",
			)?,
			waxpy: get_fn(m_elem, "waxpy_f64")?,
			aypx: get_fn(m_elem, "aypx_f64")?,
			reciprocal: get_fn(
				m_elem,
				"reciprocal_f64",
			)?,
			abs_val: get_fn(m_elem, "abs_f64")?,
			pointwise_max: get_fn(
				m_elem,
				"pointwise_max_f64",
			)?,
			pointwise_min: get_fn(
				m_elem,
				"pointwise_min_f64",
			)?,
			zero_boundaries: get_fn(
				m_elem,
				"zero_boundaries_f64",
			)?,
			pipelined_cg_fused: get_fn(
				m_elem,
				"pipelined_cg_fused_f64",
			)?,
			conv_diff_axpy: get_fn(
				m_cd,
				"conv_diff_axpy_f64",
			)?,
			conv_diff: get_fn(m_cd, "conv_diff_f64")?,
			stencil_naive: get_fn(
				m_cd,
				"stencil_naive_f64",
			)?,
		};

		// Initial reduction scratch: 256 elements.
		let init_cap = 256usize;
		let mut rptr = ptr::null_mut();
		ffi::check(unsafe {
			ffi::hipMalloc(&mut rptr, init_cap * 8)
		})
		.map_err(HipError::Runtime)?;

		let arch = ffi::detect_gpu_arch();

		// Scale reduction blocks to GPU size. 4 blocks/CU
		// for good occupancy, clamped to [32, 128].
		let mut num_cus: std::ffi::c_int = 0;
		let _ = unsafe {
			ffi::hipDeviceGetAttribute(
				&mut num_cus,
				ffi::HIP_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
				0,
			)
		};
		let reduce_blocks = if num_cus > 0 {
			((num_cus as u32) * 4).clamp(32, 128)
		} else {
			32
		};
		tracing::info!(
			"HIP GPU arch: {arch:?}, CUs: {num_cus}, \
			 reduce_blocks: {reduce_blocks}"
		);

		Ok(Self {
			_module_stencil: m_stencil,
			_module_fused: m_fused,
			_module_elem: m_elem,
			_module_reduce: m_reduce,
			_module_mg: m_mg,
			_module_cd: m_cd,
			stream,
			kernels,
			arch,
			reduce_blocks,
			reduce_buf: std::sync::RwLock::new(ReduceScratch {
				ptr: rptr,
				capacity: init_cap,
			}),
			stream_dirty: std::sync::atomic::AtomicBool::new(
				false,
			),
		})
	}

	/// Get the compute stream for graph capture.
	#[inline]
	#[must_use]
	pub fn stream(&self) -> ffi::hipStream_t {
		self.stream
	}

	/// Synchronize the compute stream (blocks until all
	/// queued operations complete).
	#[inline]
	pub fn sync_stream(&self) {
		unsafe {
			ffi::hipStreamSynchronize(self.stream);
		}
	}

	/// Launch a kernel with the given grid/block dims.
	/// Marks the stream as dirty (pending async ops).
	#[inline]
	fn launch(
		&self,
		func: ffi::hipFunction_t,
		grid: (u32, u32, u32),
		block: (u32, u32, u32),
		shared_mem: u32,
		params: &mut [*mut c_void],
	) {
		unsafe {
			ffi::hipModuleLaunchKernel(
				func,
				grid.0,
				grid.1,
				grid.2,
				block.0,
				block.1,
				block.2,
				shared_mem,
				self.stream,
				params.as_mut_ptr(),
				ptr::null_mut(),
			);
		}
		self.stream_dirty.store(
			true,
			std::sync::atomic::Ordering::Release,
		);
	}

	/// Sync the stream only if there are pending operations.
	/// Skips the expensive `hipStreamSynchronize` when no
	/// async work has been queued since the last sync.
	#[inline]
	#[allow(dead_code)]
	fn sync_if_dirty(&self) {
		if self
			.stream_dirty
			.swap(false, std::sync::atomic::Ordering::AcqRel)
		{
			self.sync_stream();
		}
	}

	/// Ensure reduction scratch can hold `n` f64 elements.
	fn ensure_reduce(&self, n: usize) {
		let mut s =
			self.reduce_buf.write().expect("reduce lock");
		if n <= s.capacity {
			return;
		}
		let new_cap = n.next_power_of_two();
		if !s.ptr.is_null() {
			unsafe { ffi::hipFree(s.ptr) };
		}
		let mut ptr = ptr::null_mut();
		unsafe {
			ffi::hipMalloc(&mut ptr, new_cap * 8);
		}
		s.ptr = ptr;
		s.capacity = new_cap;
	}

	/// Fixed number of blocks for reductions.
	/// Grid-stride loop in the kernel handles any array size.
	/// 32 blocks × 256 threads = 8192 threads, enough to
	/// saturate RX 7800XT (30 CUs) and MI250X (110 CUs).
	/// Only 32 partials to sum on host (essentially free).
	/// Max reduction blocks (used for stack-allocated partials).
	const MAX_REDUCE_BLOCKS: usize = 128;

	/// Run a reduction kernel returning a single f64.
	fn run_reduction(
		&self,
		func: ffi::hipFunction_t,
		buf: &HipBuffer,
	) -> Result<f64> {
		let len = buf.len() as i32;
		let n_blocks = self.reduce_blocks;

		self.ensure_reduce(n_blocks as usize);
		let scratch =
			self.reduce_buf.read().expect("reduce lock");

		let mut params: [*mut c_void; 3] = [
			(&buf.ptr as *const _ as *mut c_void),
			(&scratch.ptr as *const _ as *mut c_void),
			(&len as *const _ as *mut c_void),
		];

		self.launch(
			func,
			(n_blocks, 1, 1),
			(256, 1, 1),
			0,
			&mut params,
		);

		// hipMemcpy D2H is synchronous: it implicitly waits
		// for all preceding stream operations (including the
		// reduce kernel we just launched). No need for an
		// explicit sync_if_dirty() — that was a redundant
		// stream synchronization.
		let mut partials =
			[0.0_f64; Self::MAX_REDUCE_BLOCKS];
		unsafe {
			ffi::hipMemcpy(
				partials.as_mut_ptr().cast(),
				scratch.ptr,
				n_blocks as usize * 8,
				ffi::hipMemcpyDeviceToHost,
			);
		}
		// Stream is now synchronized — clear dirty flag.
		self.stream_dirty.store(
			false,
			std::sync::atomic::Ordering::Release,
		);

		// Kahan sum on host (n_blocks values = negligible).
		let mut sum = 0.0_f64;
		let mut comp = 0.0_f64;
		for &v in &partials[..n_blocks as usize] {
			let y = v - comp;
			let t = sum + y;
			comp = (t - sum) - y;
			sum = t;
		}
		Ok(sum)
	}
}

impl Drop for HipBackend {
	fn drop(&mut self) {
		unsafe {
			let s =
				self.reduce_buf.read().expect("reduce lock");
			if !s.ptr.is_null() {
				ffi::hipFree(s.ptr);
			}
			ffi::hipStreamDestroy(self.stream);
		}
	}
}

fn load_module(
	data: &[u8],
) -> std::result::Result<ffi::hipModule_t, HipError> {
	let mut module = ptr::null_mut();
	ffi::check(unsafe {
		ffi::hipModuleLoadData(
			&mut module,
			data.as_ptr().cast(),
		)
	})
	.map_err(HipError::ModuleLoad)?;
	Ok(module)
}

fn get_fn(
	module: ffi::hipModule_t,
	name: &str,
) -> std::result::Result<ffi::hipFunction_t, HipError> {
	let cname = CString::new(name).unwrap();
	let mut func = ptr::null_mut();
	ffi::check(unsafe {
		ffi::hipModuleGetFunction(
			&mut func,
			module,
			cname.as_ptr(),
		)
	})
	.map_err(|_| HipError::KernelNotFound(name.into()))?;
	Ok(func)
}


// ── Advanced kernels (not in Backend trait) ─────────

impl HipBackend {
	/// Pipelined CG fused 6-way update.
	///
	/// Single kernel: z,t,p,x,r,w all updated in one pass.
	/// Replaces 5-6 separate kernel launches.
	#[allow(clippy::too_many_arguments)]
	pub fn pipelined_cg_fused(
		&self,
		alpha: f64,
		beta: f64,
		z: &mut HipBuffer,
		t: &mut HipBuffer,
		p: &mut HipBuffer,
		x: &mut HipBuffer,
		r: &mut HipBuffer,
		w: &mut HipBuffer,
		q: &HipBuffer,
	) {
		let mut a = alpha;
		let mut b = beta;
		let mut len = z.len() as i32;
		let n = z.len() as u32;
		let mut params: [*mut c_void; 10] = [
			(&mut a as *mut _ as *mut c_void),
			(&mut b as *mut _ as *mut c_void),
			(&z.ptr as *const _ as *mut c_void),
			(&t.ptr as *const _ as *mut c_void),
			(&p.ptr as *const _ as *mut c_void),
			(&x.ptr as *const _ as *mut c_void),
			(&r.ptr as *const _ as *mut c_void),
			(&w.ptr as *const _ as *mut c_void),
			(&q.ptr as *const _ as *mut c_void),
			(&mut len as *mut _ as *mut c_void),
		];
		self.launch(
			self.kernels.pipelined_cg_fused,
			(n.div_ceil(256), 1, 1),
			(256, 1, 1),
			0,
			&mut params,
		);
	}

	/// Zero boundary rows/cols for Dirichlet BCs.
	///
	/// Single kernel launch, ~4K threads for a 1024x1024 grid.
	/// Replaces the catastrophic readback path in Field.
	pub fn zero_boundaries(
		&self,
		buf: &mut HipBuffer,
		rows: usize,
		cols: usize,
	) {
		let n = (rows * cols) as u32;
		let mut r = rows as i32;
		let mut c = cols as i32;
		let mut params: [*mut c_void; 3] = [
			(&buf.ptr as *const _ as *mut c_void),
			(&mut r as *mut _ as *mut c_void),
			(&mut c as *mut _ as *mut c_void),
		];
		self.launch(
			self.kernels.zero_boundaries,
			(n.div_ceil(256), 1, 1),
			(256, 1, 1),
			0,
			&mut params,
		);
	}

	/// Naive stencil (no LDS tiling) for benchmark comparison.
	pub fn stencil_naive(
		&self,
		input: &HipBuffer,
		output: &mut HipBuffer,
		rows: i32,
		cols: i32,
		inv_dx2: f64,
		inv_dy2: f64,
	) {
		let mut r = rows;
		let mut c = cols;
		let mut dx2 = inv_dx2;
		let mut dy2 = inv_dy2;
		let mut params: [*mut c_void; 6] = [
			(&input.ptr as *const _ as *mut c_void),
			(&output.ptr as *const _ as *mut c_void),
			(&mut r as *mut _ as *mut c_void),
			(&mut c as *mut _ as *mut c_void),
			(&mut dx2 as *mut _ as *mut c_void),
			(&mut dy2 as *mut _ as *mut c_void),
		];
		self.launch(
			self.kernels.stencil_naive,
			(
				(cols as u32).div_ceil(16),
				(rows as u32).div_ceil(16),
				1,
			),
			(16, 16, 1),
			0,
			&mut params,
		);
	}

	/// Convection-diffusion operator (standalone).
	/// `output = kappa * Lap(u) + (vx,vy) · grad(u)`
	#[allow(clippy::too_many_arguments)]
	pub fn conv_diff(
		&self,
		u: &HipBuffer,
		kappa: &HipBuffer,
		vx: &HipBuffer,
		vy: &HipBuffer,
		output: &mut HipBuffer,
		rows: i32,
		cols: i32,
		inv_dx: f64,
		inv_dy: f64,
		inv_dx2: f64,
		inv_dy2: f64,
	) {
		let mut r = rows;
		let mut c = cols;
		let mut dx = inv_dx;
		let mut dy = inv_dy;
		let mut dx2 = inv_dx2;
		let mut dy2 = inv_dy2;
		let mut params: [*mut c_void; 11] = [
			(&u.ptr as *const _ as *mut c_void),
			(&kappa.ptr as *const _ as *mut c_void),
			(&vx.ptr as *const _ as *mut c_void),
			(&vy.ptr as *const _ as *mut c_void),
			(&output.ptr as *const _ as *mut c_void),
			(&mut r as *mut _ as *mut c_void),
			(&mut c as *mut _ as *mut c_void),
			(&mut dx as *mut _ as *mut c_void),
			(&mut dy as *mut _ as *mut c_void),
			(&mut dx2 as *mut _ as *mut c_void),
			(&mut dy2 as *mut _ as *mut c_void),
		];
		self.launch(
			self.kernels.conv_diff,
			(
				(cols as u32).div_ceil(16),
				(rows as u32).div_ceil(16),
				1,
			),
			(16, 16, 1),
			0,
			&mut params,
		);
	}

	/// Fused convection-diffusion + axpy.
	/// `output += alpha * (kappa * Lap(u) + v · grad(u))`
	#[allow(clippy::too_many_arguments)]
	pub fn conv_diff_axpy(
		&self,
		alpha: f64,
		u: &HipBuffer,
		kappa: &HipBuffer,
		vx: &HipBuffer,
		vy: &HipBuffer,
		output: &mut HipBuffer,
		rows: i32,
		cols: i32,
		inv_dx: f64,
		inv_dy: f64,
		inv_dx2: f64,
		inv_dy2: f64,
	) {
		let mut a = alpha;
		let mut r = rows;
		let mut c = cols;
		let mut dx = inv_dx;
		let mut dy = inv_dy;
		let mut dx2 = inv_dx2;
		let mut dy2 = inv_dy2;
		let mut params: [*mut c_void; 12] = [
			(&mut a as *mut _ as *mut c_void),
			(&u.ptr as *const _ as *mut c_void),
			(&kappa.ptr as *const _ as *mut c_void),
			(&vx.ptr as *const _ as *mut c_void),
			(&vy.ptr as *const _ as *mut c_void),
			(&output.ptr as *const _ as *mut c_void),
			(&mut r as *mut _ as *mut c_void),
			(&mut c as *mut _ as *mut c_void),
			(&mut dx as *mut _ as *mut c_void),
			(&mut dy as *mut _ as *mut c_void),
			(&mut dx2 as *mut _ as *mut c_void),
			(&mut dy2 as *mut _ as *mut c_void),
		];
		self.launch(
			self.kernels.conv_diff_axpy,
			(
				(cols as u32).div_ceil(16),
				(rows as u32).div_ceil(16),
				1,
			),
			(16, 16, 1),
			0,
			&mut params,
		);
	}
}

impl Backend for HipBackend {
	type Buffer = HipBuffer;

	#[allow(clippy::unnecessary_literal_bound)]
	fn name(&self) -> &str {
		"hip"
	}

	#[inline]
	fn allocate(
		&self,
		len: usize,
		dtype: DType,
	) -> Result<HipBuffer> {
		HipBuffer::zeros(len, dtype)
			.map_err(CoreError::BackendError)
	}

	#[inline]
	fn upload_f64(
		&self,
		data: &[f64],
	) -> Result<HipBuffer> {
		HipBuffer::from_f64(data)
			.map_err(CoreError::BackendError)
	}

	#[inline]
	fn upload_f32(
		&self,
		data: &[f32],
	) -> Result<HipBuffer> {
		HipBuffer::from_f32(data)
			.map_err(CoreError::BackendError)
	}

	#[inline]
	fn apply_stencil(
		&self,
		input: &HipBuffer,
		output: &mut HipBuffer,
		grid: &Grid,
		stencil: &Stencil,
		_boundaries: &Boundaries,
	) -> Result<()> {
		let mut rows = grid.rows as i32;
		let mut cols = grid.cols as i32;
		let grid_dim = (
			(cols as u32).div_ceil(16),
			(rows as u32).div_ceil(16),
			1,
		);
		let block_dim = (16_u32, 16, 1);

		if stencil.len() == 9 {
			// 9-point compact Laplacian.
			// Extract weights: w_c, w_ns, w_ew, w_diag.
			let entries = stencil.entries();
			let mut w_c = 0.0_f64;
			let mut w_ns = 0.0_f64;
			let mut w_ew = 0.0_f64;
			let mut w_diag = 0.0_f64;
			for e in entries {
				match (e.dr, e.dc) {
					(0, 0) => w_c = e.weight,
					(-1, 0) | (1, 0) => w_ns = e.weight,
					(0, -1) | (0, 1) => w_ew = e.weight,
					(-1, -1) | (-1, 1) | (1, -1)
					| (1, 1) => w_diag = e.weight,
					_ => {}
				}
			}
			let mut params: [*mut c_void; 8] = [
				(&input.ptr as *const _ as *mut c_void),
				(&output.ptr as *const _ as *mut c_void),
				(&mut rows as *mut _ as *mut c_void),
				(&mut cols as *mut _ as *mut c_void),
				(&mut w_c as *mut _ as *mut c_void),
				(&mut w_ns as *mut _ as *mut c_void),
				(&mut w_ew as *mut _ as *mut c_void),
				(&mut w_diag as *mut _ as *mut c_void),
			];
			self.launch(
				self.kernels.stencil_9pt,
				grid_dim,
				block_dim,
				0,
				&mut params,
			);
		} else {
			// 5-point Laplacian (default).
			let mut inv_dx2 =
				1.0 / (grid.dx * grid.dx);
			let mut inv_dy2 =
				1.0 / (grid.dy * grid.dy);
			let mut params: [*mut c_void; 6] = [
				(&input.ptr as *const _ as *mut c_void),
				(&output.ptr as *const _ as *mut c_void),
				(&mut rows as *mut _ as *mut c_void),
				(&mut cols as *mut _ as *mut c_void),
				(&mut inv_dx2 as *mut _ as *mut c_void),
				(&mut inv_dy2 as *mut _ as *mut c_void),
			];
			let kernel = match self.arch {
				ffi::GpuArch::Rdna
				| ffi::GpuArch::Unknown => {
					self.kernels.stencil_naive
				}
				ffi::GpuArch::Cdna => {
					self.kernels.stencil_2d
				}
			};
			self.launch(
				kernel, grid_dim, block_dim, 0,
				&mut params,
			);
		}
		Ok(())
	}

	fn apply_stencil_var(
		&self,
		input: &HipBuffer,
		output: &mut HipBuffer,
		coeff: &HipBuffer,
		grid: &Grid,
		stencil: &Stencil,
		boundaries: &Boundaries,
	) -> Result<()> {
		// Apply stencil, then pointwise multiply by coeff.
		self.apply_stencil(
			input, output, grid, stencil, boundaries,
		)?;
		// output[i] *= coeff[i] via pointwise_mult kernel.
		// pointwise_mult(x,y,z) writes z=x*y. We need
		// output = output * coeff. Use output as both
		// input and output — the kernel reads before writing
		// at each index, so in-place is safe.
		let mut len = output.len() as i32;
		let mut params: [*mut c_void; 4] = [
			(&coeff.ptr as *const _ as *mut c_void),
			(&output.ptr as *const _ as *mut c_void),
			(&output.ptr as *const _ as *mut c_void),
			(&mut len as *mut _ as *mut c_void),
		];
		self.launch(
			self.kernels.pointwise_mult,
			((len as u32).div_ceil(256), 1, 1),
			(256, 1, 1),
			0,
			&mut params,
		);
		Ok(())
	}

	#[inline]
	fn copy(
		&self,
		src: &HipBuffer,
		dst: &mut HipBuffer,
	) -> Result<()> {
		if src.len() != dst.len() {
			return Err(CoreError::DimensionMismatch {
				expected: src.len(),
				got: dst.len(),
			});
		}
		ffi::check(unsafe {
			ffi::hipMemcpyAsync(
				dst.ptr,
				src.ptr,
				src.size_bytes(),
				ffi::hipMemcpyDeviceToDevice,
				self.stream,
			)
		})
		.map_err(CoreError::BackendError)?;
		// Async copy enqueued on stream — mark dirty so
		// sync_if_dirty() waits before any host readback.
		self.stream_dirty.store(
			true,
			std::sync::atomic::Ordering::Release,
		);
		Ok(())
	}

	#[inline]
	fn fill(
		&self,
		buf: &mut HipBuffer,
		value: f64,
	) -> Result<()> {
		let mut val = value;
		let mut len = buf.len() as i32;
		let n = buf.len() as u32;

		let mut params: [*mut c_void; 3] = [
			(&buf.ptr as *const _ as *mut c_void),
			(&mut val as *mut _ as *mut c_void),
			(&mut len as *mut _ as *mut c_void),
		];

		self.launch(
			self.kernels.fill,
			(n.div_ceil(256), 1, 1),
			(256, 1, 1),
			0,
			&mut params,
		);
		Ok(())
	}

	#[inline]
	fn axpy(
		&self,
		alpha: f64,
		x: &HipBuffer,
		y: &mut HipBuffer,
	) -> Result<()> {
		if x.len() != y.len() {
			return Err(CoreError::DimensionMismatch {
				expected: x.len(),
				got: y.len(),
			});
		}
		let mut a = alpha;
		let mut len = x.len() as i32;
		let n = x.len() as u32;

		let mut params: [*mut c_void; 4] = [
			(&mut a as *mut _ as *mut c_void),
			(&x.ptr as *const _ as *mut c_void),
			(&y.ptr as *const _ as *mut c_void),
			(&mut len as *mut _ as *mut c_void),
		];

		self.launch(
			self.kernels.axpy,
			(n.div_ceil(256), 1, 1),
			(256, 1, 1),
			0,
			&mut params,
		);
		Ok(())
	}

	#[inline]
	fn scale(
		&self,
		buf: &mut HipBuffer,
		alpha: f64,
	) -> Result<()> {
		let mut a = alpha;
		let mut len = buf.len() as i32;
		let n = buf.len() as u32;

		let mut params: [*mut c_void; 3] = [
			(&buf.ptr as *const _ as *mut c_void),
			(&mut a as *mut _ as *mut c_void),
			(&mut len as *mut _ as *mut c_void),
		];

		self.launch(
			self.kernels.scale,
			(n.div_ceil(256), 1, 1),
			(256, 1, 1),
			0,
			&mut params,
		);
		Ok(())
	}

	#[inline]
	fn stencil_axpy(
		&self,
		alpha: f64,
		x: &HipBuffer,
		y: &mut HipBuffer,
		grid: &Grid,
		_stencil: &Stencil,
		_boundaries: &Boundaries,
	) -> Result<()> {
		let mut a = alpha;
		let mut rows = grid.rows as i32;
		let mut cols = grid.cols as i32;
		let mut inv_dx2 = 1.0 / (grid.dx * grid.dx);
		let mut inv_dy2 = 1.0 / (grid.dy * grid.dy);

		let mut params: [*mut c_void; 7] = [
			(&mut a as *mut _ as *mut c_void),
			(&x.ptr as *const _ as *mut c_void),
			(&y.ptr as *const _ as *mut c_void),
			(&mut rows as *mut _ as *mut c_void),
			(&mut cols as *mut _ as *mut c_void),
			(&mut inv_dx2 as *mut _ as *mut c_void),
			(&mut inv_dy2 as *mut _ as *mut c_void),
		];

		self.launch(
			self.kernels.fused_stencil_axpy,
			(
				(cols as u32).div_ceil(16),
				(rows as u32).div_ceil(16),
				1,
			),
			(16, 16, 1),
			0,
			&mut params,
		);
		Ok(())
	}

	#[inline]
	fn norm_l2(&self, buf: &HipBuffer) -> Result<f64> {
		let sum = self.run_reduction(
			self.kernels.reduce_sum_sq,
			buf,
		)?;
		Ok(sum.sqrt())
	}

	#[inline]
	fn reduce_sum(&self, buf: &HipBuffer) -> Result<f64> {
		self.run_reduction(self.kernels.reduce_sum, buf)
	}

	#[inline]
	fn reduce_max(&self, buf: &HipBuffer) -> Result<f64> {
		let n_blocks = self.reduce_blocks;
		self.ensure_reduce(n_blocks as usize);
		let scratch =
			self.reduce_buf.read().expect("reduce lock");

		let mut len = buf.len() as i32;
		let mut params: [*mut c_void; 3] = [
			(&buf.ptr as *const _ as *mut c_void),
			(&scratch.ptr as *const _ as *mut c_void),
			(&mut len as *mut _ as *mut c_void),
		];

		self.launch(
			self.kernels.reduce_max,
			(n_blocks, 1, 1),
			(256, 1, 1),
			0,
			&mut params,
		);

		let mut partials =
			[0.0_f64; Self::MAX_REDUCE_BLOCKS];
		unsafe {
			ffi::hipMemcpy(
				partials.as_mut_ptr().cast(),
				scratch.ptr,
				n_blocks as usize * 8,
				ffi::hipMemcpyDeviceToHost,
			);
		}
		self.stream_dirty.store(
			false,
			std::sync::atomic::Ordering::Release,
		);

		Ok(partials[..n_blocks as usize]
			.iter()
			.copied()
			.reduce(f64::max)
			.unwrap_or(f64::NEG_INFINITY))
	}

	#[inline]
	fn reduce_min(&self, buf: &HipBuffer) -> Result<f64> {
		let n_blocks = self.reduce_blocks;
		self.ensure_reduce(n_blocks as usize);
		let scratch =
			self.reduce_buf.read().expect("reduce lock");

		let mut len = buf.len() as i32;
		let mut params: [*mut c_void; 3] = [
			(&buf.ptr as *const _ as *mut c_void),
			(&scratch.ptr as *const _ as *mut c_void),
			(&mut len as *mut _ as *mut c_void),
		];

		self.launch(
			self.kernels.reduce_min,
			(n_blocks, 1, 1),
			(256, 1, 1),
			0,
			&mut params,
		);

		let mut partials =
			[0.0_f64; Self::MAX_REDUCE_BLOCKS];
		unsafe {
			ffi::hipMemcpy(
				partials.as_mut_ptr().cast(),
				scratch.ptr,
				n_blocks as usize * 8,
				ffi::hipMemcpyDeviceToHost,
			);
		}
		self.stream_dirty.store(
			false,
			std::sync::atomic::Ordering::Release,
		);

		Ok(partials[..n_blocks as usize]
			.iter()
			.copied()
			.reduce(f64::min)
			.unwrap_or(f64::INFINITY))
	}

	#[inline]
	fn dot(
		&self,
		x: &HipBuffer,
		y: &HipBuffer,
	) -> Result<f64> {
		if x.len() != y.len() {
			return Err(CoreError::DimensionMismatch {
				expected: x.len(),
				got: y.len(),
			});
		}
		let n_blocks = self.reduce_blocks;
		self.ensure_reduce(n_blocks as usize);
		let scratch =
			self.reduce_buf.read().expect("reduce lock");

		let mut len = x.len() as i32;
		let mut params: [*mut c_void; 4] = [
			(&x.ptr as *const _ as *mut c_void),
			(&y.ptr as *const _ as *mut c_void),
			(&scratch.ptr as *const _ as *mut c_void),
			(&mut len as *mut _ as *mut c_void),
		];

		self.launch(
			self.kernels.dot,
			(n_blocks, 1, 1),
			(256, 1, 1),
			0,
			&mut params,
		);

		let mut partials =
			[0.0_f64; Self::MAX_REDUCE_BLOCKS];
		unsafe {
			ffi::hipMemcpy(
				partials.as_mut_ptr().cast(),
				scratch.ptr,
				n_blocks as usize * 8,
				ffi::hipMemcpyDeviceToHost,
			);
		}
		self.stream_dirty.store(
			false,
			std::sync::atomic::Ordering::Release,
		);

		let mut sum = 0.0_f64;
		let mut comp = 0.0_f64;
		for &v in &partials[..n_blocks as usize] {
			let t_y = v - comp;
			let t = sum + t_y;
			comp = (t - sum) - t_y;
			sum = t;
		}
		Ok(sum)
	}

	fn dot2(
		&self,
		a: &HipBuffer,
		b: &HipBuffer,
		c: &HipBuffer,
		d: &HipBuffer,
	) -> Result<(f64, f64)> {
		let n_blocks = self.reduce_blocks;
		// Need 2*n_blocks elements in scratch.
		self.ensure_reduce(2 * n_blocks as usize);
		let scratch =
			self.reduce_buf.read().expect("reduce lock");

		let mut len = a.len() as i32;
		let mut params: [*mut c_void; 6] = [
			(&a.ptr as *const _ as *mut c_void),
			(&b.ptr as *const _ as *mut c_void),
			(&c.ptr as *const _ as *mut c_void),
			(&d.ptr as *const _ as *mut c_void),
			(&scratch.ptr as *const _ as *mut c_void),
			(&mut len as *mut _ as *mut c_void),
		];

		self.launch(
			self.kernels.dot2,
			(n_blocks, 1, 1),
			(256, 1, 1),
			0,
			&mut params,
		);

		// Read back 2*n_blocks partials.
		let mut partials =
			[0.0_f64; 2 * Self::MAX_REDUCE_BLOCKS];
		unsafe {
			ffi::hipMemcpy(
				partials.as_mut_ptr().cast(),
				scratch.ptr,
				2 * n_blocks as usize * 8,
				ffi::hipMemcpyDeviceToHost,
			);
		}
		self.stream_dirty.store(
			false,
			std::sync::atomic::Ordering::Release,
		);

		// Kahan sum each half.
		let nb = n_blocks as usize;
		let mut sum_ab = 0.0_f64;
		let mut comp_ab = 0.0_f64;
		for &v in &partials[..nb] {
			let y = v - comp_ab;
			let t = sum_ab + y;
			comp_ab = (t - sum_ab) - y;
			sum_ab = t;
		}
		let mut sum_cd = 0.0_f64;
		let mut comp_cd = 0.0_f64;
		for &v in &partials[nb..] {
			let y = v - comp_cd;
			let t = sum_cd + y;
			comp_cd = (t - sum_cd) - y;
			sum_cd = t;
		}

		Ok((sum_ab, sum_cd))
	}

	fn restrict(
		&self,
		fine: &HipBuffer,
		coarse: &mut HipBuffer,
		fine_grid: &Grid,
		coarse_grid: &Grid,
	) -> Result<()> {
		let mut fr = fine_grid.rows as i32;
		let mut fc = fine_grid.cols as i32;
		let mut cr = coarse_grid.rows as i32;
		let mut cc = coarse_grid.cols as i32;

		let mut params: [*mut c_void; 6] = [
			(&fine.ptr as *const _ as *mut c_void),
			(&coarse.ptr as *const _ as *mut c_void),
			(&mut fr as *mut _ as *mut c_void),
			(&mut fc as *mut _ as *mut c_void),
			(&mut cr as *mut _ as *mut c_void),
			(&mut cc as *mut _ as *mut c_void),
		];

		self.launch(
			self.kernels.restrict,
			(
				(cc as u32).div_ceil(16),
				(cr as u32).div_ceil(16),
				1,
			),
			(16, 16, 1),
			0,
			&mut params,
		);
		Ok(())
	}

	fn prolong(
		&self,
		coarse: &HipBuffer,
		fine: &mut HipBuffer,
		coarse_grid: &Grid,
		fine_grid: &Grid,
	) -> Result<()> {
		let mut fr = fine_grid.rows as i32;
		let mut fc = fine_grid.cols as i32;
		let mut cc = coarse_grid.cols as i32;

		let mut params: [*mut c_void; 5] = [
			(&coarse.ptr as *const _ as *mut c_void),
			(&fine.ptr as *const _ as *mut c_void),
			(&mut fr as *mut _ as *mut c_void),
			(&mut fc as *mut _ as *mut c_void),
			(&mut cc as *mut _ as *mut c_void),
		];

		self.launch(
			self.kernels.prolong,
			(
				(fc as u32).div_ceil(16),
				(fr as u32).div_ceil(16),
				1,
			),
			(16, 16, 1),
			0,
			&mut params,
		);
		Ok(())
	}

	fn weighted_jacobi(
		&self,
		x: &mut HipBuffer,
		b: &HipBuffer,
		omega: f64,
		grid: &Grid,
		_stencil: &Stencil,
		_boundaries: &Boundaries,
	) -> Result<()> {
		let mut om = omega;
		let mut rows = grid.rows as i32;
		let mut cols = grid.cols as i32;
		let mut inv_dx2 = 1.0 / (grid.dx * grid.dx);
		let mut inv_dy2 = 1.0 / (grid.dy * grid.dy);

		let mut params: [*mut c_void; 7] = [
			(&x.ptr as *const _ as *mut c_void),
			(&b.ptr as *const _ as *mut c_void),
			(&mut om as *mut _ as *mut c_void),
			(&mut rows as *mut _ as *mut c_void),
			(&mut cols as *mut _ as *mut c_void),
			(&mut inv_dx2 as *mut _ as *mut c_void),
			(&mut inv_dy2 as *mut _ as *mut c_void),
		];

		self.launch(
			self.kernels.weighted_jacobi,
			(
				(cols as u32).div_ceil(16),
				(rows as u32).div_ceil(16),
				1,
			),
			(16, 16, 1),
			0,
			&mut params,
		);
		Ok(())
	}

	// ── Extended vector operations ───────────────────

	#[inline]
	fn pointwise_mult(
		&self,
		x: &HipBuffer,
		y: &HipBuffer,
		z: &mut HipBuffer,
	) -> Result<()> {
		let mut len = x.len() as i32;
		let n = x.len() as u32;
		let mut params: [*mut c_void; 4] = [
			(&x.ptr as *const _ as *mut c_void),
			(&y.ptr as *const _ as *mut c_void),
			(&z.ptr as *const _ as *mut c_void),
			(&mut len as *mut _ as *mut c_void),
		];
		self.launch(
			self.kernels.pointwise_mult,
			(n.div_ceil(256), 1, 1),
			(256, 1, 1),
			0,
			&mut params,
		);
		Ok(())
	}

	#[inline]
	fn pointwise_div(
		&self,
		x: &HipBuffer,
		y: &HipBuffer,
		z: &mut HipBuffer,
	) -> Result<()> {
		let mut len = x.len() as i32;
		let n = x.len() as u32;
		let mut params: [*mut c_void; 4] = [
			(&x.ptr as *const _ as *mut c_void),
			(&y.ptr as *const _ as *mut c_void),
			(&z.ptr as *const _ as *mut c_void),
			(&mut len as *mut _ as *mut c_void),
		];
		self.launch(
			self.kernels.pointwise_div,
			(n.div_ceil(256), 1, 1),
			(256, 1, 1),
			0,
			&mut params,
		);
		Ok(())
	}

	#[inline]
	fn waxpy(
		&self,
		alpha: f64,
		x: &HipBuffer,
		beta: f64,
		y: &HipBuffer,
		w: &mut HipBuffer,
	) -> Result<()> {
		let mut a = alpha;
		let mut b = beta;
		let mut len = x.len() as i32;
		let n = x.len() as u32;
		let mut params: [*mut c_void; 6] = [
			(&mut a as *mut _ as *mut c_void),
			(&x.ptr as *const _ as *mut c_void),
			(&mut b as *mut _ as *mut c_void),
			(&y.ptr as *const _ as *mut c_void),
			(&w.ptr as *const _ as *mut c_void),
			(&mut len as *mut _ as *mut c_void),
		];
		self.launch(
			self.kernels.waxpy,
			(n.div_ceil(256), 1, 1),
			(256, 1, 1),
			0,
			&mut params,
		);
		Ok(())
	}

	#[inline]
	fn aypx(
		&self,
		alpha: f64,
		x: &HipBuffer,
		y: &mut HipBuffer,
	) -> Result<()> {
		let mut a = alpha;
		let mut len = x.len() as i32;
		let n = x.len() as u32;
		let mut params: [*mut c_void; 4] = [
			(&mut a as *mut _ as *mut c_void),
			(&x.ptr as *const _ as *mut c_void),
			(&y.ptr as *const _ as *mut c_void),
			(&mut len as *mut _ as *mut c_void),
		];
		self.launch(
			self.kernels.aypx,
			(n.div_ceil(256), 1, 1),
			(256, 1, 1),
			0,
			&mut params,
		);
		Ok(())
	}

	#[inline]
	fn reciprocal(
		&self,
		buf: &mut HipBuffer,
	) -> Result<()> {
		let mut len = buf.len() as i32;
		let n = buf.len() as u32;
		let mut params: [*mut c_void; 2] = [
			(&buf.ptr as *const _ as *mut c_void),
			(&mut len as *mut _ as *mut c_void),
		];
		self.launch(
			self.kernels.reciprocal,
			(n.div_ceil(256), 1, 1),
			(256, 1, 1),
			0,
			&mut params,
		);
		Ok(())
	}

	#[inline]
	fn abs_val(
		&self,
		buf: &mut HipBuffer,
	) -> Result<()> {
		let mut len = buf.len() as i32;
		let n = buf.len() as u32;
		let mut params: [*mut c_void; 2] = [
			(&buf.ptr as *const _ as *mut c_void),
			(&mut len as *mut _ as *mut c_void),
		];
		self.launch(
			self.kernels.abs_val,
			(n.div_ceil(256), 1, 1),
			(256, 1, 1),
			0,
			&mut params,
		);
		Ok(())
	}

	fn pointwise_max(
		&self,
		x: &HipBuffer,
		y: &HipBuffer,
		z: &mut HipBuffer,
	) -> Result<()> {
		let mut len = x.len() as i32;
		let n = x.len() as u32;
		let mut params: [*mut c_void; 4] = [
			(&x.ptr as *const _ as *mut c_void),
			(&y.ptr as *const _ as *mut c_void),
			(&z.ptr as *const _ as *mut c_void),
			(&mut len as *mut _ as *mut c_void),
		];
		self.launch(
			self.kernels.pointwise_max,
			(n.div_ceil(256), 1, 1),
			(256, 1, 1),
			0,
			&mut params,
		);
		Ok(())
	}

	fn pointwise_min(
		&self,
		x: &HipBuffer,
		y: &HipBuffer,
		z: &mut HipBuffer,
	) -> Result<()> {
		let mut len = x.len() as i32;
		let n = x.len() as u32;
		let mut params: [*mut c_void; 4] = [
			(&x.ptr as *const _ as *mut c_void),
			(&y.ptr as *const _ as *mut c_void),
			(&z.ptr as *const _ as *mut c_void),
			(&mut len as *mut _ as *mut c_void),
		];
		self.launch(
			self.kernels.pointwise_min,
			(n.div_ceil(256), 1, 1),
			(256, 1, 1),
			0,
			&mut params,
		);
		Ok(())
	}

	// ── Convection-diffusion ────────────────────────

	#[allow(clippy::too_many_arguments)]
	fn apply_conv_diff(
		&self,
		u: &HipBuffer,
		output: &mut HipBuffer,
		kappa: &HipBuffer,
		vx: &HipBuffer,
		vy: &HipBuffer,
		grid: &Grid,
		_boundaries: &Boundaries,
	) -> Result<()> {
		let mut rows = grid.rows as i32;
		let mut cols = grid.cols as i32;
		let mut inv_dx = 1.0 / grid.dx;
		let mut inv_dy = 1.0 / grid.dy;
		let mut inv_dx2 = inv_dx * inv_dx;
		let mut inv_dy2 = inv_dy * inv_dy;
		let mut params: [*mut c_void; 11] = [
			(&u.ptr as *const _ as *mut c_void),
			(&kappa.ptr as *const _ as *mut c_void),
			(&vx.ptr as *const _ as *mut c_void),
			(&vy.ptr as *const _ as *mut c_void),
			(&output.ptr as *const _ as *mut c_void),
			(&mut rows as *mut _ as *mut c_void),
			(&mut cols as *mut _ as *mut c_void),
			(&mut inv_dx as *mut _ as *mut c_void),
			(&mut inv_dy as *mut _ as *mut c_void),
			(&mut inv_dx2 as *mut _ as *mut c_void),
			(&mut inv_dy2 as *mut _ as *mut c_void),
		];
		self.launch(
			self.kernels.conv_diff,
			(
				(cols as u32).div_ceil(16),
				(rows as u32).div_ceil(16),
				1,
			),
			(16, 16, 1),
			0,
			&mut params,
		);
		Ok(())
	}

	#[allow(clippy::too_many_arguments)]
	fn conv_diff_axpy(
		&self,
		alpha: f64,
		u: &HipBuffer,
		output: &mut HipBuffer,
		kappa: &HipBuffer,
		vx: &HipBuffer,
		vy: &HipBuffer,
		grid: &Grid,
		_boundaries: &Boundaries,
	) -> Result<()> {
		let mut a = alpha;
		let mut rows = grid.rows as i32;
		let mut cols = grid.cols as i32;
		let mut inv_dx = 1.0 / grid.dx;
		let mut inv_dy = 1.0 / grid.dy;
		let mut inv_dx2 = inv_dx * inv_dx;
		let mut inv_dy2 = inv_dy * inv_dy;
		let mut params: [*mut c_void; 12] = [
			(&mut a as *mut _ as *mut c_void),
			(&u.ptr as *const _ as *mut c_void),
			(&kappa.ptr as *const _ as *mut c_void),
			(&vx.ptr as *const _ as *mut c_void),
			(&vy.ptr as *const _ as *mut c_void),
			(&output.ptr as *const _ as *mut c_void),
			(&mut rows as *mut _ as *mut c_void),
			(&mut cols as *mut _ as *mut c_void),
			(&mut inv_dx as *mut _ as *mut c_void),
			(&mut inv_dy as *mut _ as *mut c_void),
			(&mut inv_dx2 as *mut _ as *mut c_void),
			(&mut inv_dy2 as *mut _ as *mut c_void),
		];
		self.launch(
			self.kernels.conv_diff_axpy,
			(
				(cols as u32).div_ceil(16),
				(rows as u32).div_ceil(16),
				1,
			),
			(16, 16, 1),
			0,
			&mut params,
		);
		Ok(())
	}
}
