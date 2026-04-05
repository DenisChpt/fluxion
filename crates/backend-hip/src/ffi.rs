//! Minimal HIP runtime FFI bindings.
//!
//! Only the functions we actually use. No bindgen dependency.

#![allow(
	non_camel_case_types,
	non_upper_case_globals,
	dead_code
)]

use std::ffi::{c_char, c_int, c_uint, c_void};

/// HIP error codes (subset).
pub type hipError_t = c_int;
pub const HIP_SUCCESS: hipError_t = 0;

/// Opaque device pointer.
pub type hipDeviceptr_t = *mut c_void;

/// HIP stream handle.
pub type hipStream_t = *mut c_void;

/// HIP module handle (loaded code object).
pub type hipModule_t = *mut c_void;

/// HIP function handle (kernel in a module).
pub type hipFunction_t = *mut c_void;

/// Memory copy kind.
pub type hipMemcpyKind = c_uint;
pub const hipMemcpyHostToDevice: hipMemcpyKind = 1;
pub const hipMemcpyDeviceToHost: hipMemcpyKind = 2;
pub const hipMemcpyDeviceToDevice: hipMemcpyKind = 3;

unsafe extern "C" {
	// ── Device management ────────────────────────────

	pub fn hipGetDeviceCount(count: *mut c_int) -> hipError_t;
	pub fn hipSetDevice(device: c_int) -> hipError_t;
	pub fn hipDeviceSynchronize() -> hipError_t;

	// ── Memory management ────────────────────────────

	pub fn hipMalloc(
		ptr: *mut hipDeviceptr_t,
		size: usize,
	) -> hipError_t;

	pub fn hipFree(ptr: hipDeviceptr_t) -> hipError_t;

	pub fn hipMemcpy(
		dst: *mut c_void,
		src: *const c_void,
		size: usize,
		kind: hipMemcpyKind,
	) -> hipError_t;

	pub fn hipMemcpyAsync(
		dst: *mut c_void,
		src: *const c_void,
		size: usize,
		kind: hipMemcpyKind,
		stream: hipStream_t,
	) -> hipError_t;

	pub fn hipMemset(
		ptr: hipDeviceptr_t,
		value: c_int,
		size: usize,
	) -> hipError_t;

	pub fn hipMemcpyDtoD(
		dst: hipDeviceptr_t,
		src: hipDeviceptr_t,
		size: usize,
	) -> hipError_t;

	// ── Stream management ────────────────────────────

	pub fn hipStreamCreate(
		stream: *mut hipStream_t,
	) -> hipError_t;

	pub fn hipStreamDestroy(
		stream: hipStream_t,
	) -> hipError_t;

	pub fn hipStreamSynchronize(
		stream: hipStream_t,
	) -> hipError_t;

	// ── Module / kernel management ───────────────────

	pub fn hipModuleLoad(
		module: *mut hipModule_t,
		fname: *const c_char,
	) -> hipError_t;

	pub fn hipModuleLoadData(
		module: *mut hipModule_t,
		image: *const c_void,
	) -> hipError_t;

	pub fn hipModuleGetFunction(
		function: *mut hipFunction_t,
		module: hipModule_t,
		name: *const c_char,
	) -> hipError_t;

	pub fn hipModuleLaunchKernel(
		f: hipFunction_t,
		grid_dim_x: c_uint,
		grid_dim_y: c_uint,
		grid_dim_z: c_uint,
		block_dim_x: c_uint,
		block_dim_y: c_uint,
		block_dim_z: c_uint,
		shared_mem_bytes: c_uint,
		stream: hipStream_t,
		kernel_params: *mut *mut c_void,
		extra: *mut *mut c_void,
	) -> hipError_t;

	// ── Device properties ────────────────────────────

	pub fn hipDeviceGetAttribute(
		pi: *mut c_int,
		attr: c_int,
		device: c_int,
	) -> hipError_t;

	// ── Error handling ───────────────────────────────

	pub fn hipGetErrorString(
		error: hipError_t,
	) -> *const c_char;

	// ── hipGraph ─────────────────────────────────────

	pub fn hipStreamBeginCapture(
		stream: hipStream_t,
		mode: c_uint,
	) -> hipError_t;

	pub fn hipStreamEndCapture(
		stream: hipStream_t,
		graph: *mut hipGraph_t,
	) -> hipError_t;

	pub fn hipGraphInstantiate(
		exec: *mut hipGraphExec_t,
		graph: hipGraph_t,
		err_node: *mut hipGraphNode_t,
		log: *mut c_char,
		log_size: usize,
	) -> hipError_t;

	pub fn hipGraphLaunch(
		exec: hipGraphExec_t,
		stream: hipStream_t,
	) -> hipError_t;

	pub fn hipGraphExecDestroy(
		exec: hipGraphExec_t,
	) -> hipError_t;

	pub fn hipGraphDestroy(
		graph: hipGraph_t,
	) -> hipError_t;
}

/// hipGraph handle.
pub type hipGraph_t = *mut c_void;
/// hipGraphExec handle (instantiated graph).
pub type hipGraphExec_t = *mut c_void;
/// hipGraphNode handle.
pub type hipGraphNode_t = *mut c_void;

/// `hipStreamCaptureMode` values.
pub const hipStreamCaptureModeGlobal: c_uint = 0;
pub const hipStreamCaptureModeThreadLocal: c_uint = 1;
pub const hipStreamCaptureModeRelaxed: c_uint = 2;

/// `hipDeviceAttribute_t` values we use.
pub const HIP_DEVICE_ATTRIBUTE_WARP_SIZE: c_int = 10;
pub const HIP_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: c_int = 16;
pub const HIP_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: c_int =
	75;

/// GPU architecture family detected at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuArch {
	/// RDNA (gfx10xx, gfx11xx) — wavefront 32, large L1,
	/// f64 at 1/16 rate. Prefer naive stencil (no LDS).
	Rdna,
	/// CDNA (gfx9xx) — wavefront 64, LDS-optimised,
	/// full f64 throughput. Prefer tiled stencil.
	Cdna,
	/// Unknown — fall back to naive (safe default).
	Unknown,
}

/// Detect the GPU architecture of device 0.
pub fn detect_gpu_arch() -> GpuArch {
	let mut warp_size: c_int = 0;
	let mut major: c_int = 0;

	let err = unsafe {
		hipDeviceGetAttribute(
			&mut warp_size,
			HIP_DEVICE_ATTRIBUTE_WARP_SIZE,
			0,
		)
	};
	if err != HIP_SUCCESS {
		return GpuArch::Unknown;
	}

	let _ = unsafe {
		hipDeviceGetAttribute(
			&mut major,
			HIP_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
			0,
		)
	};

	// CDNA (MI-series): major >= 9, wavefront 64.
	// RDNA (RX-series): major >= 10, wavefront 32 (default).
	if major >= 10 || warp_size == 32 {
		GpuArch::Rdna
	} else if major >= 9 || warp_size == 64 {
		GpuArch::Cdna
	} else {
		GpuArch::Unknown
	}
}

/// Check a HIP call and return a Rust Result.
#[inline]
pub fn check(err: hipError_t) -> Result<(), String> {
	if err == HIP_SUCCESS {
		Ok(())
	} else {
		let msg = unsafe {
			let ptr = hipGetErrorString(err);
			if ptr.is_null() {
				format!("HIP error {err}")
			} else {
				std::ffi::CStr::from_ptr(ptr)
					.to_string_lossy()
					.into_owned()
			}
		};
		Err(msg)
	}
}
