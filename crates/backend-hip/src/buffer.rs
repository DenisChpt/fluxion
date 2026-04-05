use fluxion_core::{BackendBuffer, DType};

use crate::ffi;

/// GPU buffer on a HIP device.
///
/// Owns a device pointer allocated via `hipMalloc`.
/// Data is stored as f64 natively (AMD GPUs have full
/// f64 throughput on CDNA).
///
/// The `dirty` flag tracks whether async operations have
/// been queued on this buffer since the last stream sync.
/// Reductions check this flag to skip redundant
/// `sync_stream()` calls when the buffer hasn't changed.
#[derive(Debug)]
pub struct HipBuffer {
	pub(crate) ptr: ffi::hipDeviceptr_t,
	len: usize,
	dtype: DType,
	/// True if async kernel writes are pending on the stream.
	pub(crate) dirty: bool,
}

// SAFETY: HIP device pointers are thread-safe (the HIP
// runtime serialises access per-stream).
unsafe impl Send for HipBuffer {}
unsafe impl Sync for HipBuffer {}

impl HipBuffer {
	/// Allocate a zero-initialized device buffer.
	pub(crate) fn zeros(
		len: usize,
		dtype: DType,
	) -> Result<Self, String> {
		let size = len * dtype.size_bytes();
		let mut ptr = std::ptr::null_mut();
		ffi::check(unsafe { ffi::hipMalloc(&mut ptr, size) })?;
		ffi::check(unsafe { ffi::hipMemset(ptr, 0, size) })?;
		Ok(Self { ptr, len, dtype, dirty: false })
	}

	/// Upload host f64 data to device.
	pub(crate) fn from_f64(data: &[f64]) -> Result<Self, String> {
		let size = data.len() * 8;
		let mut ptr = std::ptr::null_mut();
		ffi::check(unsafe { ffi::hipMalloc(&mut ptr, size) })?;
		ffi::check(unsafe {
			ffi::hipMemcpy(
				ptr,
				data.as_ptr().cast(),
				size,
				ffi::hipMemcpyHostToDevice,
			)
		})?;
		Ok(Self { ptr, len: data.len(), dtype: DType::F64, dirty: false })
	}

	/// Upload host f32 data to device.
	pub(crate) fn from_f32(data: &[f32]) -> Result<Self, String> {
		let size = data.len() * 4;
		let mut ptr = std::ptr::null_mut();
		ffi::check(unsafe { ffi::hipMalloc(&mut ptr, size) })?;
		ffi::check(unsafe {
			ffi::hipMemcpy(
				ptr,
				data.as_ptr().cast(),
				size,
				ffi::hipMemcpyHostToDevice,
			)
		})?;
		Ok(Self { ptr, len: data.len(), dtype: DType::F32, dirty: false })
	}

	/// Size in bytes on device.
	#[inline]
	pub(crate) const fn size_bytes(&self) -> usize {
		self.len * self.dtype.size_bytes()
	}
}

impl Drop for HipBuffer {
	fn drop(&mut self) {
		if !self.ptr.is_null() {
			unsafe { ffi::hipFree(self.ptr) };
		}
	}
}

impl BackendBuffer for HipBuffer {
	#[inline]
	fn len(&self) -> usize {
		self.len
	}

	#[inline]
	fn dtype(&self) -> DType {
		self.dtype
	}

	fn copy_to_host_f64(&self, dst: &mut [f64]) {
		assert_eq!(dst.len(), self.len);
		unsafe {
			ffi::hipMemcpy(
				dst.as_mut_ptr().cast(),
				self.ptr,
				self.len * 8,
				ffi::hipMemcpyDeviceToHost,
			);
		}
	}

	fn copy_to_host_f32(&self, dst: &mut [f32]) {
		assert_eq!(dst.len(), self.len);
		unsafe {
			ffi::hipMemcpy(
				dst.as_mut_ptr().cast(),
				self.ptr,
				self.len * 4,
				ffi::hipMemcpyDeviceToHost,
			);
		}
	}
}
