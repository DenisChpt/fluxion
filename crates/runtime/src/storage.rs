use fluxion_core::{BackendBuffer, DType};

use crate::device::Device;

/// Feature-gated enum dispatch over backend buffers.
///
/// Each variant wraps the concrete buffer type from its backend.
/// A match on this enum is the only runtime cost — one branch,
/// always correctly predicted since the backend never changes
/// mid-computation.
#[derive(Debug)]
pub enum BufferStorage {
	#[cfg(feature = "cpu")]
	Cpu(fluxion_backend_cpu::CpuBuffer),
	#[cfg(feature = "wgpu")]
	Wgpu(fluxion_backend_wgpu::WgpuBuffer),
	#[cfg(feature = "hip")]
	Hip(fluxion_backend_hip::HipBuffer),
}

impl BufferStorage {
	#[inline]
	#[must_use]
	pub const fn device(&self) -> Device {
		match self {
			#[cfg(feature = "cpu")]
			Self::Cpu(_) => Device::Cpu,
			#[cfg(feature = "wgpu")]
			Self::Wgpu(_) => Device::Wgpu { adapter: 0 },
			#[cfg(feature = "hip")]
			Self::Hip(_) => Device::Hip { ordinal: 0 },
		}
	}

	#[inline]
	#[must_use]
	pub fn len(&self) -> usize {
		match self {
			#[cfg(feature = "cpu")]
			Self::Cpu(b) => b.len(),
			#[cfg(feature = "wgpu")]
			Self::Wgpu(b) => b.len(),
			#[cfg(feature = "hip")]
			Self::Hip(b) => b.len(),
		}
	}

	#[inline]
	#[must_use]
	pub fn is_empty(&self) -> bool {
		self.len() == 0
	}

	#[inline]
	#[must_use]
	pub fn dtype(&self) -> DType {
		match self {
			#[cfg(feature = "cpu")]
			Self::Cpu(b) => b.dtype(),
			#[cfg(feature = "wgpu")]
			Self::Wgpu(b) => b.dtype(),
			#[cfg(feature = "hip")]
			Self::Hip(b) => b.dtype(),
		}
	}

	pub fn copy_to_host_f64(&self, dst: &mut [f64]) {
		match self {
			#[cfg(feature = "cpu")]
			Self::Cpu(b) => b.copy_to_host_f64(dst),
			#[cfg(feature = "wgpu")]
			Self::Wgpu(b) => b.copy_to_host_f64(dst),
			#[cfg(feature = "hip")]
			Self::Hip(b) => b.copy_to_host_f64(dst),
		}
	}

	pub fn copy_to_host_f32(&self, dst: &mut [f32]) {
		match self {
			#[cfg(feature = "cpu")]
			Self::Cpu(b) => b.copy_to_host_f32(dst),
			#[cfg(feature = "wgpu")]
			Self::Wgpu(b) => b.copy_to_host_f32(dst),
			#[cfg(feature = "hip")]
			Self::Hip(b) => b.copy_to_host_f32(dst),
		}
	}
}

// ── Backend singletons ──────────────────────────────

#[cfg(feature = "cpu")]
pub(crate) const fn cpu_backend(
) -> fluxion_backend_cpu::CpuBackend {
	fluxion_backend_cpu::CpuBackend::new()
}

#[cfg(feature = "wgpu")]
pub(crate) fn wgpu_backend(
) -> &'static fluxion_backend_wgpu::WgpuBackend {
	use std::sync::OnceLock;
	static INSTANCE: OnceLock<
		fluxion_backend_wgpu::WgpuBackend,
	> = OnceLock::new();
	INSTANCE.get_or_init(|| {
		fluxion_backend_wgpu::WgpuBackend::new()
			.expect("wgpu backend initialization failed")
	})
}

#[cfg(feature = "hip")]
pub(crate) fn hip_backend(
) -> &'static fluxion_backend_hip::HipBackend {
	use std::sync::OnceLock;
	static INSTANCE: OnceLock<
		fluxion_backend_hip::HipBackend,
	> = OnceLock::new();
	INSTANCE.get_or_init(|| {
		fluxion_backend_hip::HipBackend::new()
			.expect("HIP backend initialization failed")
	})
}
