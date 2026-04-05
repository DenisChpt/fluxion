use std::sync::Arc;

use pollster::FutureExt as _;

use crate::error::WgpuError;

/// Shared GPU context — device + queue.
///
/// Cheap to clone (Arc-wrapped).
#[derive(Debug, Clone)]
pub struct GpuContext {
	pub device: Arc<wgpu::Device>,
	pub queue: Arc<wgpu::Queue>,
}

impl GpuContext {
	/// Create a context on the default adapter.
	///
	/// # Errors
	/// Returns an error if no GPU adapter is available.
	pub fn new() -> Result<Self, WgpuError> {
		let instance = wgpu::Instance::new(
			wgpu::InstanceDescriptor::new_without_display_handle_from_env(),
		);
		let adapter = instance
			.request_adapter(&wgpu::RequestAdapterOptions {
				power_preference:
					wgpu::PowerPreference::HighPerformance,
				..Default::default()
			})
			.block_on()
			.map_err(|_| WgpuError::NoAdapter)?;

		let (device, queue) = adapter
			.request_device(
				&wgpu::DeviceDescriptor::default(),
			)
			.block_on()
			.map_err(|e: wgpu::RequestDeviceError| {
				WgpuError::DeviceCreation(e.to_string())
			})?;

		Ok(Self {
			device: Arc::new(device),
			queue: Arc::new(queue),
		})
	}
}
