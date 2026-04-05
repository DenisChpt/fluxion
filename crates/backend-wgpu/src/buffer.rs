use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use fluxion_core::{BackendBuffer, DType};

/// Monotonic counter for unique buffer IDs.
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

/// GPU buffer backed by a `wgpu::Buffer`.
///
/// Data is stored as f32 on the GPU (WGSL has no f64).
/// Conversions happen at upload/download boundaries.
#[derive(Debug)]
pub struct WgpuBuffer {
	pub(crate) buf: wgpu::Buffer,
	pub(crate) device: Arc<wgpu::Device>,
	pub(crate) queue: Arc<wgpu::Queue>,
	/// Stable unique ID for bind-group caching.
	pub(crate) id: u64,
	len: usize,
	dtype: DType,
}

impl WgpuBuffer {
	pub(crate) fn new(
		device: Arc<wgpu::Device>,
		queue: Arc<wgpu::Queue>,
		buf: wgpu::Buffer,
		len: usize,
		dtype: DType,
	) -> Self {
		let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
		Self { buf, device, queue, id, len, dtype }
	}

	pub(crate) const fn size_bytes(&self) -> u64 {
		(self.len * 4) as u64 // Always f32 on GPU.
	}

	/// Read buffer contents back as f32.
	pub(crate) fn read_f32(&self) -> Vec<f32> {
		let size = self.size_bytes();
		let staging = self.device.create_buffer(
			&wgpu::BufferDescriptor {
				label: Some("staging_read"),
				size,
				usage: wgpu::BufferUsages::MAP_READ
					| wgpu::BufferUsages::COPY_DST,
				mapped_at_creation: false,
			},
		);

		let mut encoder = self.device.create_command_encoder(
			&wgpu::CommandEncoderDescriptor::default(),
		);
		encoder.copy_buffer_to_buffer(
			&self.buf, 0, &staging, 0, size,
		);
		self.queue.submit(std::iter::once(encoder.finish()));

		let slice = staging.slice(..);
		slice.map_async(wgpu::MapMode::Read, |_| {});
		self.device
			.poll(wgpu::PollType::wait_indefinitely())
			.expect("device poll failed");

		let data = slice.get_mapped_range();
		let result: Vec<f32> =
			bytemuck::cast_slice(&data).to_vec();
		drop(data);
		staging.unmap();
		result
	}
}

impl BackendBuffer for WgpuBuffer {
	fn len(&self) -> usize {
		self.len
	}

	fn dtype(&self) -> DType {
		self.dtype
	}

	fn copy_to_host_f64(&self, dst: &mut [f64]) {
		assert_eq!(dst.len(), self.len);
		let f32_data = self.read_f32();
		for (d, &s) in dst.iter_mut().zip(f32_data.iter()) {
			*d = f64::from(s);
		}
	}

	fn copy_to_host_f32(&self, dst: &mut [f32]) {
		assert_eq!(dst.len(), self.len);
		let f32_data = self.read_f32();
		dst.copy_from_slice(&f32_data);
	}
}
