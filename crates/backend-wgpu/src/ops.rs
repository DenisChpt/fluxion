use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use fluxion_core::{
	Backend, BackendBuffer, Boundaries, CoreError, DType, Grid,
	Result, Stencil,
};
use wgpu::util::DeviceExt as _;

use crate::buffer::WgpuBuffer;
use crate::context::GpuContext;

const STENCIL_WGSL: &str =
	include_str!("../../../kernels/wgsl/stencil_2d.wgsl");
const AXPY_WGSL: &str =
	include_str!("../../../kernels/wgsl/axpy.wgsl");
const FILL_WGSL: &str =
	include_str!("../../../kernels/wgsl/fill.wgsl");
const SCALE_WGSL: &str =
	include_str!("../../../kernels/wgsl/scale.wgsl");
const REDUCE_SUM_SQ_WGSL: &str =
	include_str!("../../../kernels/wgsl/reduce_sum_sq.wgsl");
const REDUCE_SUM_WGSL: &str =
	include_str!("../../../kernels/wgsl/reduce_sum.wgsl");
const REDUCE_MAX_WGSL: &str =
	include_str!("../../../kernels/wgsl/reduce_max.wgsl");
const REDUCE_MIN_WGSL: &str =
	include_str!("../../../kernels/wgsl/reduce_min.wgsl");
const FUSED_STENCIL_AXPY_WGSL: &str = include_str!(
	"../../../kernels/wgsl/fused_stencil_axpy.wgsl"
);
const DOT_PRODUCT_WGSL: &str =
	include_str!("../../../kernels/wgsl/dot_product.wgsl");
const RESTRICT_WGSL: &str =
	include_str!("../../../kernels/wgsl/restrict.wgsl");
const PROLONG_WGSL: &str =
	include_str!("../../../kernels/wgsl/prolong.wgsl");
const JACOBI_WGSL: &str =
	include_str!("../../../kernels/wgsl/jacobi.wgsl");

/// Maximum uniform buffer size (24 bytes for fused `stencil_axpy`).
const UNIFORM_SIZE: u64 = 24;

/// Pre-allocated reduction scratch buffers (grow-only).
#[derive(Debug)]
struct ReduceScratch {
	/// Partial results (one f32 per workgroup).
	partial: wgpu::Buffer,
	/// Staging for CPU readback.
	staging: wgpu::Buffer,
	/// Current capacity in number of f32 elements.
	capacity: u32,
}

/// Bind group cache keyed by buffer IDs.
/// Holds all recently-used bind groups for a given pipeline shape.
#[derive(Debug, Default)]
struct BindGroupCache {
	groups: HashMap<u128, wgpu::BindGroup>,
}

impl BindGroupCache {
	/// Pack two u64 buffer IDs into a single u128 key.
	fn key2(a: u64, b: u64) -> u128 {
		(u128::from(a) << 64) | u128::from(b)
	}

	/// Single buffer ID as key.
	fn key1(a: u64) -> u128 {
		u128::from(a)
	}
}

/// Portable GPU backend via wgpu.
///
/// All data is stored as f32 on the GPU (WGSL limitation).
/// Uniform buffers, pipelines, and reduction scratch are
/// pre-allocated — minimal allocation in the hot path.
#[derive(Debug)]
pub struct WgpuBackend {
	ctx: GpuContext,
	stencil_pipeline: wgpu::ComputePipeline,
	axpy_pipeline: wgpu::ComputePipeline,
	fill_pipeline: wgpu::ComputePipeline,
	scale_pipeline: wgpu::ComputePipeline,
	reduce_sum_sq_pipeline: wgpu::ComputePipeline,
	reduce_sum_pipeline: wgpu::ComputePipeline,
	reduce_max_pipeline: wgpu::ComputePipeline,
	reduce_min_pipeline: wgpu::ComputePipeline,
	fused_stencil_axpy_pipeline: wgpu::ComputePipeline,
	dot_pipeline: wgpu::ComputePipeline,
	restrict_pipeline: wgpu::ComputePipeline,
	prolong_pipeline: wgpu::ComputePipeline,
	jacobi_pipeline: wgpu::ComputePipeline,
	/// Reusable uniform buffer (16 bytes, covers all ops).
	uniform_buf: wgpu::Buffer,
	/// Grow-only scratch for reduction operations.
	reduce_scratch: RwLock<ReduceScratch>,
	/// Bind group caches keyed by buffer IDs.
	bg_stencil: RwLock<BindGroupCache>,
	bg_axpy: RwLock<BindGroupCache>,
	bg_fill: RwLock<BindGroupCache>,
	bg_scale: RwLock<BindGroupCache>,
	bg_fused: RwLock<BindGroupCache>,
}

impl WgpuBackend {
	/// Create a new wgpu backend.
	///
	/// Pre-allocates reusable buffers. Pipelines are compiled
	/// once and reused for every operation.
	///
	/// # Errors
	/// Returns an error if no GPU is available.
	pub fn new(
	) -> std::result::Result<Self, crate::error::WgpuError> {
		let ctx = GpuContext::new()?;
		let device = &ctx.device;

		let stencil_pipeline =
			create_pipeline(device, STENCIL_WGSL, "stencil");
		let axpy_pipeline =
			create_pipeline(device, AXPY_WGSL, "axpy");
		let fill_pipeline =
			create_pipeline(device, FILL_WGSL, "fill");
		let scale_pipeline =
			create_pipeline(device, SCALE_WGSL, "scale");
		let reduce_sum_sq_pipeline = create_pipeline(
			device,
			REDUCE_SUM_SQ_WGSL,
			"reduce_sum_sq",
		);
		let reduce_sum_pipeline = create_pipeline(
			device,
			REDUCE_SUM_WGSL,
			"reduce_sum",
		);
		let reduce_max_pipeline = create_pipeline(
			device,
			REDUCE_MAX_WGSL,
			"reduce_max",
		);
		let reduce_min_pipeline = create_pipeline(
			device,
			REDUCE_MIN_WGSL,
			"reduce_min",
		);
		let fused_stencil_axpy_pipeline = create_pipeline(
			device,
			FUSED_STENCIL_AXPY_WGSL,
			"fused_stencil_axpy",
		);
		let dot_pipeline = create_pipeline(
			device,
			DOT_PRODUCT_WGSL,
			"dot_product",
		);
		let restrict_pipeline = create_pipeline(
			device,
			RESTRICT_WGSL,
			"restrict",
		);
		let prolong_pipeline = create_pipeline(
			device, PROLONG_WGSL, "prolong",
		);
		let jacobi_pipeline = create_pipeline(
			device, JACOBI_WGSL, "jacobi",
		);

		let uniform_buf =
			device.create_buffer(&wgpu::BufferDescriptor {
				label: Some("shared_uniform"),
				size: UNIFORM_SIZE,
				usage: wgpu::BufferUsages::UNIFORM
					| wgpu::BufferUsages::COPY_DST,
				mapped_at_creation: false,
			});

		// Initial reduction scratch: 64 workgroups worth.
		let init_cap: u32 = 64;
		let reduce_scratch = RwLock::new(ReduceScratch {
			partial: create_reduce_partial(device, init_cap),
			staging: create_reduce_staging(device, init_cap),
			capacity: init_cap,
		});

		Ok(Self {
			ctx,
			stencil_pipeline,
			axpy_pipeline,
			fill_pipeline,
			scale_pipeline,
			reduce_sum_sq_pipeline,
			reduce_sum_pipeline,
			reduce_max_pipeline,
			reduce_min_pipeline,
			fused_stencil_axpy_pipeline,
			dot_pipeline,
			restrict_pipeline,
			prolong_pipeline,
			jacobi_pipeline,
			uniform_buf,
			reduce_scratch,
			bg_stencil: RwLock::new(BindGroupCache::default()),
			bg_axpy: RwLock::new(BindGroupCache::default()),
			bg_fill: RwLock::new(BindGroupCache::default()),
			bg_scale: RwLock::new(BindGroupCache::default()),
			bg_fused: RwLock::new(BindGroupCache::default()),
		})
	}

	fn make_buffer(
		&self,
		data: &[f32],
		dtype: DType,
	) -> WgpuBuffer {
		let buf = self.ctx.device.create_buffer_init(
			&wgpu::util::BufferInitDescriptor {
				label: None,
				contents: bytemuck::cast_slice(data),
				usage: wgpu::BufferUsages::STORAGE
					| wgpu::BufferUsages::COPY_SRC
					| wgpu::BufferUsages::COPY_DST,
			},
		);
		WgpuBuffer::new(
			Arc::clone(&self.ctx.device),
			Arc::clone(&self.ctx.queue),
			buf,
			data.len(),
			dtype,
		)
	}

	/// Write params into the shared uniform buffer (no alloc).
	#[inline]
	fn write_uniform(&self, data: &[u8]) {
		self.ctx
			.queue
			.write_buffer(&self.uniform_buf, 0, data);
	}

	/// Encode + submit a single compute dispatch.
	#[inline]
	fn dispatch(
		&self,
		pipeline: &wgpu::ComputePipeline,
		bind_group: &wgpu::BindGroup,
		groups: (u32, u32, u32),
	) {
		let mut encoder =
			self.ctx.device.create_command_encoder(
				&wgpu::CommandEncoderDescriptor::default(),
			);
		Self::encode_dispatch(
			&mut encoder, pipeline, bind_group, groups,
		);
		self.ctx
			.queue
			.submit(std::iter::once(encoder.finish()));
	}

	/// Encode a dispatch into an existing encoder (no submit).
	#[inline]
	fn encode_dispatch(
		encoder: &mut wgpu::CommandEncoder,
		pipeline: &wgpu::ComputePipeline,
		bind_group: &wgpu::BindGroup,
		groups: (u32, u32, u32),
	) {
		let mut pass = encoder.begin_compute_pass(
			&wgpu::ComputePassDescriptor::default(),
		);
		pass.set_pipeline(pipeline);
		pass.set_bind_group(0, bind_group, &[]);
		pass.dispatch_workgroups(
			groups.0, groups.1, groups.2,
		);
	}

	/// Begin a batch: returns an encoder that accumulates
	/// GPU commands. Call `submit_batch` to flush.
	pub fn begin_batch(&self) -> wgpu::CommandEncoder {
		self.ctx.device.create_command_encoder(
			&wgpu::CommandEncoderDescriptor {
				label: Some("batch"),
			},
		)
	}

	/// Submit a batched encoder — single GPU submit.
	pub fn submit_batch(
		&self,
		encoder: wgpu::CommandEncoder,
	) {
		self.ctx
			.queue
			.submit(std::iter::once(encoder.finish()));
	}

	/// Encode a `stencil_axpy` into an existing batch.
	#[allow(clippy::similar_names)]
	pub fn encode_stencil_axpy(
		&self,
		encoder: &mut wgpu::CommandEncoder,
		alpha: f64,
		x: &WgpuBuffer,
		y: &WgpuBuffer,
		grid: &Grid,
	) {
		let inv_dx2 = (1.0 / (grid.dx * grid.dx)) as f32;
		let inv_dy2 = (1.0 / (grid.dy * grid.dy)) as f32;
		let mut params = [0u8; 24];
		params[0..4].copy_from_slice(
			&(grid.rows as u32).to_le_bytes(),
		);
		params[4..8].copy_from_slice(
			&(grid.cols as u32).to_le_bytes(),
		);
		params[8..12]
			.copy_from_slice(&inv_dx2.to_le_bytes());
		params[12..16]
			.copy_from_slice(&inv_dy2.to_le_bytes());
		params[16..20].copy_from_slice(
			&(alpha as f32).to_le_bytes(),
		);
		params[20..24]
			.copy_from_slice(&0u32.to_le_bytes());
		self.write_uniform(&params);

		let bind_group = self.get_or_create_bg2(
			&self.bg_fused,
			&self.fused_stencil_axpy_pipeline,
			x,
			y,
		);

		Self::encode_dispatch(
			encoder,
			&self.fused_stencil_axpy_pipeline,
			&bind_group,
			(
				(grid.cols as u32).div_ceil(16),
				(grid.rows as u32).div_ceil(8),
				1,
			),
		);
	}

	/// Encode an axpy into an existing batch.
	pub fn encode_axpy(
		&self,
		encoder: &mut wgpu::CommandEncoder,
		alpha: f64,
		x: &WgpuBuffer,
		y: &WgpuBuffer,
	) {
		let len = x.len() as u32;
		let mut params = [0u8; 24];
		params[0..4].copy_from_slice(
			&(alpha as f32).to_le_bytes(),
		);
		params[4..8]
			.copy_from_slice(&len.to_le_bytes());
		self.write_uniform(&params);

		let bind_group = self.get_or_create_bg2(
			&self.bg_axpy,
			&self.axpy_pipeline,
			x,
			y,
		);

		Self::encode_dispatch(
			encoder,
			&self.axpy_pipeline,
			&bind_group,
			(len.div_ceil(256), 1, 1),
		);
	}

	/// Encode a buffer copy into an existing batch.
	pub fn encode_copy(
		&self,
		encoder: &mut wgpu::CommandEncoder,
		src: &WgpuBuffer,
		dst: &WgpuBuffer,
	) {
		encoder.copy_buffer_to_buffer(
			&src.buf,
			0,
			&dst.buf,
			0,
			src.size_bytes(),
		);
	}

	/// Encode a scale into an existing batch.
	pub fn encode_scale(
		&self,
		encoder: &mut wgpu::CommandEncoder,
		buf: &WgpuBuffer,
		alpha: f64,
	) {
		let len = buf.len() as u32;
		let mut params = [0u8; 24];
		params[0..4].copy_from_slice(
			&(alpha as f32).to_le_bytes(),
		);
		params[4..8]
			.copy_from_slice(&len.to_le_bytes());
		self.write_uniform(&params);

		let bind_group = self.get_or_create_bg1(
			&self.bg_scale,
			&self.scale_pipeline,
			buf,
		);

		Self::encode_dispatch(
			encoder,
			&self.scale_pipeline,
			&bind_group,
			(len.div_ceil(256), 1, 1),
		);
	}

	/// Encode a stencil into an existing batch.
	#[allow(clippy::similar_names)]
	pub fn encode_stencil(
		&self,
		encoder: &mut wgpu::CommandEncoder,
		input: &WgpuBuffer,
		output: &WgpuBuffer,
		grid: &Grid,
	) {
		let inv_dx2 = (1.0 / (grid.dx * grid.dx)) as f32;
		let inv_dy2 = (1.0 / (grid.dy * grid.dy)) as f32;
		let mut params = [0u8; 24];
		params[0..4].copy_from_slice(
			&(grid.rows as u32).to_le_bytes(),
		);
		params[4..8].copy_from_slice(
			&(grid.cols as u32).to_le_bytes(),
		);
		params[8..12]
			.copy_from_slice(&inv_dx2.to_le_bytes());
		params[12..16]
			.copy_from_slice(&inv_dy2.to_le_bytes());
		self.write_uniform(&params);

		let bind_group = self.get_or_create_bg2(
			&self.bg_stencil,
			&self.stencil_pipeline,
			input,
			output,
		);

		Self::encode_dispatch(
			encoder,
			&self.stencil_pipeline,
			&bind_group,
			(
				(grid.cols as u32).div_ceil(16),
				(grid.rows as u32).div_ceil(8),
				1,
			),
		);
	}

	/// Get or create a bind group for a 3-binding op.
	fn get_or_create_bg2(
		&self,
		cache: &RwLock<BindGroupCache>,
		pipeline: &wgpu::ComputePipeline,
		a: &WgpuBuffer,
		b: &WgpuBuffer,
	) -> wgpu::BindGroup {
		let key = BindGroupCache::key2(a.id, b.id);
		{
			let c = cache.read().expect("bg cache lock");
			if let Some(bg) = c.groups.get(&key) {
				return bg.clone();
			}
		}
		let layout = pipeline.get_bind_group_layout(0);
		let bg = self.ctx.device.create_bind_group(
			&wgpu::BindGroupDescriptor {
				label: None,
				layout: &layout,
				entries: &[
					wgpu::BindGroupEntry {
						binding: 0,
						resource: self
							.uniform_buf
							.as_entire_binding(),
					},
					wgpu::BindGroupEntry {
						binding: 1,
						resource: a.buf.as_entire_binding(),
					},
					wgpu::BindGroupEntry {
						binding: 2,
						resource: b.buf.as_entire_binding(),
					},
				],
			},
		);
		cache
			.write()
			.expect("bg cache lock")
			.groups
			.insert(key, bg.clone());
		bg
	}

	/// Get or create a bind group for a 2-binding op.
	fn get_or_create_bg1(
		&self,
		cache: &RwLock<BindGroupCache>,
		pipeline: &wgpu::ComputePipeline,
		buf: &WgpuBuffer,
	) -> wgpu::BindGroup {
		let key = BindGroupCache::key1(buf.id);
		{
			let c = cache.read().expect("bg cache lock");
			if let Some(bg) = c.groups.get(&key) {
				return bg.clone();
			}
		}
		let layout = pipeline.get_bind_group_layout(0);
		let bg = self.ctx.device.create_bind_group(
			&wgpu::BindGroupDescriptor {
				label: None,
				layout: &layout,
				entries: &[
					wgpu::BindGroupEntry {
						binding: 0,
						resource: self
							.uniform_buf
							.as_entire_binding(),
					},
					wgpu::BindGroupEntry {
						binding: 1,
						resource: buf.buf.as_entire_binding(),
					},
				],
			},
		);
		cache
			.write()
			.expect("bg cache lock")
			.groups
			.insert(key, bg.clone());
		bg
	}

	/// Ensure reduction scratch can hold `n_groups` partials.
	/// Grows but never shrinks.
	fn ensure_reduce_capacity(&self, n_groups: u32) {
		let mut scratch = self.reduce_scratch.write().expect("reduce scratch lock");
		if n_groups <= scratch.capacity {
			return;
		}
		// Grow to next power of two.
		let new_cap = n_groups.next_power_of_two();
		scratch.partial =
			create_reduce_partial(&self.ctx.device, new_cap);
		scratch.staging =
			create_reduce_staging(&self.ctx.device, new_cap);
		scratch.capacity = new_cap;
	}

	/// Run a reduction pipeline and read back partial results.
	fn run_reduction(
		&self,
		pipeline: &wgpu::ComputePipeline,
		buf: &WgpuBuffer,
	) -> Vec<f32> {
		let len = buf.len() as u32;
		let n_groups = len.div_ceil(256);

		let mut params = [0u8; 16];
		params[0..4]
			.copy_from_slice(&len.to_le_bytes());
		self.write_uniform(&params);

		self.ensure_reduce_capacity(n_groups);
		let scratch = self.reduce_scratch.read().expect("reduce scratch lock");

		let layout = pipeline.get_bind_group_layout(0);
		let bind_group =
			self.ctx.device.create_bind_group(
				&wgpu::BindGroupDescriptor {
					label: None,
					layout: &layout,
					entries: &[
						wgpu::BindGroupEntry {
							binding: 0,
							resource: self
								.uniform_buf
								.as_entire_binding(),
						},
						wgpu::BindGroupEntry {
							binding: 1,
							resource: buf
								.buf
								.as_entire_binding(),
						},
						wgpu::BindGroupEntry {
							binding: 2,
							resource: scratch
								.partial
								.as_entire_binding(),
						},
					],
				},
			);

		let staging_size = u64::from(n_groups) * 4;
		let mut encoder =
			self.ctx.device.create_command_encoder(
				&wgpu::CommandEncoderDescriptor::default(),
			);
		{
			let mut pass = encoder.begin_compute_pass(
				&wgpu::ComputePassDescriptor::default(),
			);
			pass.set_pipeline(pipeline);
			pass.set_bind_group(0, &bind_group, &[]);
			pass.dispatch_workgroups(n_groups, 1, 1);
		}
		encoder.copy_buffer_to_buffer(
			&scratch.partial,
			0,
			&scratch.staging,
			0,
			staging_size,
		);
		self.ctx
			.queue
			.submit(std::iter::once(encoder.finish()));

		let slice = scratch.staging.slice(..staging_size);
		slice.map_async(wgpu::MapMode::Read, |_| {});
		self.ctx
			.device
			.poll(wgpu::PollType::wait_indefinitely())
			.expect("device poll failed");

		let data = slice.get_mapped_range();
		let partials: Vec<f32> =
			bytemuck::cast_slice(&data).to_vec();
		drop(data);
		scratch.staging.unmap();
		drop(scratch);

		partials
	}
}

fn create_pipeline(
	device: &wgpu::Device,
	source: &str,
	label: &str,
) -> wgpu::ComputePipeline {
	let module = device.create_shader_module(
		wgpu::ShaderModuleDescriptor {
			label: Some(label),
			source: wgpu::ShaderSource::Wgsl(source.into()),
		},
	);
	device.create_compute_pipeline(
		&wgpu::ComputePipelineDescriptor {
			label: Some(label),
			layout: None,
			module: &module,
			entry_point: Some("main"),
			compilation_options:
				wgpu::PipelineCompilationOptions::default(),
			cache: None,
		},
	)
}

fn create_reduce_partial(
	device: &wgpu::Device,
	capacity: u32,
) -> wgpu::Buffer {
	device.create_buffer(&wgpu::BufferDescriptor {
		label: Some("reduce_partial"),
		size: u64::from(capacity) * 4,
		usage: wgpu::BufferUsages::STORAGE
			| wgpu::BufferUsages::COPY_SRC,
		mapped_at_creation: false,
	})
}

fn create_reduce_staging(
	device: &wgpu::Device,
	capacity: u32,
) -> wgpu::Buffer {
	device.create_buffer(&wgpu::BufferDescriptor {
		label: Some("reduce_staging"),
		size: u64::from(capacity) * 4,
		usage: wgpu::BufferUsages::MAP_READ
			| wgpu::BufferUsages::COPY_DST,
		mapped_at_creation: false,
	})
}

/// Kahan-compensated summation over f32 partials.
fn kahan_sum(partials: &[f32]) -> f64 {
	let mut sum = 0.0_f64;
	let mut comp = 0.0_f64;
	for &v in partials {
		let y = f64::from(v) - comp;
		let t = sum + y;
		comp = (t - sum) - y;
		sum = t;
	}
	sum
}

impl Backend for WgpuBackend {
	type Buffer = WgpuBuffer;

	#[allow(clippy::unnecessary_literal_bound)]
	fn name(&self) -> &str {
		"wgpu"
	}

	fn allocate(
		&self,
		len: usize,
		dtype: DType,
	) -> Result<WgpuBuffer> {
		let buf =
			self.ctx
				.device
				.create_buffer(&wgpu::BufferDescriptor {
					label: None,
					size: (len as u64) * 4,
					usage: wgpu::BufferUsages::STORAGE
						| wgpu::BufferUsages::COPY_SRC
						| wgpu::BufferUsages::COPY_DST,
					mapped_at_creation: true,
				});
		buf.unmap();
		Ok(WgpuBuffer::new(
			Arc::clone(&self.ctx.device),
			Arc::clone(&self.ctx.queue),
			buf,
			len,
			dtype,
		))
	}

	fn upload_f64(
		&self,
		data: &[f64],
	) -> Result<WgpuBuffer> {
		let f32_data: Vec<f32> =
			data.iter().map(|&v| v as f32).collect();
		Ok(self.make_buffer(&f32_data, DType::F64))
	}

	fn upload_f32(
		&self,
		data: &[f32],
	) -> Result<WgpuBuffer> {
		Ok(self.make_buffer(data, DType::F32))
	}

	#[allow(clippy::similar_names)]
	#[inline]
	fn apply_stencil(
		&self,
		input: &WgpuBuffer,
		output: &mut WgpuBuffer,
		grid: &Grid,
		_stencil: &Stencil,
		_boundaries: &Boundaries,
	) -> Result<()> {
		let n = grid.len();
		if input.len() != n {
			return Err(CoreError::DimensionMismatch {
				expected: n,
				got: input.len(),
			});
		}
		if output.len() != n {
			return Err(CoreError::DimensionMismatch {
				expected: n,
				got: output.len(),
			});
		}

		let inv_dx2 = (1.0 / (grid.dx * grid.dx)) as f32;
		let inv_dy2 = (1.0 / (grid.dy * grid.dy)) as f32;
		let mut params = [0u8; 16];
		params[0..4].copy_from_slice(
			&(grid.rows as u32).to_le_bytes(),
		);
		params[4..8].copy_from_slice(
			&(grid.cols as u32).to_le_bytes(),
		);
		params[8..12]
			.copy_from_slice(&inv_dx2.to_le_bytes());
		params[12..16]
			.copy_from_slice(&inv_dy2.to_le_bytes());
		self.write_uniform(&params);

		let bind_group = self.get_or_create_bg2(
			&self.bg_stencil,
			&self.stencil_pipeline,
			input,
			output,
		);

		self.dispatch(
			&self.stencil_pipeline,
			&bind_group,
			(
				(grid.cols as u32).div_ceil(16),
				(grid.rows as u32).div_ceil(8),
				1,
			),
		);
		Ok(())
	}

	fn apply_stencil_var(
		&self,
		input: &WgpuBuffer,
		output: &mut WgpuBuffer,
		coeff: &WgpuBuffer,
		grid: &Grid,
		stencil: &Stencil,
		boundaries: &Boundaries,
	) -> Result<()> {
		// Fallback: apply_stencil then pointwise_mult.
		self.apply_stencil(
			input, output, grid, stencil, boundaries,
		)?;
		// output = output * coeff (in-place via copy).
		// pointwise_mult(x, y, z) writes z = x*y. We
		// need z=output, but we can't alias. Use readback.
		let n = output.len();
		let mut out_data = vec![0.0_f64; n];
		output.copy_to_host_f64(&mut out_data);
		let mut coeff_data = vec![0.0_f64; n];
		coeff.copy_to_host_f64(&mut coeff_data);
		for i in 0..n {
			out_data[i] *= coeff_data[i];
		}
		// Re-upload via a temporary buffer and copy.
		let tmp = self.upload_f64(&out_data)?;
		self.copy(&tmp, output)?;
		Ok(())
	}

	#[inline]
	fn fill(
		&self,
		buf: &mut WgpuBuffer,
		value: f64,
	) -> Result<()> {
		let len = buf.len() as u32;
		let mut params = [0u8; 16];
		params[0..4].copy_from_slice(
			&(value as f32).to_le_bytes(),
		);
		params[4..8]
			.copy_from_slice(&len.to_le_bytes());
		self.write_uniform(&params);

		let bind_group = self.get_or_create_bg1(
			&self.bg_fill,
			&self.fill_pipeline,
			buf,
		);

		self.dispatch(
			&self.fill_pipeline,
			&bind_group,
			(len.div_ceil(256), 1, 1),
		);
		Ok(())
	}

	#[inline]
	fn axpy(
		&self,
		alpha: f64,
		x: &WgpuBuffer,
		y: &mut WgpuBuffer,
	) -> Result<()> {
		if x.len() != y.len() {
			return Err(CoreError::DimensionMismatch {
				expected: x.len(),
				got: y.len(),
			});
		}
		let len = x.len() as u32;

		let mut params = [0u8; 16];
		params[0..4].copy_from_slice(
			&(alpha as f32).to_le_bytes(),
		);
		params[4..8]
			.copy_from_slice(&len.to_le_bytes());
		self.write_uniform(&params);

		let bind_group = self.get_or_create_bg2(
			&self.bg_axpy,
			&self.axpy_pipeline,
			x,
			y,
		);

		self.dispatch(
			&self.axpy_pipeline,
			&bind_group,
			(len.div_ceil(256), 1, 1),
		);
		Ok(())
	}

	#[inline]
	fn scale(
		&self,
		buf: &mut WgpuBuffer,
		alpha: f64,
	) -> Result<()> {
		let len = buf.len() as u32;
		let mut params = [0u8; 16];
		params[0..4].copy_from_slice(
			&(alpha as f32).to_le_bytes(),
		);
		params[4..8]
			.copy_from_slice(&len.to_le_bytes());
		self.write_uniform(&params);

		let bind_group = self.get_or_create_bg1(
			&self.bg_scale,
			&self.scale_pipeline,
			buf,
		);

		self.dispatch(
			&self.scale_pipeline,
			&bind_group,
			(len.div_ceil(256), 1, 1),
		);
		Ok(())
	}

	#[inline]
	fn copy(
		&self,
		src: &WgpuBuffer,
		dst: &mut WgpuBuffer,
	) -> Result<()> {
		if src.len() != dst.len() {
			return Err(CoreError::DimensionMismatch {
				expected: src.len(),
				got: dst.len(),
			});
		}
		let mut encoder =
			self.ctx.device.create_command_encoder(
				&wgpu::CommandEncoderDescriptor::default(),
			);
		encoder.copy_buffer_to_buffer(
			&src.buf,
			0,
			&dst.buf,
			0,
			src.size_bytes(),
		);
		self.ctx
			.queue
			.submit(std::iter::once(encoder.finish()));
		Ok(())
	}

	#[inline]
	fn norm_l2(&self, buf: &WgpuBuffer) -> Result<f64> {
		let partials = self.run_reduction(
			&self.reduce_sum_sq_pipeline,
			buf,
		);
		Ok(kahan_sum(&partials).sqrt())
	}

	fn reduce_sum(&self, buf: &WgpuBuffer) -> Result<f64> {
		let partials = self.run_reduction(
			&self.reduce_sum_pipeline,
			buf,
		);
		Ok(kahan_sum(&partials))
	}

	fn reduce_max(&self, buf: &WgpuBuffer) -> Result<f64> {
		let partials = self.run_reduction(
			&self.reduce_max_pipeline,
			buf,
		);
		Ok(partials
			.iter()
			.copied()
			.map(f64::from)
			.reduce(f64::max)
			.unwrap_or(f64::NEG_INFINITY))
	}

	fn reduce_min(&self, buf: &WgpuBuffer) -> Result<f64> {
		let partials = self.run_reduction(
			&self.reduce_min_pipeline,
			buf,
		);
		Ok(partials
			.iter()
			.copied()
			.map(f64::from)
			.reduce(f64::min)
			.unwrap_or(f64::INFINITY))
	}

	#[allow(clippy::similar_names)]
	#[inline]
	fn stencil_axpy(
		&self,
		alpha: f64,
		x: &WgpuBuffer,
		y: &mut WgpuBuffer,
		grid: &Grid,
		_stencil: &Stencil,
		_boundaries: &Boundaries,
	) -> Result<()> {
		let n = grid.len();
		if x.len() != n {
			return Err(CoreError::DimensionMismatch {
				expected: n,
				got: x.len(),
			});
		}
		if y.len() != n {
			return Err(CoreError::DimensionMismatch {
				expected: n,
				got: y.len(),
			});
		}

		let inv_dx2 = (1.0 / (grid.dx * grid.dx)) as f32;
		let inv_dy2 = (1.0 / (grid.dy * grid.dy)) as f32;
		// 24 bytes: rows(u32), cols(u32), inv_dx2(f32),
		//           inv_dy2(f32), alpha(f32), pad(u32)
		let mut params = [0u8; 24];
		params[0..4].copy_from_slice(
			&(grid.rows as u32).to_le_bytes(),
		);
		params[4..8].copy_from_slice(
			&(grid.cols as u32).to_le_bytes(),
		);
		params[8..12]
			.copy_from_slice(&inv_dx2.to_le_bytes());
		params[12..16]
			.copy_from_slice(&inv_dy2.to_le_bytes());
		params[16..20].copy_from_slice(
			&(alpha as f32).to_le_bytes(),
		);
		params[20..24].copy_from_slice(&0u32.to_le_bytes());
		self.write_uniform(&params);

		let bind_group = self.get_or_create_bg2(
			&self.bg_fused,
			&self.fused_stencil_axpy_pipeline,
			x,
			y,
		);

		self.dispatch(
			&self.fused_stencil_axpy_pipeline,
			&bind_group,
			(
				(grid.cols as u32).div_ceil(16),
				(grid.rows as u32).div_ceil(8),
				1,
			),
		);
		Ok(())
	}

	#[inline]
	fn dot(
		&self,
		x: &WgpuBuffer,
		y: &WgpuBuffer,
	) -> Result<f64> {
		if x.len() != y.len() {
			return Err(CoreError::DimensionMismatch {
				expected: x.len(),
				got: y.len(),
			});
		}
		let len = x.len() as u32;
		let n_groups = len.div_ceil(256);

		let mut params = [0u8; 24];
		params[0..4]
			.copy_from_slice(&len.to_le_bytes());
		self.write_uniform(&params);

		self.ensure_reduce_capacity(n_groups);
		let scratch = self.reduce_scratch.read().expect(
			"reduce scratch lock",
		);

		let layout =
			self.dot_pipeline.get_bind_group_layout(0);
		let bind_group =
			self.ctx.device.create_bind_group(
				&wgpu::BindGroupDescriptor {
					label: None,
					layout: &layout,
					entries: &[
						wgpu::BindGroupEntry {
							binding: 0,
							resource: self
								.uniform_buf
								.as_entire_binding(),
						},
						wgpu::BindGroupEntry {
							binding: 1,
							resource: x
								.buf
								.as_entire_binding(),
						},
						wgpu::BindGroupEntry {
							binding: 2,
							resource: y
								.buf
								.as_entire_binding(),
						},
						wgpu::BindGroupEntry {
							binding: 3,
							resource: scratch
								.partial
								.as_entire_binding(),
						},
					],
				},
			);

		let staging_size = u64::from(n_groups) * 4;
		let mut encoder =
			self.ctx.device.create_command_encoder(
				&wgpu::CommandEncoderDescriptor::default(),
			);
		{
			let mut pass = encoder.begin_compute_pass(
				&wgpu::ComputePassDescriptor::default(),
			);
			pass.set_pipeline(&self.dot_pipeline);
			pass.set_bind_group(0, &bind_group, &[]);
			pass.dispatch_workgroups(n_groups, 1, 1);
		}
		encoder.copy_buffer_to_buffer(
			&scratch.partial,
			0,
			&scratch.staging,
			0,
			staging_size,
		);
		self.ctx
			.queue
			.submit(std::iter::once(encoder.finish()));

		let slice =
			scratch.staging.slice(..staging_size);
		slice.map_async(wgpu::MapMode::Read, |_| {});
		self.ctx
			.device
			.poll(wgpu::PollType::wait_indefinitely())
			.expect("device poll failed");

		let data = slice.get_mapped_range();
		let partials: &[f32] =
			bytemuck::cast_slice(&data);
		let result = kahan_sum(partials);
		drop(data);
		scratch.staging.unmap();
		drop(scratch);

		Ok(result)
	}

	fn restrict(
		&self,
		fine: &WgpuBuffer,
		coarse: &mut WgpuBuffer,
		fine_grid: &Grid,
		coarse_grid: &Grid,
	) -> Result<()> {
		let mut params = [0u8; 16];
		params[0..4].copy_from_slice(
			&(fine_grid.rows as u32).to_le_bytes(),
		);
		params[4..8].copy_from_slice(
			&(fine_grid.cols as u32).to_le_bytes(),
		);
		params[8..12].copy_from_slice(
			&(coarse_grid.rows as u32).to_le_bytes(),
		);
		params[12..16].copy_from_slice(
			&(coarse_grid.cols as u32).to_le_bytes(),
		);
		self.write_uniform(&params);

		let layout =
			self.restrict_pipeline.get_bind_group_layout(0);
		let bg = self.ctx.device.create_bind_group(
			&wgpu::BindGroupDescriptor {
				label: None,
				layout: &layout,
				entries: &[
					wgpu::BindGroupEntry {
						binding: 0,
						resource: self
							.uniform_buf
							.as_entire_binding(),
					},
					wgpu::BindGroupEntry {
						binding: 1,
						resource: fine
							.buf
							.as_entire_binding(),
					},
					wgpu::BindGroupEntry {
						binding: 2,
						resource: coarse
							.buf
							.as_entire_binding(),
					},
				],
			},
		);

		self.dispatch(
			&self.restrict_pipeline,
			&bg,
			(
				(coarse_grid.cols as u32).div_ceil(16),
				(coarse_grid.rows as u32).div_ceil(8),
				1,
			),
		);
		Ok(())
	}

	fn prolong(
		&self,
		coarse: &WgpuBuffer,
		fine: &mut WgpuBuffer,
		coarse_grid: &Grid,
		fine_grid: &Grid,
	) -> Result<()> {
		let mut params = [0u8; 16];
		params[0..4].copy_from_slice(
			&(fine_grid.rows as u32).to_le_bytes(),
		);
		params[4..8].copy_from_slice(
			&(fine_grid.cols as u32).to_le_bytes(),
		);
		params[8..12].copy_from_slice(
			&(coarse_grid.cols as u32).to_le_bytes(),
		);
		self.write_uniform(&params);

		let layout =
			self.prolong_pipeline.get_bind_group_layout(0);
		let bg = self.ctx.device.create_bind_group(
			&wgpu::BindGroupDescriptor {
				label: None,
				layout: &layout,
				entries: &[
					wgpu::BindGroupEntry {
						binding: 0,
						resource: self
							.uniform_buf
							.as_entire_binding(),
					},
					wgpu::BindGroupEntry {
						binding: 1,
						resource: coarse
							.buf
							.as_entire_binding(),
					},
					wgpu::BindGroupEntry {
						binding: 2,
						resource: fine
							.buf
							.as_entire_binding(),
					},
				],
			},
		);

		self.dispatch(
			&self.prolong_pipeline,
			&bg,
			(
				(fine_grid.cols as u32).div_ceil(16),
				(fine_grid.rows as u32).div_ceil(8),
				1,
			),
		);
		Ok(())
	}

	#[allow(clippy::similar_names)]
	fn weighted_jacobi(
		&self,
		x: &mut WgpuBuffer,
		b: &WgpuBuffer,
		omega: f64,
		grid: &Grid,
		_stencil: &Stencil,
		_boundaries: &Boundaries,
	) -> Result<()> {
		let inv_dx2 = (1.0 / (grid.dx * grid.dx)) as f32;
		let inv_dy2 = (1.0 / (grid.dy * grid.dy)) as f32;
		let mut params = [0u8; 24];
		params[0..4].copy_from_slice(
			&(grid.rows as u32).to_le_bytes(),
		);
		params[4..8].copy_from_slice(
			&(grid.cols as u32).to_le_bytes(),
		);
		params[8..12]
			.copy_from_slice(&inv_dx2.to_le_bytes());
		params[12..16]
			.copy_from_slice(&inv_dy2.to_le_bytes());
		params[16..20].copy_from_slice(
			&(omega as f32).to_le_bytes(),
		);
		params[20..24]
			.copy_from_slice(&0u32.to_le_bytes());
		self.write_uniform(&params);

		let layout =
			self.jacobi_pipeline.get_bind_group_layout(0);
		let bg = self.ctx.device.create_bind_group(
			&wgpu::BindGroupDescriptor {
				label: None,
				layout: &layout,
				entries: &[
					wgpu::BindGroupEntry {
						binding: 0,
						resource: self
							.uniform_buf
							.as_entire_binding(),
					},
					wgpu::BindGroupEntry {
						binding: 1,
						resource: b
							.buf
							.as_entire_binding(),
					},
					wgpu::BindGroupEntry {
						binding: 2,
						resource: x
							.buf
							.as_entire_binding(),
					},
				],
			},
		);

		self.dispatch(
			&self.jacobi_pipeline,
			&bg,
			(
				(grid.cols as u32).div_ceil(16),
				(grid.rows as u32).div_ceil(8),
				1,
			),
		);
		Ok(())
	}

	// ── Extended vector operations ───────────────────
	// Composed from existing ops or host-side fallback.
	// wgpu is f32-limited — HIP is the perf target.

	fn pointwise_mult(
		&self,
		x: &WgpuBuffer,
		y: &WgpuBuffer,
		z: &mut WgpuBuffer,
	) -> Result<()> {
		let xs = x.read_f32();
		let ys = y.read_f32();
		let mut zs = vec![0.0_f32; xs.len()];
		for i in 0..xs.len() {
			zs[i] = xs[i] * ys[i];
		}
		self.ctx
			.queue
			.write_buffer(&z.buf, 0, bytemuck::cast_slice(&zs));
		Ok(())
	}

	fn pointwise_div(
		&self,
		x: &WgpuBuffer,
		y: &WgpuBuffer,
		z: &mut WgpuBuffer,
	) -> Result<()> {
		let xs = x.read_f32();
		let ys = y.read_f32();
		let mut zs = vec![0.0_f32; xs.len()];
		for i in 0..xs.len() {
			zs[i] = xs[i] / ys[i];
		}
		self.ctx
			.queue
			.write_buffer(&z.buf, 0, bytemuck::cast_slice(&zs));
		Ok(())
	}

	fn waxpy(
		&self,
		alpha: f64,
		x: &WgpuBuffer,
		beta: f64,
		y: &WgpuBuffer,
		w: &mut WgpuBuffer,
	) -> Result<()> {
		// w = alpha*x + beta*y
		// → copy x→w, scale(w, alpha), axpy(beta, y, w)
		self.copy(x, w)?;
		self.scale(w, alpha)?;
		self.axpy(beta, y, w)?;
		Ok(())
	}

	fn aypx(
		&self,
		alpha: f64,
		x: &WgpuBuffer,
		y: &mut WgpuBuffer,
	) -> Result<()> {
		// y = x + alpha*y → scale(y, alpha), axpy(1, x, y)
		self.scale(y, alpha)?;
		self.axpy(1.0, x, y)?;
		Ok(())
	}

	fn reciprocal(
		&self,
		buf: &mut WgpuBuffer,
	) -> Result<()> {
		let mut data = buf.read_f32();
		for v in &mut data {
			*v = 1.0 / *v;
		}
		self.ctx
			.queue
			.write_buffer(
				&buf.buf,
				0,
				bytemuck::cast_slice(&data),
			);
		Ok(())
	}

	fn abs_val(
		&self,
		buf: &mut WgpuBuffer,
	) -> Result<()> {
		let mut data = buf.read_f32();
		for v in &mut data {
			*v = v.abs();
		}
		self.ctx
			.queue
			.write_buffer(
				&buf.buf,
				0,
				bytemuck::cast_slice(&data),
			);
		Ok(())
	}

	fn pointwise_max(
		&self,
		x: &WgpuBuffer,
		y: &WgpuBuffer,
		z: &mut WgpuBuffer,
	) -> Result<()> {
		let xd = x.read_f32();
		let yd = y.read_f32();
		let zd: Vec<f32> = xd
			.iter()
			.zip(yd.iter())
			.map(|(&a, &b)| a.max(b))
			.collect();
		self.ctx.queue.write_buffer(
			&z.buf,
			0,
			bytemuck::cast_slice(&zd),
		);
		Ok(())
	}

	fn pointwise_min(
		&self,
		x: &WgpuBuffer,
		y: &WgpuBuffer,
		z: &mut WgpuBuffer,
	) -> Result<()> {
		let xd = x.read_f32();
		let yd = y.read_f32();
		let zd: Vec<f32> = xd
			.iter()
			.zip(yd.iter())
			.map(|(&a, &b)| a.min(b))
			.collect();
		self.ctx.queue.write_buffer(
			&z.buf,
			0,
			bytemuck::cast_slice(&zd),
		);
		Ok(())
	}

	#[allow(clippy::too_many_arguments)]
	fn apply_conv_diff(
		&self,
		u: &WgpuBuffer,
		output: &mut WgpuBuffer,
		kappa: &WgpuBuffer,
		vx: &WgpuBuffer,
		vy: &WgpuBuffer,
		grid: &Grid,
		_boundaries: &Boundaries,
	) -> Result<()> {
		let rows = grid.rows;
		let cols = grid.cols;
		let inv_dx = (1.0 / grid.dx) as f32;
		let inv_dy = (1.0 / grid.dy) as f32;
		let inv_dx2 = inv_dx * inv_dx;
		let inv_dy2 = inv_dy * inv_dy;
		let src = u.read_f32();
		let k = kappa.read_f32();
		let vxs = vx.read_f32();
		let vys = vy.read_f32();
		let mut dst = vec![0.0_f32; src.len()];
		for row in 1..rows - 1 {
			for col in 1..cols - 1 {
				let i = row * cols + col;
				let c = src[i];
				let lap = (src[i - 1] + src[i + 1]) * inv_dx2
					+ (src[i - cols] + src[i + cols]) * inv_dy2
					- 2.0 * (inv_dx2 + inv_dy2) * c;
				let vxi = vxs[i];
				let vyi = vys[i];
				let dudx = if vxi >= 0.0 {
					(c - src[i - 1]) * inv_dx
				} else {
					(src[i + 1] - c) * inv_dx
				};
				let dudy = if vyi >= 0.0 {
					(c - src[i - cols]) * inv_dy
				} else {
					(src[i + cols] - c) * inv_dy
				};
				dst[i] = k[i] * lap + vxi * dudx + vyi * dudy;
			}
		}
		self.ctx.queue.write_buffer(
			&output.buf,
			0,
			bytemuck::cast_slice(&dst),
		);
		Ok(())
	}

	#[allow(clippy::too_many_arguments)]
	fn conv_diff_axpy(
		&self,
		alpha: f64,
		u: &WgpuBuffer,
		output: &mut WgpuBuffer,
		kappa: &WgpuBuffer,
		vx: &WgpuBuffer,
		vy: &WgpuBuffer,
		grid: &Grid,
		_boundaries: &Boundaries,
	) -> Result<()> {
		let a = alpha as f32;
		let rows = grid.rows;
		let cols = grid.cols;
		let inv_dx = (1.0 / grid.dx) as f32;
		let inv_dy = (1.0 / grid.dy) as f32;
		let inv_dx2 = inv_dx * inv_dx;
		let inv_dy2 = inv_dy * inv_dy;
		let src = u.read_f32();
		let k = kappa.read_f32();
		let vxs = vx.read_f32();
		let vys = vy.read_f32();
		let mut dst = output.read_f32();
		for row in 1..rows - 1 {
			for col in 1..cols - 1 {
				let i = row * cols + col;
				let c = src[i];
				let lap = (src[i - 1] + src[i + 1]) * inv_dx2
					+ (src[i - cols] + src[i + cols]) * inv_dy2
					- 2.0 * (inv_dx2 + inv_dy2) * c;
				let vxi = vxs[i];
				let vyi = vys[i];
				let dudx = if vxi >= 0.0 {
					(c - src[i - 1]) * inv_dx
				} else {
					(src[i + 1] - c) * inv_dx
				};
				let dudy = if vyi >= 0.0 {
					(c - src[i - cols]) * inv_dy
				} else {
					(src[i + cols] - c) * inv_dy
				};
				dst[i] += a * (k[i] * lap + vxi * dudx + vyi * dudy);
			}
		}
		self.ctx.queue.write_buffer(
			&output.buf,
			0,
			bytemuck::cast_slice(&dst),
		);
		Ok(())
	}
}
