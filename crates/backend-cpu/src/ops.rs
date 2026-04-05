use fluxion_core::{
	Backend, BackendBuffer, Boundaries, BoundaryCondition,
	CoreError, DType, Grid, Result, Stencil,
};
use rayon::prelude::*;

use crate::buffer::CpuBuffer;

/// Returns an error if any boundary is `Periodic` (not yet
/// implemented — requires index wrapping in the stencil kernel).
fn reject_periodic(bc: &Boundaries) -> Result<()> {
	let sides = [bc.top, bc.bottom, bc.left, bc.right];
	if sides
		.iter()
		.any(|s| matches!(s, BoundaryCondition::Periodic))
	{
		return Err(CoreError::BackendError(
			"periodic boundaries are not yet implemented"
				.into(),
		));
	}
	Ok(())
}

/// CPU reference backend.
///
/// All operations run with tight loops, parallelised via rayon
/// where beneficial. Intended as correctness reference and
/// high-performance multi-core fallback.
#[derive(Debug)]
pub struct CpuBackend;

impl CpuBackend {
	#[must_use]
	pub const fn new() -> Self {
		Self
	}
}

impl Default for CpuBackend {
	fn default() -> Self {
		Self::new()
	}
}

impl Backend for CpuBackend {
	type Buffer = CpuBuffer;

	#[allow(clippy::unnecessary_literal_bound)]
	fn name(&self) -> &str {
		"cpu"
	}

	fn allocate(
		&self,
		len: usize,
		dtype: DType,
	) -> Result<CpuBuffer> {
		Ok(CpuBuffer::zeros(len, dtype))
	}

	fn upload_f64(
		&self,
		data: &[f64],
	) -> Result<CpuBuffer> {
		Ok(CpuBuffer::from_f64(data))
	}

	fn upload_f32(
		&self,
		data: &[f32],
	) -> Result<CpuBuffer> {
		Ok(CpuBuffer::from_f32(data))
	}

	#[inline]
	fn apply_stencil(
		&self,
		input: &CpuBuffer,
		output: &mut CpuBuffer,
		grid: &Grid,
		stencil: &Stencil,
		boundaries: &Boundaries,
	) -> Result<()> {
		reject_periodic(boundaries)?;
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

		let src = input.as_slice();
		let dst = output.as_mut_slice();
		let rows = grid.rows;
		let cols = grid.cols;
		let entries = stencil.entries();

		// ── Pass 1: Boundaries (sequential, small) ──
		apply_boundaries(
			src, dst, rows, cols, grid.dx, grid.dy, boundaries,
		);

		// ── Pass 2: Interior (parallel, SIMD-friendly) ──
		// Rows 1..rows-1, cols 1..cols-1.
		// Each row is independent → parallelise over rows.
		let interior_dst =
			&mut dst[cols..(rows - 1) * cols];
		interior_dst
			.par_chunks_mut(cols)
			.enumerate()
			.for_each(|(ri, row_dst)| {
				let row = ri + 1; // actual row index
				apply_interior_row(
					src, row_dst, row, cols, entries,
				);
			});

		Ok(())
	}

	fn apply_stencil_var(
		&self,
		input: &CpuBuffer,
		output: &mut CpuBuffer,
		coeff: &CpuBuffer,
		grid: &Grid,
		stencil: &Stencil,
		boundaries: &Boundaries,
	) -> Result<()> {
		// Apply standard stencil, then pointwise multiply
		// by coefficient field.
		self.apply_stencil(
			input, output, grid, stencil, boundaries,
		)?;
		let cs = coeff.as_slice();
		let os = output.as_mut_slice();
		os.par_chunks_mut(1024)
			.enumerate()
			.for_each(|(ci, chunk)| {
				let base = ci * 1024;
				for (j, v) in chunk.iter_mut().enumerate() {
					*v *= cs[base + j];
				}
			});
		Ok(())
	}

	#[inline]
	fn copy(
		&self,
		src: &CpuBuffer,
		dst: &mut CpuBuffer,
	) -> Result<()> {
		if src.len() != dst.len() {
			return Err(CoreError::DimensionMismatch {
				expected: src.len(),
				got: dst.len(),
			});
		}
		dst.as_mut_slice()
			.copy_from_slice(src.as_slice());
		Ok(())
	}

	#[inline]
	fn fill(
		&self,
		buf: &mut CpuBuffer,
		value: f64,
	) -> Result<()> {
		buf.as_mut_slice().fill(value);
		Ok(())
	}

	#[inline]
	fn axpy(
		&self,
		alpha: f64,
		x: &CpuBuffer,
		y: &mut CpuBuffer,
	) -> Result<()> {
		if x.len() != y.len() {
			return Err(CoreError::DimensionMismatch {
				expected: x.len(),
				got: y.len(),
			});
		}
		let xs = x.as_slice();
		let ys = y.as_mut_slice();
		ys.par_chunks_mut(1024)
			.enumerate()
			.for_each(|(ci, chunk)| {
				let base = ci * 1024;
				for (j, v) in chunk.iter_mut().enumerate() {
					*v = alpha.mul_add(xs[base + j], *v);
				}
			});
		Ok(())
	}

	#[inline]
	fn scale(
		&self,
		buf: &mut CpuBuffer,
		alpha: f64,
	) -> Result<()> {
		buf.as_mut_slice()
			.par_chunks_mut(1024)
			.for_each(|chunk| {
				for v in chunk {
					*v *= alpha;
				}
			});
		Ok(())
	}

	#[inline]
	fn stencil_axpy(
		&self,
		alpha: f64,
		x: &CpuBuffer,
		y: &mut CpuBuffer,
		grid: &Grid,
		stencil: &Stencil,
		_boundaries: &Boundaries,
	) -> Result<()> {
		reject_periodic(_boundaries)?;
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
		let src = x.as_slice();
		let dst = y.as_mut_slice();
		let rows = grid.rows;
		let cols = grid.cols;
		let entries = stencil.entries();

		// Interior only — boundary points in y are untouched.
		let interior_dst =
			&mut dst[cols..(rows - 1) * cols];
		interior_dst
			.par_chunks_mut(cols)
			.enumerate()
			.for_each(|(ri, row_dst)| {
				let row = ri + 1;
				fused_5pt_axpy_row(
					alpha, src, row_dst, row, cols, entries,
				);
			});

		Ok(())
	}

	#[inline]
	fn norm_l2(&self, buf: &CpuBuffer) -> Result<f64> {
		let s = buf.as_slice();
		let (sum, _) = s
			.par_chunks(4096)
			.map(|chunk| {
				let mut acc = 0.0_f64;
				let mut comp = 0.0_f64;
				for &v in chunk {
					let y = v.mul_add(v, -comp);
					let t = acc + y;
					comp = (t - acc) - y;
					acc = t;
				}
				(acc, comp)
			})
			.reduce(
				|| (0.0, 0.0),
				kahan_merge,
			);
		Ok(sum.sqrt())
	}

	#[inline]
	fn reduce_sum(&self, buf: &CpuBuffer) -> Result<f64> {
		let s = buf.as_slice();
		let (sum, _) = s
			.par_chunks(4096)
			.map(|chunk| {
				let mut acc = 0.0_f64;
				let mut comp = 0.0_f64;
				for &v in chunk {
					let y = v - comp;
					let t = acc + y;
					comp = (t - acc) - y;
					acc = t;
				}
				(acc, comp)
			})
			.reduce(
				|| (0.0, 0.0),
				kahan_merge,
			);
		Ok(sum)
	}

	#[inline]
	fn reduce_max(&self, buf: &CpuBuffer) -> Result<f64> {
		let s = buf.as_slice();
		let max = s
			.par_chunks(4096)
			.map(|chunk| {
				chunk
					.iter()
					.copied()
					.reduce(f64::max)
					.unwrap_or(f64::NEG_INFINITY)
			})
			.reduce(|| f64::NEG_INFINITY, f64::max);
		Ok(max)
	}

	#[inline]
	fn dot(
		&self,
		x: &CpuBuffer,
		y: &CpuBuffer,
	) -> Result<f64> {
		if x.len() != y.len() {
			return Err(CoreError::DimensionMismatch {
				expected: x.len(),
				got: y.len(),
			});
		}
		let xs = x.as_slice();
		let ys = y.as_slice();
		let (sum, _) = xs
			.par_chunks(4096)
			.zip(ys.par_chunks(4096))
			.map(|(xc, yc)| {
				let mut acc = 0.0_f64;
				let mut comp = 0.0_f64;
				for i in 0..xc.len() {
					let prod = xc[i] * yc[i];
					let t_y = prod - comp;
					let t = acc + t_y;
					comp = (t - acc) - t_y;
					acc = t;
				}
				(acc, comp)
			})
			.reduce(
				|| (0.0, 0.0),
				kahan_merge,
			);
		Ok(sum)
	}

	fn restrict(
		&self,
		fine: &CpuBuffer,
		coarse: &mut CpuBuffer,
		fine_grid: &Grid,
		coarse_grid: &Grid,
	) -> Result<()> {
		let fs = fine.as_slice();
		let cs = coarse.as_mut_slice();
		let fc = fine_grid.cols;
		let cc = coarse_grid.cols;
		let fr_max = fine_grid.rows - 1;
		let fc_max = fc - 1;

		cs.par_chunks_mut(cc)
			.enumerate()
			.for_each(|(row, row_dst)| {
				for col in 0..cc {
					let fr = row * 2;
					let fcc = col * 2;
					let fr1 = (fr + 1).min(fr_max);
					let fc1 = (fcc + 1).min(fc_max);
					row_dst[col] = 0.25
						* (fs[fr * fc + fcc]
							+ fs[fr1 * fc + fcc]
							+ fs[fr * fc + fc1]
							+ fs[fr1 * fc + fc1]);
				}
			});
		Ok(())
	}

	fn prolong(
		&self,
		coarse: &CpuBuffer,
		fine: &mut CpuBuffer,
		coarse_grid: &Grid,
		fine_grid: &Grid,
	) -> Result<()> {
		let cs = coarse.as_slice();
		let fs = fine.as_mut_slice();
		let fc = fine_grid.cols;
		let cc = coarse_grid.cols;

		fs.par_chunks_mut(fc)
			.enumerate()
			.for_each(|(row, row_dst)| {
				let cr = row / 2;
				for col in 0..fc {
					let ccc = col / 2;
					row_dst[col] += cs[cr * cc + ccc];
				}
			});
		Ok(())
	}

	fn weighted_jacobi(
		&self,
		x: &mut CpuBuffer,
		b: &CpuBuffer,
		omega: f64,
		grid: &Grid,
		stencil: &Stencil,
		_boundaries: &Boundaries,
	) -> Result<()> {
		let rows = grid.rows;
		let cols = grid.cols;
		let bs = b.as_slice();
		let xs = x.as_mut_slice();
		let entries = stencil.entries();

		// Extract diagonal coefficient.
		let diag = entries
			.iter()
			.find(|e| e.dr == 0 && e.dc == 0)
			.map_or(-4.0, |e| e.weight);
		let inv_diag = 1.0 / diag;

		for row in 1..rows - 1 {
			for col in 1..cols - 1 {
				let idx = row * cols + col;
				// Compute A·x at this point.
				let mut ax = 0.0_f64;
				#[allow(
					clippy::cast_possible_wrap,
					clippy::cast_sign_loss
				)]
				for e in entries {
					let r =
						(row as i32 + e.dr) as usize;
					let c =
						(col as i32 + e.dc) as usize;
					ax = e
						.weight
						.mul_add(xs[r * cols + c], ax);
				}
				let residual = bs[idx] - ax;
				xs[idx] = omega.mul_add(
					inv_diag * residual,
					xs[idx],
				);
			}
		}
		Ok(())
	}

	#[inline]
	fn pointwise_mult(
		&self,
		x: &CpuBuffer,
		y: &CpuBuffer,
		z: &mut CpuBuffer,
	) -> Result<()> {
		if x.len() != y.len() || x.len() != z.len() {
			return Err(CoreError::DimensionMismatch {
				expected: x.len(),
				got: y.len(),
			});
		}
		let xs = x.as_slice();
		let ys = y.as_slice();
		let zs = z.as_mut_slice();
		zs.par_chunks_mut(1024)
			.enumerate()
			.for_each(|(ci, chunk)| {
				let b = ci * 1024;
				for (j, v) in chunk.iter_mut().enumerate() {
					*v = xs[b + j] * ys[b + j];
				}
			});
		Ok(())
	}

	#[inline]
	fn pointwise_div(
		&self,
		x: &CpuBuffer,
		y: &CpuBuffer,
		z: &mut CpuBuffer,
	) -> Result<()> {
		if x.len() != y.len() || x.len() != z.len() {
			return Err(CoreError::DimensionMismatch {
				expected: x.len(),
				got: y.len(),
			});
		}
		let xs = x.as_slice();
		let ys = y.as_slice();
		let zs = z.as_mut_slice();
		zs.par_chunks_mut(1024)
			.enumerate()
			.for_each(|(ci, chunk)| {
				let b = ci * 1024;
				for (j, v) in chunk.iter_mut().enumerate() {
					*v = xs[b + j] / ys[b + j];
				}
			});
		Ok(())
	}

	#[inline]
	fn waxpy(
		&self,
		alpha: f64,
		x: &CpuBuffer,
		beta: f64,
		y: &CpuBuffer,
		w: &mut CpuBuffer,
	) -> Result<()> {
		if x.len() != y.len() || x.len() != w.len() {
			return Err(CoreError::DimensionMismatch {
				expected: x.len(),
				got: y.len(),
			});
		}
		let xs = x.as_slice();
		let ys = y.as_slice();
		let ws = w.as_mut_slice();
		ws.par_chunks_mut(1024)
			.enumerate()
			.for_each(|(ci, chunk)| {
				let b = ci * 1024;
				for (j, v) in chunk.iter_mut().enumerate() {
					*v = alpha.mul_add(
						xs[b + j],
						beta * ys[b + j],
					);
				}
			});
		Ok(())
	}

	#[inline]
	fn aypx(
		&self,
		alpha: f64,
		x: &CpuBuffer,
		y: &mut CpuBuffer,
	) -> Result<()> {
		if x.len() != y.len() {
			return Err(CoreError::DimensionMismatch {
				expected: x.len(),
				got: y.len(),
			});
		}
		let xs = x.as_slice();
		let ys = y.as_mut_slice();
		ys.par_chunks_mut(1024)
			.enumerate()
			.for_each(|(ci, chunk)| {
				let b = ci * 1024;
				for (j, v) in chunk.iter_mut().enumerate() {
					*v = alpha.mul_add(*v, xs[b + j]);
				}
			});
		Ok(())
	}

	#[inline]
	fn reciprocal(
		&self,
		buf: &mut CpuBuffer,
	) -> Result<()> {
		buf.as_mut_slice()
			.par_chunks_mut(1024)
			.for_each(|chunk| {
				for v in chunk {
					*v = 1.0 / *v;
				}
			});
		Ok(())
	}

	#[inline]
	fn abs_val(
		&self,
		buf: &mut CpuBuffer,
	) -> Result<()> {
		buf.as_mut_slice()
			.par_chunks_mut(1024)
			.for_each(|chunk| {
				for v in chunk {
					*v = v.abs();
				}
			});
		Ok(())
	}

	fn pointwise_max(
		&self,
		x: &CpuBuffer,
		y: &CpuBuffer,
		z: &mut CpuBuffer,
	) -> Result<()> {
		let xs = x.as_slice();
		let ys = y.as_slice();
		let zs = z.as_mut_slice();
		zs.par_chunks_mut(1024)
			.enumerate()
			.for_each(|(ci, chunk)| {
				let b = ci * 1024;
				for (j, v) in chunk.iter_mut().enumerate() {
					*v = xs[b + j].max(ys[b + j]);
				}
			});
		Ok(())
	}

	fn pointwise_min(
		&self,
		x: &CpuBuffer,
		y: &CpuBuffer,
		z: &mut CpuBuffer,
	) -> Result<()> {
		let xs = x.as_slice();
		let ys = y.as_slice();
		let zs = z.as_mut_slice();
		zs.par_chunks_mut(1024)
			.enumerate()
			.for_each(|(ci, chunk)| {
				let b = ci * 1024;
				for (j, v) in chunk.iter_mut().enumerate() {
					*v = xs[b + j].min(ys[b + j]);
				}
			});
		Ok(())
	}

	#[inline]
	fn reduce_min(&self, buf: &CpuBuffer) -> Result<f64> {
		let s = buf.as_slice();
		let min = s
			.par_chunks(4096)
			.map(|chunk| {
				chunk
					.iter()
					.copied()
					.reduce(f64::min)
					.unwrap_or(f64::INFINITY)
			})
			.reduce(|| f64::INFINITY, f64::min);
		Ok(min)
	}

	#[allow(clippy::too_many_arguments)]
	fn apply_conv_diff(
		&self,
		u: &CpuBuffer,
		output: &mut CpuBuffer,
		kappa: &CpuBuffer,
		vx: &CpuBuffer,
		vy: &CpuBuffer,
		grid: &Grid,
		_boundaries: &Boundaries,
	) -> Result<()> {
		let rows = grid.rows;
		let cols = grid.cols;
		let inv_dx = 1.0 / grid.dx;
		let inv_dy = 1.0 / grid.dy;
		let inv_dx2 = inv_dx * inv_dx;
		let inv_dy2 = inv_dy * inv_dy;
		let src = u.as_slice();
		let dst = output.as_mut_slice();
		let k = kappa.as_slice();
		let vxs = vx.as_slice();
		let vys = vy.as_slice();

		// Boundaries = 0 (Dirichlet).
		for col in 0..cols {
			dst[col] = 0.0;
			dst[(rows - 1) * cols + col] = 0.0;
		}
		for row in 1..rows - 1 {
			dst[row * cols] = 0.0;
			dst[row * cols + cols - 1] = 0.0;
		}

		// Interior: κ·Δu + v·∇u (upwind).
		for row in 1..rows - 1 {
			for col in 1..cols - 1 {
				let idx = row * cols + col;
				let c = src[idx];
				let lap = (src[idx - 1] + src[idx + 1])
					.mul_add(inv_dx2, (src[idx - cols] + src[idx + cols]) * inv_dy2)
					- 2.0 * (inv_dx2 + inv_dy2) * c;
				let diff = k[idx] * lap;

				// Upwind convection.
				let vx_i = vxs[idx];
				let vy_i = vys[idx];
				let dudx = if vx_i >= 0.0 {
					(c - src[idx - 1]) * inv_dx
				} else {
					(src[idx + 1] - c) * inv_dx
				};
				let dudy = if vy_i >= 0.0 {
					(c - src[idx - cols]) * inv_dy
				} else {
					(src[idx + cols] - c) * inv_dy
				};
				let conv =
					vx_i.mul_add(dudx, vy_i * dudy);

				dst[idx] = diff + conv;
			}
		}
		Ok(())
	}

	#[allow(clippy::too_many_arguments)]
	fn conv_diff_axpy(
		&self,
		alpha: f64,
		u: &CpuBuffer,
		output: &mut CpuBuffer,
		kappa: &CpuBuffer,
		vx: &CpuBuffer,
		vy: &CpuBuffer,
		grid: &Grid,
		_boundaries: &Boundaries,
	) -> Result<()> {
		let rows = grid.rows;
		let cols = grid.cols;
		let inv_dx = 1.0 / grid.dx;
		let inv_dy = 1.0 / grid.dy;
		let inv_dx2 = inv_dx * inv_dx;
		let inv_dy2 = inv_dy * inv_dy;
		let src = u.as_slice();
		let dst = output.as_mut_slice();
		let k = kappa.as_slice();
		let vxs = vx.as_slice();
		let vys = vy.as_slice();

		for row in 1..rows - 1 {
			for col in 1..cols - 1 {
				let idx = row * cols + col;
				let c = src[idx];
				let lap = (src[idx - 1] + src[idx + 1])
					.mul_add(inv_dx2, (src[idx - cols] + src[idx + cols]) * inv_dy2)
					- 2.0 * (inv_dx2 + inv_dy2) * c;
				let diff = k[idx] * lap;

				let vx_i = vxs[idx];
				let vy_i = vys[idx];
				let dudx = if vx_i >= 0.0 {
					(c - src[idx - 1]) * inv_dx
				} else {
					(src[idx + 1] - c) * inv_dx
				};
				let dudy = if vy_i >= 0.0 {
					(c - src[idx - cols]) * inv_dy
				} else {
					(src[idx + cols] - c) * inv_dy
				};
				let conv =
					vx_i.mul_add(dudx, vy_i * dudy);

				dst[idx] =
					alpha.mul_add(diff + conv, dst[idx]);
			}
		}
		Ok(())
	}
}

// ── Stencil helpers ─────────────────────────────────

/// Apply boundary conditions to all 4 edges.
/// Sequential — boundary cells are O(rows + cols).
#[inline]
fn apply_boundaries(
	src: &[f64],
	dst: &mut [f64],
	rows: usize,
	cols: usize,
	dx: f64,
	dy: f64,
	bc: &Boundaries,
) {
	let last_row = rows - 1;
	let last_col = cols - 1;

	// Top & bottom rows.
	for col in 0..cols {
		dst[col] = bc_value(
			bc.top,
			src[col],
			src[cols + col],
			dy,
		);
		dst[last_row * cols + col] = bc_value(
			bc.bottom,
			src[last_row * cols + col],
			src[(last_row - 1) * cols + col],
			dy,
		);
	}

	// Left & right columns (skip corners, already done).
	for row in 1..last_row {
		let idx = row * cols;
		dst[idx] = bc_value(
			bc.left,
			src[idx],
			src[idx + 1],
			dx,
		);
		dst[idx + last_col] = bc_value(
			bc.right,
			src[idx + last_col],
			src[idx + last_col - 1],
			dx,
		);
	}
}

/// Apply stencil to one interior row (cols 1..cols-1).
///
/// Separated from boundaries so the compiler can
/// auto-vectorise this tight loop without branches.
#[inline]
fn apply_interior_row(
	src: &[f64],
	row_dst: &mut [f64],
	row: usize,
	cols: usize,
	entries: &[fluxion_core::StencilEntry],
) {
	// Fast path: unrolled 5-point Laplacian (most common case).
	if entries.len() == 5 {
		apply_5pt_row(src, row_dst, row, cols, entries);
		return;
	}

	// Fast path: unrolled 9-point compact Laplacian.
	if entries.len() == 9 {
		apply_9pt_row(src, row_dst, row, cols, entries);
		return;
	}

	// Generic stencil path.
	#[allow(
		clippy::cast_possible_wrap,
		clippy::cast_sign_loss,
		clippy::needless_range_loop,
	)]
	for col in 1..cols - 1 {
		let mut acc = 0.0_f64;
		for e in entries {
			let r = (row as i32 + e.dr) as usize;
			let c = (col as i32 + e.dc) as usize;
			acc = e.weight.mul_add(src[r * cols + c], acc);
		}
		row_dst[col] = acc;
	}
}

/// Unrolled 5-point Laplacian for one row.
///
/// Avoids the inner loop over entries, allowing the compiler
/// to auto-vectorise the column loop with FMA instructions.
#[inline]
fn apply_5pt_row(
	src: &[f64],
	row_dst: &mut [f64],
	row: usize,
	cols: usize,
	entries: &[fluxion_core::StencilEntry],
) {
	// Extract weights by offset. The canonical order from
	// `Stencil::laplacian_2d_5pt` is:
	//   (0,0), (-1,0), (+1,0), (0,-1), (0,+1)
	// but we handle any ordering.
	let mut w_c = 0.0_f64;
	let mut w_n = 0.0_f64;
	let mut w_s = 0.0_f64;
	let mut w_w = 0.0_f64;
	let mut w_e = 0.0_f64;
	for e in entries {
		match (e.dr, e.dc) {
			(0, 0) => w_c = e.weight,
			(-1, 0) => w_n = e.weight,
			(1, 0) => w_s = e.weight,
			(0, -1) => w_w = e.weight,
			(0, 1) => w_e = e.weight,
			_ => {}
		}
	}

	let base = row * cols;
	let north = (row - 1) * cols;
	let south = (row + 1) * cols;

	for col in 1..cols - 1 {
		let val = w_c.mul_add(
			src[base + col],
			w_n.mul_add(
				src[north + col],
				w_s.mul_add(
					src[south + col],
					w_w.mul_add(
						src[base + col - 1],
						w_e * src[base + col + 1],
					),
				),
			),
		);
		row_dst[col] = val;
	}
}

/// Unrolled 9-point compact Laplacian for one row.
///
/// Same structure as `apply_5pt_row` but with 4 extra
/// diagonal neighbors. Compiler auto-vectorises the
/// column loop with chained FMA.
#[inline]
fn apply_9pt_row(
	src: &[f64],
	row_dst: &mut [f64],
	row: usize,
	cols: usize,
	entries: &[fluxion_core::StencilEntry],
) {
	let mut w_c = 0.0_f64;
	let mut w_n = 0.0_f64;
	let mut w_s = 0.0_f64;
	let mut w_w = 0.0_f64;
	let mut w_e = 0.0_f64;
	let mut w_nw = 0.0_f64;
	let mut w_ne = 0.0_f64;
	let mut w_sw = 0.0_f64;
	let mut w_se = 0.0_f64;
	for e in entries {
		match (e.dr, e.dc) {
			(0, 0) => w_c = e.weight,
			(-1, 0) => w_n = e.weight,
			(1, 0) => w_s = e.weight,
			(0, -1) => w_w = e.weight,
			(0, 1) => w_e = e.weight,
			(-1, -1) => w_nw = e.weight,
			(-1, 1) => w_ne = e.weight,
			(1, -1) => w_sw = e.weight,
			(1, 1) => w_se = e.weight,
			_ => {}
		}
	}

	let base = row * cols;
	let north = (row - 1) * cols;
	let south = (row + 1) * cols;

	for col in 1..cols - 1 {
		let val = w_c.mul_add(
			src[base + col],
			w_n.mul_add(
				src[north + col],
				w_s.mul_add(
					src[south + col],
					w_w.mul_add(
						src[base + col - 1],
						w_e.mul_add(
							src[base + col + 1],
							w_nw.mul_add(
								src[north + col - 1],
								w_ne.mul_add(
									src[north + col + 1],
									w_sw.mul_add(
										src[south + col - 1],
										w_se * src[south + col + 1],
									),
								),
							),
						),
					),
				),
			),
		);
		row_dst[col] = val;
	}
}

/// Fused stencil + axpy for one interior row:
/// `y[col] += alpha * laplacian(x)[col]`.
///
/// Single pass over x and y — halves memory traffic.
#[inline]
fn fused_5pt_axpy_row(
	alpha: f64,
	src: &[f64],
	row_dst: &mut [f64],
	row: usize,
	cols: usize,
	entries: &[fluxion_core::StencilEntry],
) {
	let mut w_c = 0.0_f64;
	let mut w_n = 0.0_f64;
	let mut w_s = 0.0_f64;
	let mut w_w = 0.0_f64;
	let mut w_e = 0.0_f64;
	// Extract cardinal + center weights (common to 5pt and 9pt).
	if entries.len() == 5 || entries.len() == 9 {
		for e in entries {
			match (e.dr, e.dc) {
				(0, 0) => w_c = e.weight,
				(-1, 0) => w_n = e.weight,
				(1, 0) => w_s = e.weight,
				(0, -1) => w_w = e.weight,
				(0, 1) => w_e = e.weight,
				_ => {}
			}
		}
	}

	let base = row * cols;
	let north = (row - 1) * cols;
	let south = (row + 1) * cols;

	if entries.len() == 9 {
		// 9-point fast path (compact Laplacian).
		let mut w_nw = 0.0_f64;
		let mut w_ne = 0.0_f64;
		let mut w_sw = 0.0_f64;
		let mut w_se = 0.0_f64;
		for e in entries {
			match (e.dr, e.dc) {
				(-1, -1) => w_nw = e.weight,
				(-1, 1) => w_ne = e.weight,
				(1, -1) => w_sw = e.weight,
				(1, 1) => w_se = e.weight,
				_ => {}
			}
		}
		for col in 1..cols - 1 {
			let lap = w_c.mul_add(
				src[base + col],
				w_n.mul_add(
					src[north + col],
					w_s.mul_add(
						src[south + col],
						w_w.mul_add(
							src[base + col - 1],
							w_e.mul_add(
								src[base + col + 1],
								w_nw.mul_add(
									src[north + col - 1],
									w_ne.mul_add(
										src[north + col + 1],
										w_sw.mul_add(
											src[south + col - 1],
											w_se * src[south + col + 1],
										),
									),
								),
							),
						),
					),
				),
			);
			row_dst[col] =
				alpha.mul_add(lap, row_dst[col]);
		}
	} else if entries.len() == 5 {
		for col in 1..cols - 1 {
			let lap = w_c.mul_add(
				src[base + col],
				w_n.mul_add(
					src[north + col],
					w_s.mul_add(
						src[south + col],
						w_w.mul_add(
							src[base + col - 1],
							w_e * src[base + col + 1],
						),
					),
				),
			);
			row_dst[col] = alpha.mul_add(lap, row_dst[col]);
		}
	} else {
		#[allow(
			clippy::cast_possible_wrap,
			clippy::cast_sign_loss,
			clippy::needless_range_loop,
		)]
		for col in 1..cols - 1 {
			let mut lap = 0.0_f64;
			for e in entries {
				let r = (row as i32 + e.dr) as usize;
				let c = (col as i32 + e.dc) as usize;
				lap =
					e.weight.mul_add(src[r * cols + c], lap);
			}
			row_dst[col] = alpha.mul_add(lap, row_dst[col]);
		}
	}
}

/// Merge two Kahan-compensated partial sums.
#[inline]
fn kahan_merge(
	(a, ca): (f64, f64),
	(b, cb): (f64, f64),
) -> (f64, f64) {
	let y = b - (ca + cb);
	let t = a + y;
	let c = (t - a) - y;
	(t, c)
}

/// Compute the boundary value for a single point.
#[inline]
fn bc_value(
	bc: BoundaryCondition,
	self_val: f64,
	neighbor_val: f64,
	spacing: f64,
) -> f64 {
	match bc {
		BoundaryCondition::Dirichlet(val) => val,
		BoundaryCondition::Neumann(flux) => {
			// u_boundary = u_interior + h·(∂u/∂n)
			// Sign is the same for both sides because ∂u/∂n
			// is defined w.r.t. the outward normal, which
			// already flips direction at each boundary.
			spacing.mul_add(flux, neighbor_val)
		}
		BoundaryCondition::Robin { alpha, beta, g } => {
			// Robin: alpha·u + beta·∂u/∂n = g
			// Ghost cell: u_ghost = u_boundary ± h·∂u/∂n
			// ∂u/∂n ≈ (g - alpha·u) / beta
			// u_ghost = u ± h·(g - alpha·u) / beta
			if beta.abs() < 1e-30 {
				// Degenerate: pure Dirichlet (alpha·u = g).
				if alpha.abs() < 1e-30 {
					return 0.0;
				}
				return g / alpha;
			}
			let du_dn = (g - alpha * self_val) / beta;
			spacing.mul_add(du_dn, neighbor_val)
		}
		BoundaryCondition::Periodic => {
			// Periodic BCs require index wrapping in the
			// stencil kernel, not a ghost-cell value.
			// This path should never be reached — periodic
			// boundaries must be intercepted before calling
			// apply_boundaries.
			0.0
		}
	}
}

#[cfg(test)]
mod tests {
	use fluxion_core::{
		Backend, BackendBuffer, Boundaries, DType, Grid,
		Stencil,
	};

	use super::*;

	#[test]
	fn allocate_zeros() {
		let b = CpuBackend::new();
		let buf = b.allocate(100, DType::F64).unwrap();
		assert_eq!(buf.len(), 100);
		assert!(
			buf.as_slice().iter().all(|&v| v == 0.0)
		);
	}

	#[test]
	fn fill_and_norm() {
		let b = CpuBackend::new();
		let mut buf =
			b.allocate(16, DType::F64).unwrap();
		b.fill(&mut buf, 3.0).unwrap();
		let norm = b.norm_l2(&buf).unwrap();
		assert!((norm - 12.0).abs() < 1e-12);
	}

	#[test]
	fn axpy_basic() {
		let b = CpuBackend::new();
		let x =
			b.upload_f64(&[1.0, 2.0, 3.0]).unwrap();
		let mut y =
			b.upload_f64(&[10.0, 20.0, 30.0]).unwrap();
		b.axpy(2.0, &x, &mut y).unwrap();
		let ys = y.as_slice();
		assert!((ys[0] - 12.0).abs() < 1e-14);
		assert!((ys[1] - 24.0).abs() < 1e-14);
		assert!((ys[2] - 36.0).abs() < 1e-14);
	}

	#[test]
	#[allow(clippy::many_single_char_names)]
	fn laplacian_on_quadratic_field() {
		let n = 32;
		let h = 1.0 / (n - 1) as f64;
		let grid = Grid::square(n, h).unwrap();
		let stencil = Stencil::laplacian_2d_5pt(h, h);
		let bcs = Boundaries::zero_dirichlet();
		let b = CpuBackend::new();

		let mut data = vec![0.0_f64; n * n];
		for row in 0..n {
			for col in 0..n {
				let x = col as f64 * h;
				let y = row as f64 * h;
				data[row * n + col] = x * x + y * y;
			}
		}

		let input = b.upload_f64(&data).unwrap();
		let mut output =
			b.allocate(n * n, DType::F64).unwrap();
		b.apply_stencil(
			&input,
			&mut output,
			&grid,
			&stencil,
			&bcs,
		)
		.unwrap();

		let out = output.as_slice();
		for row in 2..n - 2 {
			for col in 2..n - 2 {
				let val = out[row * n + col];
				assert!(
					(val - 4.0).abs() < 1e-10,
					"Δu at ({row},{col}) = {val}, expected 4.0"
				);
			}
		}
	}

	#[test]
	fn dirichlet_nonzero_boundary() {
		let n = 16;
		let h = 1.0 / (n - 1) as f64;
		let grid = Grid::square(n, h).unwrap();
		let stencil = Stencil::laplacian_2d_5pt(h, h);
		let bcs = Boundaries::uniform(
			BoundaryCondition::Dirichlet(100.0),
		);
		let b = CpuBackend::new();

		let data = vec![50.0_f64; n * n];
		let input = b.upload_f64(&data).unwrap();
		let mut output =
			b.allocate(n * n, DType::F64).unwrap();
		b.apply_stencil(
			&input,
			&mut output,
			&grid,
			&stencil,
			&bcs,
		)
		.unwrap();

		let out = output.as_slice();
		for &val in &out[..n] {
			assert!(
				(val - 100.0).abs() < 1e-12,
				"top boundary: got {val}, expected 100",
			);
		}
		for &val in &out[(n - 1) * n..] {
			assert!(
				(val - 100.0).abs() < 1e-12,
				"bottom boundary: got {val}, expected 100",
			);
		}
	}

	#[test]
	fn neumann_zero_flux_preserves_neighbor() {
		let n = 16;
		let h = 1.0 / (n - 1) as f64;
		let grid = Grid::square(n, h).unwrap();
		let stencil = Stencil::laplacian_2d_5pt(h, h);
		let bcs = Boundaries::uniform(
			BoundaryCondition::Neumann(0.0),
		);
		let b = CpuBackend::new();

		let data = vec![42.0_f64; n * n];
		let input = b.upload_f64(&data).unwrap();
		let mut output =
			b.allocate(n * n, DType::F64).unwrap();
		b.apply_stencil(
			&input,
			&mut output,
			&grid,
			&stencil,
			&bcs,
		)
		.unwrap();

		let out = output.as_slice();
		for &val in &out[..n] {
			assert!(
				(val - 42.0).abs() < 1e-10,
				"Neumann top: got {val}, expected 42",
			);
		}
	}
}
