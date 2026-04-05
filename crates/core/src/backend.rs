use std::fmt::Debug;

use crate::boundary::Boundaries;
use crate::dtype::DType;
use crate::error::Result;
use crate::grid::Grid;
use crate::stencil::Stencil;

/// Device-side buffer handle.
///
/// Each backend defines its own concrete buffer type
/// and implements this trait.
pub trait BackendBuffer: Send + Sync + Debug {
	/// Number of scalar elements in the buffer.
	fn len(&self) -> usize;

	/// Whether the buffer is empty.
	fn is_empty(&self) -> bool {
		self.len() == 0
	}

	/// Element type stored in the buffer.
	fn dtype(&self) -> DType;

	/// Copy buffer contents to a host `f64` slice.
	///
	/// # Panics
	/// Panics if `dst.len() != self.len()`.
	fn copy_to_host_f64(&self, dst: &mut [f64]);

	/// Copy buffer contents to a host `f32` slice.
	///
	/// # Panics
	/// Panics if `dst.len() != self.len()`.
	fn copy_to_host_f32(&self, dst: &mut [f32]);
}

/// Computational backend for PDE operations.
///
/// Each backend (CPU/wgpu/CUDA/HIP) implements this trait.
/// The `runtime` crate wraps them behind an enum for
/// type-erased dispatch.
pub trait Backend: Send + Sync + Debug {
	type Buffer: BackendBuffer;

	/// Human-readable backend name.
	fn name(&self) -> &str;

	// ── Memory ───────────────────────────────────────

	/// Allocate a zero-initialized buffer.
	///
	/// # Errors
	/// Returns an error if allocation fails.
	fn allocate(
		&self,
		len: usize,
		dtype: DType,
	) -> Result<Self::Buffer>;

	/// Upload host `f64` data to a device buffer.
	///
	/// # Errors
	/// Returns an error if the transfer fails.
	fn upload_f64(
		&self,
		data: &[f64],
	) -> Result<Self::Buffer>;

	/// Upload host `f32` data to a device buffer.
	///
	/// # Errors
	/// Returns an error if the transfer fails.
	fn upload_f32(
		&self,
		data: &[f32],
	) -> Result<Self::Buffer>;

	// ── PDE Operations (hot path) ────────────────────

	/// Apply a stencil operation: `output = stencil(input)`.
	///
	/// Both buffers must have `grid.len()` elements.
	/// Boundary conditions are enforced on the output.
	///
	/// # Errors
	/// Returns an error on dimension mismatch or backend failure.
	fn apply_stencil(
		&self,
		input: &Self::Buffer,
		output: &mut Self::Buffer,
		grid: &Grid,
		stencil: &Stencil,
		boundaries: &Boundaries,
	) -> Result<()>;

	/// Copy `src` into `dst` (both must have same length).
	///
	/// # Errors
	/// Returns an error on dimension mismatch.
	fn copy(
		&self,
		src: &Self::Buffer,
		dst: &mut Self::Buffer,
	) -> Result<()>;

	/// Fill buffer with a constant value.
	///
	/// # Errors
	/// Returns an error on backend failure.
	fn fill(
		&self,
		buf: &mut Self::Buffer,
		value: f64,
	) -> Result<()>;

	/// AXPY: `y = alpha * x + y`.
	///
	/// # Errors
	/// Returns an error on dimension mismatch or backend failure.
	fn axpy(
		&self,
		alpha: f64,
		x: &Self::Buffer,
		y: &mut Self::Buffer,
	) -> Result<()>;

	/// Scale: `buf *= alpha`.
	///
	/// # Errors
	/// Returns an error on backend failure.
	fn scale(
		&self,
		buf: &mut Self::Buffer,
		alpha: f64,
	) -> Result<()>;

	/// Variable-coefficient stencil:
	/// `output[i] = coeff[i] * stencil(input)[i]`.
	///
	/// Like `apply_stencil` but each grid point is scaled by
	/// a spatially-varying coefficient field `coeff(x,y)`.
	///
	/// Default implementation: `apply_stencil` + pointwise_mult.
	///
	/// # Errors
	/// Returns an error on backend failure.
	fn apply_stencil_var(
		&self,
		input: &Self::Buffer,
		output: &mut Self::Buffer,
		coeff: &Self::Buffer,
		grid: &Grid,
		stencil: &Stencil,
		boundaries: &Boundaries,
	) -> Result<()>;

	/// Fused stencil + axpy: `y += alpha * stencil(x)`.
	///
	/// Single memory pass — halves bandwidth vs separate
	/// `apply_stencil` + `axpy`.  Boundary points in `y`
	/// are left untouched.
	///
	/// # Errors
	/// Returns an error on dimension mismatch or backend failure.
	fn stencil_axpy(
		&self,
		alpha: f64,
		x: &Self::Buffer,
		y: &mut Self::Buffer,
		grid: &Grid,
		stencil: &Stencil,
		boundaries: &Boundaries,
	) -> Result<()>;

	/// L2 norm: `sqrt(sum(x_i²))`.
	///
	/// # Errors
	/// Returns an error on backend failure.
	fn norm_l2(&self, buf: &Self::Buffer) -> Result<f64>;

	// ── Reductions ───────────────────────────────────

	/// Sum of all elements.
	///
	/// # Errors
	/// Returns an error on backend failure.
	fn reduce_sum(&self, buf: &Self::Buffer) -> Result<f64>;

	/// Maximum element value.
	///
	/// # Errors
	/// Returns an error on backend failure.
	fn reduce_max(&self, buf: &Self::Buffer) -> Result<f64>;

	/// Minimum element value.
	///
	/// # Errors
	/// Returns an error on backend failure.
	fn reduce_min(&self, buf: &Self::Buffer) -> Result<f64>;

	/// Dot product: `sum(x_i * y_i)`.
	///
	/// # Errors
	/// Returns an error on dimension mismatch or backend failure.
	fn dot(
		&self,
		x: &Self::Buffer,
		y: &Self::Buffer,
	) -> Result<f64>;

	/// Dual dot product: computes `dot(a,b)` and `dot(c,d)`
	/// in a single pass. One kernel launch, one sync, one
	/// readback — halves the overhead vs two separate `dot`
	/// calls. Critical in CG/BiCGSTAB inner loops.
	///
	/// # Errors
	/// Returns an error on dimension mismatch or backend failure.
	fn dot2(
		&self,
		a: &Self::Buffer,
		b: &Self::Buffer,
		c: &Self::Buffer,
		d: &Self::Buffer,
	) -> Result<(f64, f64)> {
		// Default: two separate dots. Override for GPU.
		let ab = self.dot(a, b)?;
		let cd = self.dot(c, d)?;
		Ok((ab, cd))
	}

	// ── Multigrid operations ─────────────────────────

	/// Full-weighting restriction: fine → coarse.
	///
	/// `coarse[i,j] = avg(fine[2i..2i+1, 2j..2j+1])`.
	///
	/// # Errors
	/// Returns an error on backend failure.
	fn restrict(
		&self,
		fine: &Self::Buffer,
		coarse: &mut Self::Buffer,
		fine_grid: &Grid,
		coarse_grid: &Grid,
	) -> Result<()>;

	/// Prolongation: coarse → fine (additive).
	///
	/// `fine[i,j] += coarse[i/2, j/2]`.
	///
	/// # Errors
	/// Returns an error on backend failure.
	fn prolong(
		&self,
		coarse: &Self::Buffer,
		fine: &mut Self::Buffer,
		coarse_grid: &Grid,
		fine_grid: &Grid,
	) -> Result<()>;

	/// Weighted Jacobi: `x += ω·D⁻¹·(b - A·x)`.
	///
	/// For the Laplacian stencil, `D = diag(A)`.
	/// Interior points only; boundaries untouched.
	///
	/// # Errors
	/// Returns an error on backend failure.
	fn weighted_jacobi(
		&self,
		x: &mut Self::Buffer,
		b: &Self::Buffer,
		omega: f64,
		grid: &Grid,
		stencil: &Stencil,
		boundaries: &Boundaries,
	) -> Result<()>;

	// ── Extended vector operations ───────────────────

	/// Pointwise multiply: `z[i] = x[i] * y[i]`.
	///
	/// # Errors
	/// Returns an error on dimension mismatch or backend failure.
	fn pointwise_mult(
		&self,
		x: &Self::Buffer,
		y: &Self::Buffer,
		z: &mut Self::Buffer,
	) -> Result<()>;

	/// Pointwise divide: `z[i] = x[i] / y[i]`.
	///
	/// # Errors
	/// Returns an error on dimension mismatch or backend failure.
	fn pointwise_div(
		&self,
		x: &Self::Buffer,
		y: &Self::Buffer,
		z: &mut Self::Buffer,
	) -> Result<()>;

	/// WAXPY: `w[i] = alpha * x[i] + beta * y[i]`.
	///
	/// Three-buffer variant — avoids a copy + axpy.
	///
	/// # Errors
	/// Returns an error on dimension mismatch or backend failure.
	fn waxpy(
		&self,
		alpha: f64,
		x: &Self::Buffer,
		beta: f64,
		y: &Self::Buffer,
		w: &mut Self::Buffer,
	) -> Result<()>;

	/// AYPX: `y[i] = x[i] + alpha * y[i]`.
	///
	/// # Errors
	/// Returns an error on dimension mismatch or backend failure.
	fn aypx(
		&self,
		alpha: f64,
		x: &Self::Buffer,
		y: &mut Self::Buffer,
	) -> Result<()>;

	/// Reciprocal: `x[i] = 1.0 / x[i]`.
	///
	/// # Errors
	/// Returns an error on backend failure.
	fn reciprocal(
		&self,
		buf: &mut Self::Buffer,
	) -> Result<()>;

	/// Absolute value: `x[i] = |x[i]|`.
	///
	/// # Errors
	/// Returns an error on backend failure.
	fn abs_val(
		&self,
		buf: &mut Self::Buffer,
	) -> Result<()>;

	/// Pointwise max: `z[i] = max(x[i], y[i])`.
	fn pointwise_max(
		&self,
		x: &Self::Buffer,
		y: &Self::Buffer,
		z: &mut Self::Buffer,
	) -> Result<()>;

	/// Pointwise min: `z[i] = min(x[i], y[i])`.
	fn pointwise_min(
		&self,
		x: &Self::Buffer,
		y: &Self::Buffer,
		z: &mut Self::Buffer,
	) -> Result<()>;

	// ── Convection-diffusion operator ────────────────

	/// Apply convection-diffusion operator:
	/// `output = κ·Δ(u) + v·∇(u)`
	///
	/// where κ is the diffusion coefficient field, (vx,vy) is the
	/// velocity field, Δ is the Laplacian (5-point) and ∇ uses
	/// first-order upwind differencing for the convection term.
	///
	/// # Errors
	/// Returns an error on backend failure.
	#[allow(clippy::too_many_arguments)]
	fn apply_conv_diff(
		&self,
		u: &Self::Buffer,
		output: &mut Self::Buffer,
		kappa: &Self::Buffer,
		vx: &Self::Buffer,
		vy: &Self::Buffer,
		grid: &Grid,
		boundaries: &Boundaries,
	) -> Result<()>;

	/// Fused convection-diffusion + axpy:
	/// `output += alpha * (κ·Δ(u) + v·∇(u))`
	///
	/// Single memory pass — halves bandwidth vs separate
	/// `apply_conv_diff` + `axpy`.
	///
	/// # Errors
	/// Returns an error on backend failure.
	#[allow(clippy::too_many_arguments)]
	fn conv_diff_axpy(
		&self,
		alpha: f64,
		u: &Self::Buffer,
		output: &mut Self::Buffer,
		kappa: &Self::Buffer,
		vx: &Self::Buffer,
		vy: &Self::Buffer,
		grid: &Grid,
		boundaries: &Boundaries,
	) -> Result<()>;
}
