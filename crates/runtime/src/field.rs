use fluxion_core::{
	Backend, Boundaries, DType, Grid, Stencil,
};

use crate::device::Device;
use crate::error::{Result, RuntimeError};
use crate::storage::BufferStorage;

/// A scalar field on a structured 2D grid.
///
/// Wraps a device buffer and grid metadata. All operations
/// dispatch to the correct backend through `BufferStorage` —
/// zero trait objects, zero vtable overhead.
#[derive(Debug)]
pub struct Field {
	pub(crate) storage: BufferStorage,
	grid: Grid,
}

impl Field {
	// ── Constructors (allocating) ────────────────────

	/// Allocate a zero-initialized field.
	///
	/// # Errors
	/// Returns an error if the device is unavailable.
	pub fn zeros(
		grid: Grid,
		dtype: DType,
		device: Device,
	) -> Result<Self> {
		let storage = match device {
			#[cfg(feature = "cpu")]
			Device::Cpu => {
				let b = crate::storage::cpu_backend();
				BufferStorage::Cpu(
					b.allocate(grid.len(), dtype)?,
				)
			}
			#[cfg(feature = "wgpu")]
			Device::Wgpu { .. } => {
				let b = crate::storage::wgpu_backend();
				BufferStorage::Wgpu(
					b.allocate(grid.len(), dtype)?,
				)
			}
			#[cfg(feature = "hip")]
			Device::Hip { .. } => {
				let b = crate::storage::hip_backend();
				BufferStorage::Hip(
					b.allocate(grid.len(), dtype)?,
				)
			}
			#[allow(unreachable_patterns)]
			_ => {
				return Err(
					RuntimeError::BackendUnavailable(
						device.to_string(),
					),
				);
			}
		};
		Ok(Self { storage, grid })
	}

	/// Create a field from host `f64` data.
	///
	/// # Errors
	/// Returns an error on dimension mismatch or unavailable device.
	pub fn from_f64(
		grid: Grid,
		data: &[f64],
		device: Device,
	) -> Result<Self> {
		if data.len() != grid.len() {
			return Err(
				fluxion_core::CoreError::DimensionMismatch {
					expected: grid.len(),
					got: data.len(),
				}
				.into(),
			);
		}
		let storage = match device {
			#[cfg(feature = "cpu")]
			Device::Cpu => {
				let b = crate::storage::cpu_backend();
				BufferStorage::Cpu(b.upload_f64(data)?)
			}
			#[cfg(feature = "wgpu")]
			Device::Wgpu { .. } => {
				let b = crate::storage::wgpu_backend();
				BufferStorage::Wgpu(b.upload_f64(data)?)
			}
			#[cfg(feature = "hip")]
			Device::Hip { .. } => {
				let b = crate::storage::hip_backend();
				BufferStorage::Hip(b.upload_f64(data)?)
			}
			#[allow(unreachable_patterns)]
			_ => {
				return Err(
					RuntimeError::BackendUnavailable(
						device.to_string(),
					),
				);
			}
		};
		Ok(Self { storage, grid })
	}

	// ── Accessors ────────────────────────────────────

	#[inline]
	#[must_use]
	pub const fn grid(&self) -> &Grid {
		&self.grid
	}

	#[inline]
	#[must_use]
	pub const fn device(&self) -> Device {
		self.storage.device()
	}

	#[inline]
	#[must_use]
	pub fn dtype(&self) -> DType {
		self.storage.dtype()
	}

	/// Copy field data back to host as `f64`.
	#[must_use]
	pub fn to_vec_f64(&self) -> Vec<f64> {
		let mut v = vec![0.0_f64; self.storage.len()];
		self.storage.copy_to_host_f64(&mut v);
		v
	}

	// ── Operations (zero-alloc hot path) ─────────────

	/// Copy contents from `other` into `self`. Zero allocation.
	///
	/// # Errors
	/// Returns an error on device mismatch or dimension error.
	#[inline]
	pub fn copy_from(&mut self, other: &Self) -> Result<()> {
		match (&other.storage, &mut self.storage) {
			#[cfg(feature = "cpu")]
			(BufferStorage::Cpu(s), BufferStorage::Cpu(d)) => {
				crate::storage::cpu_backend()
					.copy(s, d)?;
			}
			#[cfg(feature = "wgpu")]
			(
				BufferStorage::Wgpu(s),
				BufferStorage::Wgpu(d),
			) => {
				crate::storage::wgpu_backend()
					.copy(s, d)?;
			}
			#[cfg(feature = "hip")]
			(
				BufferStorage::Hip(s),
				BufferStorage::Hip(d),
			) => {
				crate::storage::hip_backend()
					.copy(s, d)?;
			}
			#[allow(unreachable_patterns)]
			_ => {
				return Err(RuntimeError::DeviceMismatch {
					a: other.storage.device(),
					b: self.storage.device(),
				});
			}
		}
		Ok(())
	}


	/// Apply a stencil: `output = stencil(self)`.
	///
	/// **Zero allocation** — `output` must be pre-allocated.
	///
	/// # Errors
	/// Returns an error on device mismatch or dimension error.
	#[inline]
	pub fn apply_stencil_into(
		&self,
		stencil: &Stencil,
		boundaries: &Boundaries,
		output: &mut Self,
	) -> Result<()> {
		let grid = self.grid;
		match (&self.storage, &mut output.storage) {
			#[cfg(feature = "cpu")]
			(BufferStorage::Cpu(inp), BufferStorage::Cpu(out)) => {
				let b = crate::storage::cpu_backend();
				b.apply_stencil(
					inp, out, &grid, stencil, boundaries,
				)?;
			}
			#[cfg(feature = "wgpu")]
			(
				BufferStorage::Wgpu(inp),
				BufferStorage::Wgpu(out),
			) => {
				let b = crate::storage::wgpu_backend();
				b.apply_stencil(
					inp, out, &grid, stencil, boundaries,
				)?;
			}
			#[cfg(feature = "hip")]
			(
				BufferStorage::Hip(inp),
				BufferStorage::Hip(out),
			) => {
				let b = crate::storage::hip_backend();
				b.apply_stencil(
					inp, out, &grid, stencil, boundaries,
				)?;
			}
			#[allow(unreachable_patterns)]
			_ => {
				return Err(RuntimeError::DeviceMismatch {
					a: self.storage.device(),
					b: output.storage.device(),
				});
			}
		}
		Ok(())
	}

	/// Variable-coefficient stencil:
	/// `output[i] = coeff[i] * stencil(self)[i]`.
	///
	/// # Errors
	/// Returns an error on device mismatch or dimension error.
	pub fn apply_stencil_var_into(
		&self,
		stencil: &Stencil,
		boundaries: &Boundaries,
		coeff: &Self,
		output: &mut Self,
	) -> Result<()> {
		let grid = self.grid;
		match (
			&self.storage,
			&mut output.storage,
			&coeff.storage,
		) {
			#[cfg(feature = "cpu")]
			(
				BufferStorage::Cpu(inp),
				BufferStorage::Cpu(out),
				BufferStorage::Cpu(c),
			) => {
				crate::storage::cpu_backend()
					.apply_stencil_var(
						inp, out, c, &grid, stencil,
						boundaries,
					)?;
			}
			#[cfg(feature = "wgpu")]
			(
				BufferStorage::Wgpu(inp),
				BufferStorage::Wgpu(out),
				BufferStorage::Wgpu(c),
			) => {
				crate::storage::wgpu_backend()
					.apply_stencil_var(
						inp, out, c, &grid, stencil,
						boundaries,
					)?;
			}
			#[cfg(feature = "hip")]
			(
				BufferStorage::Hip(inp),
				BufferStorage::Hip(out),
				BufferStorage::Hip(c),
			) => {
				crate::storage::hip_backend()
					.apply_stencil_var(
						inp, out, c, &grid, stencil,
						boundaries,
					)?;
			}
			#[allow(unreachable_patterns)]
			_ => {
				return Err(RuntimeError::DeviceMismatch {
					a: self.storage.device(),
					b: output.storage.device(),
				});
			}
		}
		Ok(())
	}

	/// Fused stencil + axpy: `self += alpha * stencil(other)`.
	///
	/// Single memory pass — halves bandwidth vs separate
	/// `apply_stencil_into` + `axpy`. Boundary points in
	/// `self` are left untouched.
	///
	/// # Errors
	/// Returns an error on device mismatch or dimension error.
	#[inline]
	pub fn stencil_axpy(
		&mut self,
		alpha: f64,
		other: &Self,
		stencil: &Stencil,
		boundaries: &Boundaries,
	) -> Result<()> {
		let grid = self.grid;
		match (&other.storage, &mut self.storage) {
			#[cfg(feature = "cpu")]
			(BufferStorage::Cpu(x), BufferStorage::Cpu(y)) => {
				crate::storage::cpu_backend().stencil_axpy(
					alpha, x, y, &grid, stencil, boundaries,
				)?;
			}
			#[cfg(feature = "wgpu")]
			(
				BufferStorage::Wgpu(x),
				BufferStorage::Wgpu(y),
			) => {
				crate::storage::wgpu_backend().stencil_axpy(
					alpha, x, y, &grid, stencil, boundaries,
				)?;
			}
			#[cfg(feature = "hip")]
			(
				BufferStorage::Hip(x),
				BufferStorage::Hip(y),
			) => {
				crate::storage::hip_backend().stencil_axpy(
					alpha, x, y, &grid, stencil, boundaries,
				)?;
			}
			#[allow(unreachable_patterns)]
			_ => {
				return Err(RuntimeError::DeviceMismatch {
					a: other.storage.device(),
					b: self.storage.device(),
				});
			}
		}
		Ok(())
	}

	/// Fill with a constant value.
	///
	/// # Errors
	/// Returns an error on backend failure.
	#[inline]
	pub fn fill(&mut self, value: f64) -> Result<()> {
		match &mut self.storage {
			#[cfg(feature = "cpu")]
			BufferStorage::Cpu(b) => {
				crate::storage::cpu_backend().fill(b, value)?;
			}
			#[cfg(feature = "wgpu")]
			BufferStorage::Wgpu(b) => {
				crate::storage::wgpu_backend()
					.fill(b, value)?;
			}
			#[cfg(feature = "hip")]
			BufferStorage::Hip(b) => {
				crate::storage::hip_backend()
					.fill(b, value)?;
			}
		}
		Ok(())
	}

	/// Zero out boundary rows/cols (for Dirichlet BCs).
	///
	/// Sets first/last row and first/last column to 0.0.
	/// This ensures global vector operations (axpy, scale)
	/// don't corrupt boundary values needed by stencils.
	///
	/// # Errors
	/// Returns an error on backend failure.
	pub(crate) fn zero_boundaries(
		&mut self,
	) -> Result<()> {
		let rows = self.grid.rows;
		let cols = self.grid.cols;
		match &mut self.storage {
			#[cfg(feature = "cpu")]
			BufferStorage::Cpu(b) => {
				let s = b.as_mut_slice();
				// Top and bottom rows.
				for c in 0..cols {
					s[c] = 0.0;
					s[(rows - 1) * cols + c] = 0.0;
				}
				// Left and right columns.
				for r in 1..rows - 1 {
					s[r * cols] = 0.0;
					s[r * cols + cols - 1] = 0.0;
				}
			}
			#[cfg(feature = "wgpu")]
			BufferStorage::Wgpu(_b) => {
				// Fallback: read-modify-write.
				let mut data = self.to_vec_f64();
				for c in 0..cols {
					data[c] = 0.0;
					data[(rows - 1) * cols + c] = 0.0;
				}
				for r in 1..rows - 1 {
					data[r * cols] = 0.0;
					data[r * cols + cols - 1] = 0.0;
				}
				let tmp = Self::from_f64(
					self.grid, &data,
					crate::Device::Wgpu { adapter: 0 },
				)?;
				self.copy_from(&tmp)?;
			}
			#[cfg(feature = "hip")]
			BufferStorage::Hip(b) => {
				crate::storage::hip_backend()
					.zero_boundaries(b, rows, cols);
			}
		}
		Ok(())
	}

	/// Scale: `self *= alpha`.
	///
	/// # Errors
	/// Returns an error on backend failure.
	#[inline]
	pub fn scale(&mut self, alpha: f64) -> Result<()> {
		match &mut self.storage {
			#[cfg(feature = "cpu")]
			BufferStorage::Cpu(b) => {
				crate::storage::cpu_backend()
					.scale(b, alpha)?;
			}
			#[cfg(feature = "wgpu")]
			BufferStorage::Wgpu(b) => {
				crate::storage::wgpu_backend()
					.scale(b, alpha)?;
			}
			#[cfg(feature = "hip")]
			BufferStorage::Hip(b) => {
				crate::storage::hip_backend()
					.scale(b, alpha)?;
			}
		}
		Ok(())
	}

	/// AXPY: `self += alpha * other`.
	///
	/// # Errors
	/// Returns an error on device mismatch or dimension error.
	#[inline]
	pub fn axpy(
		&mut self,
		alpha: f64,
		other: &Self,
	) -> Result<()> {
		match (&other.storage, &mut self.storage) {
			#[cfg(feature = "cpu")]
			(BufferStorage::Cpu(x), BufferStorage::Cpu(y)) => {
				crate::storage::cpu_backend()
					.axpy(alpha, x, y)?;
			}
			#[cfg(feature = "wgpu")]
			(BufferStorage::Wgpu(x), BufferStorage::Wgpu(y)) => {
				crate::storage::wgpu_backend()
					.axpy(alpha, x, y)?;
			}
			#[cfg(feature = "hip")]
			(BufferStorage::Hip(x), BufferStorage::Hip(y)) => {
				crate::storage::hip_backend()
					.axpy(alpha, x, y)?;
			}
			#[allow(unreachable_patterns)]
			_ => {
				return Err(RuntimeError::DeviceMismatch {
					a: other.storage.device(),
					b: self.storage.device(),
				});
			}
		}
		Ok(())
	}

	/// L2 norm: `sqrt(sum(x_i²))`.
	///
	/// # Errors
	/// Returns an error on backend failure.
	#[inline]
	pub fn norm_l2(&self) -> Result<f64> {
		match &self.storage {
			#[cfg(feature = "cpu")]
			BufferStorage::Cpu(b) => {
				Ok(crate::storage::cpu_backend().norm_l2(b)?)
			}
			#[cfg(feature = "wgpu")]
			BufferStorage::Wgpu(b) => {
				Ok(crate::storage::wgpu_backend()
					.norm_l2(b)?)
			}
			#[cfg(feature = "hip")]
			BufferStorage::Hip(b) => {
				Ok(crate::storage::hip_backend()
					.norm_l2(b)?)
			}
		}
	}

	// ── Reductions / diagnostics ─────────────────────

	/// Sum of all elements.
	///
	/// # Errors
	/// Returns an error on backend failure.
	#[inline]
	pub fn sum(&self) -> Result<f64> {
		match &self.storage {
			#[cfg(feature = "cpu")]
			BufferStorage::Cpu(b) => {
				Ok(crate::storage::cpu_backend()
					.reduce_sum(b)?)
			}
			#[cfg(feature = "wgpu")]
			BufferStorage::Wgpu(b) => {
				Ok(crate::storage::wgpu_backend()
					.reduce_sum(b)?)
			}
			#[cfg(feature = "hip")]
			BufferStorage::Hip(b) => {
				Ok(crate::storage::hip_backend()
					.reduce_sum(b)?)
			}
		}
	}

	/// Maximum element value.
	///
	/// # Errors
	/// Returns an error on backend failure.
	#[inline]
	pub fn max(&self) -> Result<f64> {
		match &self.storage {
			#[cfg(feature = "cpu")]
			BufferStorage::Cpu(b) => {
				Ok(crate::storage::cpu_backend()
					.reduce_max(b)?)
			}
			#[cfg(feature = "wgpu")]
			BufferStorage::Wgpu(b) => {
				Ok(crate::storage::wgpu_backend()
					.reduce_max(b)?)
			}
			#[cfg(feature = "hip")]
			BufferStorage::Hip(b) => {
				Ok(crate::storage::hip_backend()
					.reduce_max(b)?)
			}
		}
	}

	/// Minimum element value.
	///
	/// # Errors
	/// Returns an error on backend failure.
	#[inline]
	pub fn min(&self) -> Result<f64> {
		match &self.storage {
			#[cfg(feature = "cpu")]
			BufferStorage::Cpu(b) => {
				Ok(crate::storage::cpu_backend()
					.reduce_min(b)?)
			}
			#[cfg(feature = "wgpu")]
			BufferStorage::Wgpu(b) => {
				Ok(crate::storage::wgpu_backend()
					.reduce_min(b)?)
			}
			#[cfg(feature = "hip")]
			BufferStorage::Hip(b) => {
				Ok(crate::storage::hip_backend()
					.reduce_min(b)?)
			}
		}
	}

	/// Dot product: `sum(self_i * other_i)`.
	///
	/// # Errors
	/// Returns an error on device mismatch or dimension error.
	#[inline]
	pub fn dot(&self, other: &Self) -> Result<f64> {
		match (&self.storage, &other.storage) {
			#[cfg(feature = "cpu")]
			(BufferStorage::Cpu(a), BufferStorage::Cpu(b)) => {
				Ok(crate::storage::cpu_backend()
					.dot(a, b)?)
			}
			#[cfg(feature = "wgpu")]
			(
				BufferStorage::Wgpu(a),
				BufferStorage::Wgpu(b),
			) => Ok(crate::storage::wgpu_backend()
				.dot(a, b)?),
			#[cfg(feature = "hip")]
			(
				BufferStorage::Hip(a),
				BufferStorage::Hip(b),
			) => Ok(crate::storage::hip_backend()
				.dot(a, b)?),
			#[allow(unreachable_patterns)]
			_ => Err(RuntimeError::DeviceMismatch {
				a: self.storage.device(),
				b: other.storage.device(),
			}),
		}
	}

	/// Dual dot product: `(dot(self, b), dot(c, d))`.
	///
	/// Single kernel + single readback on GPU backends.
	/// Halves the sync overhead vs two separate `dot` calls.
	///
	/// # Errors
	/// Returns an error on device mismatch or dimension error.
	#[inline]
	pub fn dot2(
		&self,
		b: &Self,
		c: &Self,
		d: &Self,
	) -> Result<(f64, f64)> {
		match (
			&self.storage,
			&b.storage,
			&c.storage,
			&d.storage,
		) {
			#[cfg(feature = "cpu")]
			(
				BufferStorage::Cpu(a),
				BufferStorage::Cpu(b),
				BufferStorage::Cpu(c),
				BufferStorage::Cpu(d),
			) => Ok(crate::storage::cpu_backend()
				.dot2(a, b, c, d)?),
			#[cfg(feature = "wgpu")]
			(
				BufferStorage::Wgpu(a),
				BufferStorage::Wgpu(b),
				BufferStorage::Wgpu(c),
				BufferStorage::Wgpu(d),
			) => Ok(crate::storage::wgpu_backend()
				.dot2(a, b, c, d)?),
			#[cfg(feature = "hip")]
			(
				BufferStorage::Hip(a),
				BufferStorage::Hip(b),
				BufferStorage::Hip(c),
				BufferStorage::Hip(d),
			) => Ok(crate::storage::hip_backend()
				.dot2(a, b, c, d)?),
			#[allow(unreachable_patterns)]
			_ => Err(RuntimeError::DeviceMismatch {
				a: self.storage.device(),
				b: b.storage.device(),
			}),
		}
	}

	/// Integral over the domain: `sum(u) * dx * dy`.
	///
	/// # Errors
	/// Returns an error on backend failure.
	#[inline]
	pub fn integral(&self) -> Result<f64> {
		let s = self.sum()?;
		Ok(s * self.grid.dx * self.grid.dy)
	}

	/// Swap two fields in O(1) — pointer swap, no data copy.
	pub const fn swap(a: &mut Self, b: &mut Self) {
		std::mem::swap(a, b);
	}

	// ── Extended vector operations ───────────────────

	/// Pointwise multiply: `self = x * y` (element-wise).
	///
	/// # Errors
	/// Returns an error on device mismatch or dimension error.
	#[inline]
	pub fn pointwise_mult(
		&mut self,
		x: &Self,
		y: &Self,
	) -> Result<()> {
		match (
			&x.storage,
			&y.storage,
			&mut self.storage,
		) {
			#[cfg(feature = "cpu")]
			(
				BufferStorage::Cpu(a),
				BufferStorage::Cpu(b),
				BufferStorage::Cpu(c),
			) => {
				crate::storage::cpu_backend()
					.pointwise_mult(a, b, c)?;
			}
			#[cfg(feature = "wgpu")]
			(
				BufferStorage::Wgpu(a),
				BufferStorage::Wgpu(b),
				BufferStorage::Wgpu(c),
			) => {
				crate::storage::wgpu_backend()
					.pointwise_mult(a, b, c)?;
			}
			#[cfg(feature = "hip")]
			(
				BufferStorage::Hip(a),
				BufferStorage::Hip(b),
				BufferStorage::Hip(c),
			) => {
				crate::storage::hip_backend()
					.pointwise_mult(a, b, c)?;
			}
			#[allow(unreachable_patterns)]
			_ => {
				return Err(RuntimeError::DeviceMismatch {
					a: x.storage.device(),
					b: self.storage.device(),
				});
			}
		}
		Ok(())
	}

	/// Pointwise divide: `self = x / y` (element-wise).
	///
	/// # Errors
	/// Returns an error on device mismatch or dimension error.
	#[inline]
	pub fn pointwise_div(
		&mut self,
		x: &Self,
		y: &Self,
	) -> Result<()> {
		match (
			&x.storage,
			&y.storage,
			&mut self.storage,
		) {
			#[cfg(feature = "cpu")]
			(
				BufferStorage::Cpu(a),
				BufferStorage::Cpu(b),
				BufferStorage::Cpu(c),
			) => {
				crate::storage::cpu_backend()
					.pointwise_div(a, b, c)?;
			}
			#[cfg(feature = "wgpu")]
			(
				BufferStorage::Wgpu(a),
				BufferStorage::Wgpu(b),
				BufferStorage::Wgpu(c),
			) => {
				crate::storage::wgpu_backend()
					.pointwise_div(a, b, c)?;
			}
			#[cfg(feature = "hip")]
			(
				BufferStorage::Hip(a),
				BufferStorage::Hip(b),
				BufferStorage::Hip(c),
			) => {
				crate::storage::hip_backend()
					.pointwise_div(a, b, c)?;
			}
			#[allow(unreachable_patterns)]
			_ => {
				return Err(RuntimeError::DeviceMismatch {
					a: x.storage.device(),
					b: self.storage.device(),
				});
			}
		}
		Ok(())
	}

	/// Pointwise max: `self[i] = max(x[i], y[i])`.
	pub fn pointwise_max(
		&mut self,
		x: &Self,
		y: &Self,
	) -> Result<()> {
		match (
			&x.storage,
			&y.storage,
			&mut self.storage,
		) {
			#[cfg(feature = "cpu")]
			(
				BufferStorage::Cpu(a),
				BufferStorage::Cpu(b),
				BufferStorage::Cpu(c),
			) => {
				crate::storage::cpu_backend()
					.pointwise_max(a, b, c)?;
			}
			#[cfg(feature = "wgpu")]
			(
				BufferStorage::Wgpu(a),
				BufferStorage::Wgpu(b),
				BufferStorage::Wgpu(c),
			) => {
				crate::storage::wgpu_backend()
					.pointwise_max(a, b, c)?;
			}
			#[cfg(feature = "hip")]
			(
				BufferStorage::Hip(a),
				BufferStorage::Hip(b),
				BufferStorage::Hip(c),
			) => {
				crate::storage::hip_backend()
					.pointwise_max(a, b, c)?;
			}
			#[allow(unreachable_patterns)]
			_ => {
				return Err(RuntimeError::DeviceMismatch {
					a: x.storage.device(),
					b: self.storage.device(),
				});
			}
		}
		Ok(())
	}

	/// Pointwise min: `self[i] = min(x[i], y[i])`.
	pub fn pointwise_min(
		&mut self,
		x: &Self,
		y: &Self,
	) -> Result<()> {
		match (
			&x.storage,
			&y.storage,
			&mut self.storage,
		) {
			#[cfg(feature = "cpu")]
			(
				BufferStorage::Cpu(a),
				BufferStorage::Cpu(b),
				BufferStorage::Cpu(c),
			) => {
				crate::storage::cpu_backend()
					.pointwise_min(a, b, c)?;
			}
			#[cfg(feature = "wgpu")]
			(
				BufferStorage::Wgpu(a),
				BufferStorage::Wgpu(b),
				BufferStorage::Wgpu(c),
			) => {
				crate::storage::wgpu_backend()
					.pointwise_min(a, b, c)?;
			}
			#[cfg(feature = "hip")]
			(
				BufferStorage::Hip(a),
				BufferStorage::Hip(b),
				BufferStorage::Hip(c),
			) => {
				crate::storage::hip_backend()
					.pointwise_min(a, b, c)?;
			}
			#[allow(unreachable_patterns)]
			_ => {
				return Err(RuntimeError::DeviceMismatch {
					a: x.storage.device(),
					b: self.storage.device(),
				});
			}
		}
		Ok(())
	}

	/// WAXPY: `self = alpha * x + beta * y`.
	///
	/// # Errors
	/// Returns an error on device mismatch or dimension error.
	#[inline]
	pub fn waxpy(
		&mut self,
		alpha: f64,
		x: &Self,
		beta: f64,
		y: &Self,
	) -> Result<()> {
		match (
			&x.storage,
			&y.storage,
			&mut self.storage,
		) {
			#[cfg(feature = "cpu")]
			(
				BufferStorage::Cpu(a),
				BufferStorage::Cpu(b),
				BufferStorage::Cpu(c),
			) => {
				crate::storage::cpu_backend()
					.waxpy(alpha, a, beta, b, c)?;
			}
			#[cfg(feature = "wgpu")]
			(
				BufferStorage::Wgpu(a),
				BufferStorage::Wgpu(b),
				BufferStorage::Wgpu(c),
			) => {
				crate::storage::wgpu_backend()
					.waxpy(alpha, a, beta, b, c)?;
			}
			#[cfg(feature = "hip")]
			(
				BufferStorage::Hip(a),
				BufferStorage::Hip(b),
				BufferStorage::Hip(c),
			) => {
				crate::storage::hip_backend()
					.waxpy(alpha, a, beta, b, c)?;
			}
			#[allow(unreachable_patterns)]
			_ => {
				return Err(RuntimeError::DeviceMismatch {
					a: x.storage.device(),
					b: self.storage.device(),
				});
			}
		}
		Ok(())
	}

	/// AYPX: `self = other + alpha * self`.
	///
	/// # Errors
	/// Returns an error on device mismatch or dimension error.
	#[inline]
	pub fn aypx(
		&mut self,
		alpha: f64,
		other: &Self,
	) -> Result<()> {
		match (&other.storage, &mut self.storage) {
			#[cfg(feature = "cpu")]
			(BufferStorage::Cpu(x), BufferStorage::Cpu(y)) => {
				crate::storage::cpu_backend()
					.aypx(alpha, x, y)?;
			}
			#[cfg(feature = "wgpu")]
			(
				BufferStorage::Wgpu(x),
				BufferStorage::Wgpu(y),
			) => {
				crate::storage::wgpu_backend()
					.aypx(alpha, x, y)?;
			}
			#[cfg(feature = "hip")]
			(
				BufferStorage::Hip(x),
				BufferStorage::Hip(y),
			) => {
				crate::storage::hip_backend()
					.aypx(alpha, x, y)?;
			}
			#[allow(unreachable_patterns)]
			_ => {
				return Err(RuntimeError::DeviceMismatch {
					a: other.storage.device(),
					b: self.storage.device(),
				});
			}
		}
		Ok(())
	}

	/// Reciprocal: `self[i] = 1 / self[i]`.
	///
	/// # Errors
	/// Returns an error on backend failure.
	#[inline]
	pub fn reciprocal(&mut self) -> Result<()> {
		match &mut self.storage {
			#[cfg(feature = "cpu")]
			BufferStorage::Cpu(b) => {
				crate::storage::cpu_backend()
					.reciprocal(b)?;
			}
			#[cfg(feature = "wgpu")]
			BufferStorage::Wgpu(b) => {
				crate::storage::wgpu_backend()
					.reciprocal(b)?;
			}
			#[cfg(feature = "hip")]
			BufferStorage::Hip(b) => {
				crate::storage::hip_backend()
					.reciprocal(b)?;
			}
		}
		Ok(())
	}

	/// Absolute value: `self[i] = |self[i]|`.
	///
	/// # Errors
	/// Returns an error on backend failure.
	#[inline]
	pub fn abs_val(&mut self) -> Result<()> {
		match &mut self.storage {
			#[cfg(feature = "cpu")]
			BufferStorage::Cpu(b) => {
				crate::storage::cpu_backend().abs_val(b)?;
			}
			#[cfg(feature = "wgpu")]
			BufferStorage::Wgpu(b) => {
				crate::storage::wgpu_backend()
					.abs_val(b)?;
			}
			#[cfg(feature = "hip")]
			BufferStorage::Hip(b) => {
				crate::storage::hip_backend()
					.abs_val(b)?;
			}
		}
		Ok(())
	}
}
