use crate::error::{CoreError, Result};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Grid {
	pub rows: usize,
	pub cols: usize,
	pub dx: f64,
	pub dy: f64,
}

impl Grid {
	/// # Errors
	/// Returns an error if dimensions are less than 3
	/// or spacing is not finite and positive.
	pub fn new(
		rows: usize,
		cols: usize,
		dx: f64,
		dy: f64,
	) -> Result<Self> {
		if rows < 3 || cols < 3 {
			return Err(CoreError::InvalidGrid(format!(
				"minimum 3x3, got {rows}x{cols}"
			)));
		}
		if !dx.is_finite()
			|| !dy.is_finite()
			|| dx <= 0.0
			|| dy <= 0.0
		{
			return Err(CoreError::InvalidGrid(format!(
				"spacing must be finite and positive, got dx={dx}, dy={dy}"
			)));
		}
		Ok(Self { rows, cols, dx, dy })
	}

	/// # Errors
	/// See [`Grid::new`].
	pub fn square(n: usize, h: f64) -> Result<Self> {
		Self::new(n, n, h, h)
	}

	#[inline]
	#[must_use]
	pub const fn len(&self) -> usize {
		self.rows * self.cols
	}

	#[inline]
	#[must_use]
	pub const fn is_empty(&self) -> bool {
		self.rows == 0 || self.cols == 0
	}

	/// Row-major linear index.
	#[inline]
	#[must_use]
	pub const fn idx(&self, row: usize, col: usize) -> usize {
		row * self.cols + col
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn valid_grid() {
		let g = Grid::new(10, 20, 0.1, 0.2).unwrap();
		assert_eq!(g.len(), 200);
		assert_eq!(g.idx(3, 5), 3 * 20 + 5);
	}

	#[test]
	fn square_grid() {
		let g = Grid::square(100, 0.01).unwrap();
		assert_eq!(g.rows, 100);
		assert_eq!(g.cols, 100);
		assert!((g.dx - 0.01).abs() < f64::EPSILON);
	}

	#[test]
	fn rejects_too_small() {
		assert!(Grid::new(2, 10, 1.0, 1.0).is_err());
		assert!(Grid::new(10, 2, 1.0, 1.0).is_err());
	}

	#[test]
	fn rejects_bad_spacing() {
		assert!(Grid::new(10, 10, 0.0, 1.0).is_err());
		assert!(Grid::new(10, 10, -1.0, 1.0).is_err());
		assert!(Grid::new(10, 10, f64::NAN, 1.0).is_err());
		assert!(Grid::new(10, 10, f64::INFINITY, 1.0).is_err());
	}
}
