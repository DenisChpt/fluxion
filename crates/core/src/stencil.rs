use crate::error::{CoreError, Result};

/// A single stencil entry: row offset, column offset, weight.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StencilEntry {
	pub dr: i32,
	pub dc: i32,
	pub weight: f64,
}

/// Describes a finite-difference stencil pattern.
///
/// Pure data — backends read offsets and weights to
/// parameterize their kernels.
#[derive(Debug, Clone, PartialEq)]
pub struct Stencil {
	entries: Vec<StencilEntry>,
}

impl Stencil {
	/// # Errors
	/// Returns an error if `entries` is empty.
	pub fn new(entries: Vec<StencilEntry>) -> Result<Self> {
		if entries.is_empty() {
			return Err(CoreError::InvalidStencil(
				"stencil must have at least one entry".into(),
			));
		}
		Ok(Self { entries })
	}

	#[must_use]
	pub fn entries(&self) -> &[StencilEntry] {
		&self.entries
	}

	#[must_use]
	pub const fn len(&self) -> usize {
		self.entries.len()
	}

	#[must_use]
	pub const fn is_empty(&self) -> bool {
		self.entries.is_empty()
	}

	/// Standard 5-point Laplacian for a 2D grid (O(h^2)).
	///
	/// ```text
	///        1/dy²
	/// 1/dx²  -2(1/dx²+1/dy²)  1/dx²
	///        1/dy²
	/// ```
	#[must_use]
	#[allow(clippy::similar_names)]
	pub fn laplacian_2d_5pt(dx: f64, dy: f64) -> Self {
		let inv_dx2 = 1.0 / (dx * dx);
		let inv_dy2 = 1.0 / (dy * dy);
		Self {
			entries: vec![
				StencilEntry {
					dr: 0,
					dc: 0,
					weight: -2.0 * (inv_dx2 + inv_dy2),
				},
				StencilEntry { dr: -1, dc: 0, weight: inv_dy2 },
				StencilEntry { dr: 1, dc: 0, weight: inv_dy2 },
				StencilEntry { dr: 0, dc: -1, weight: inv_dx2 },
				StencilEntry { dr: 0, dc: 1, weight: inv_dx2 },
			],
		}
	}

	/// Compact 9-point Laplacian for a 2D grid (O(h^2)).
	///
	/// Same stencil radius as 5-point but 4th-order accurate
	/// on isotropic grids. Uses Mehrstellen (Collatz) weights.
	///
	/// For `dx = dy = h`:
	/// ```text
	/// (1/6h²) [ 1   4   1]
	///         [ 4  -20  4]
	///         [ 1   4   1]
	/// ```
	///
	/// For general `dx, dy` — standard tensor-product 9-point:
	/// ```text
	/// w_nw  w_n   w_ne
	/// w_w   w_c   w_e
	/// w_sw  w_s   w_se
	/// ```
	/// Weights sum to zero. Still only radius-1.
	#[must_use]
	#[allow(clippy::similar_names)]
	pub fn laplacian_2d_9pt(dx: f64, dy: f64) -> Self {
		let inv_dx2 = 1.0 / (dx * dx);
		let inv_dy2 = 1.0 / (dy * dy);

		// Mehrstellen compact 4th-order Laplacian (Collatz).
		//
		// Δ_9pt = Δ_5pt + K·D²_x·D²_y
		// where K = (dx² + dy²)/12.
		//
		// This is O(h⁴) when dx = dy, O(h²) otherwise.
		// Note: for strongly anisotropic grids (dx >> dy),
		// cardinal weights can go negative. Use 5-point
		// stencil for aspect ratios > ~5.
		let dx2 = dx * dx;
		let dy2 = dy * dy;
		let k = (dx2 + dy2) / 12.0;
		let inv_dx2dy2 = inv_dx2 * inv_dy2;

		// D²_x·D²_y stencil weights (tensor product of
		// [1 -2 1]/dx² and [1 -2 1]/dy²):
		//   center:  4/(dx²dy²)
		//   N/S/W/E: -2/(dx²dy²)
		//   corners: 1/(dx²dy²)

		let w_c = -2.0 * (inv_dx2 + inv_dy2)
			+ 4.0 * k * inv_dx2dy2;
		let w_n = inv_dy2 - 2.0 * k * inv_dx2dy2;
		let w_s = w_n;
		let w_w = inv_dx2 - 2.0 * k * inv_dx2dy2;
		let w_e = w_w;
		let w_diag = k * inv_dx2dy2;

		Self {
			entries: vec![
				StencilEntry { dr: 0, dc: 0, weight: w_c },
				// Cardinals.
				StencilEntry { dr: -1, dc: 0, weight: w_n },
				StencilEntry { dr: 1, dc: 0, weight: w_s },
				StencilEntry { dr: 0, dc: -1, weight: w_w },
				StencilEntry { dr: 0, dc: 1, weight: w_e },
				// Diagonals.
				StencilEntry { dr: -1, dc: -1, weight: w_diag },
				StencilEntry { dr: -1, dc: 1, weight: w_diag },
				StencilEntry { dr: 1, dc: -1, weight: w_diag },
				StencilEntry { dr: 1, dc: 1, weight: w_diag },
			],
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn laplacian_5pt_weights_sum_to_zero() {
		let s = Stencil::laplacian_2d_5pt(1.0, 1.0);
		let sum: f64 =
			s.entries().iter().map(|e| e.weight).sum();
		assert!(
			sum.abs() < 1e-14,
			"weights should sum to 0, got {sum}"
		);
	}

	#[test]
	fn laplacian_5pt_uniform_spacing() {
		let s = Stencil::laplacian_2d_5pt(1.0, 1.0);
		assert_eq!(s.len(), 5);
		let center = s
			.entries()
			.iter()
			.find(|e| e.dr == 0 && e.dc == 0)
			.unwrap();
		assert!((center.weight - (-4.0)).abs() < 1e-14);
	}

	#[test]
	fn laplacian_5pt_anisotropic() {
		let dx = 0.1;
		let dy = 0.2;
		let s = Stencil::laplacian_2d_5pt(dx, dy);
		let sum: f64 =
			s.entries().iter().map(|e| e.weight).sum();
		assert!(sum.abs() < 1e-10);
		let east = s
			.entries()
			.iter()
			.find(|e| e.dr == 0 && e.dc == 1)
			.unwrap();
		assert!(
			(east.weight - 1.0 / (dx * dx)).abs() < 1e-14
		);
	}

	#[test]
	fn rejects_empty_stencil() {
		assert!(Stencil::new(vec![]).is_err());
	}

	// ── 9-point stencil tests ──────────────────────

	#[test]
	fn laplacian_9pt_weights_sum_to_zero() {
		let s = Stencil::laplacian_2d_9pt(1.0, 1.0);
		let sum: f64 =
			s.entries().iter().map(|e| e.weight).sum();
		assert!(
			sum.abs() < 1e-14,
			"9pt weights should sum to 0, got {sum}"
		);
	}

	#[test]
	fn laplacian_9pt_has_9_entries() {
		let s = Stencil::laplacian_2d_9pt(0.1, 0.2);
		assert_eq!(s.len(), 9);
	}

	#[test]
	fn laplacian_9pt_uniform_center() {
		let h = 1.0;
		let s = Stencil::laplacian_2d_9pt(h, h);
		let center = s
			.entries()
			.iter()
			.find(|e| e.dr == 0 && e.dc == 0)
			.unwrap();
		// center = -2(1+1) + 4·(1/6)·1 = -4 + 2/3 = -10/3.
		assert!(
			(center.weight - (-10.0 / 3.0)).abs() < 1e-14,
			"center = {}, expected -10/3",
			center.weight,
		);
	}

	#[test]
	fn laplacian_9pt_isotropic_matches_collatz() {
		let h = 1.0;
		let s = Stencil::laplacian_2d_9pt(h, h);
		// For dx=dy=h=1:
		//   K = 2h²/12 = 1/6
		//   center = -4/h² + 4K/(h⁴) = -4 + 4/6 = -10/3
		//   cardinal = 1/h² - 2K/h⁴ = 1 - 1/3 = 2/3
		//   diagonal = K/h⁴ = 1/6
		//
		// Multiply by 6h² to get: [1 4 1; 4 -20 4; 1 4 1].
		let factor = 6.0 * h * h;
		for e in s.entries() {
			let scaled = e.weight * factor;
			let expected = match (e.dr.abs(), e.dc.abs()) {
				(0, 0) => -20.0,
				(0, _) | (_, 0) => 4.0,
				(1, 1) => 1.0,
				_ => unreachable!(),
			};
			assert!(
				(scaled - expected).abs() < 1e-12,
				"({},{}) scaled={scaled}, expected={expected}",
				e.dr,
				e.dc,
			);
		}
	}

	#[test]
	fn laplacian_9pt_anisotropic_sums_to_zero() {
		let s = Stencil::laplacian_2d_9pt(0.05, 0.1);
		let sum: f64 =
			s.entries().iter().map(|e| e.weight).sum();
		assert!(sum.abs() < 1e-10);
	}
}
