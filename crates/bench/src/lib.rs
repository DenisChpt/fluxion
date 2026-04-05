use fluxion_core::Grid;
use fluxion_runtime::{Device, Field};

/// Build a Gaussian field centered at (0.5, 0.5).
///
/// # Errors
/// Returns an error if field creation fails.
pub fn gaussian_field(
	n: usize,
	device: Device,
) -> fluxion_runtime::Result<(Grid, Field)> {
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h)?;
	let cx = 0.5_f64;
	let cy = 0.5_f64;
	let sigma = 0.1_f64;
	let inv_2s2 = 1.0 / (2.0 * sigma * sigma);

	let mut data = vec![0.0_f64; n * n];
	for row in 0..n {
		for col in 0..n {
			let x = col as f64 * h;
			let y = row as f64 * h;
			let dx = x - cx;
			let dy = y - cy;
			let r2 = dx.mul_add(dx, dy * dy);
			data[row * n + col] = (-r2 * inv_2s2).exp();
		}
	}

	let field = Field::from_f64(grid, &data, device)?;
	Ok((grid, field))
}

/// Build a sinusoidal field: `sin(πx)·sin(πy)`.
///
/// # Errors
/// Returns an error if field creation fails.
pub fn sinusoidal_field(
	n: usize,
	device: Device,
) -> fluxion_runtime::Result<(Grid, Field)> {
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h)?;

	let mut data = vec![0.0_f64; n * n];
	for row in 0..n {
		for col in 0..n {
			let x = col as f64 * h;
			let y = row as f64 * h;
			data[row * n + col] = (std::f64::consts::PI * x)
				.sin()
				* (std::f64::consts::PI * y).sin();
		}
	}

	let field = Field::from_f64(grid, &data, device)?;
	Ok((grid, field))
}

/// Build a quadratic field: `u(x,y) = x² + y²`.
///
/// # Errors
/// Returns an error if field creation fails.
pub fn quadratic_field(
	n: usize,
	device: Device,
) -> fluxion_runtime::Result<(Grid, Field)> {
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h)?;

	let mut data = vec![0.0_f64; n * n];
	for row in 0..n {
		for col in 0..n {
			let x = col as f64 * h;
			let y = row as f64 * h;
			data[row * n + col] = x.mul_add(x, y * y);
		}
	}

	let field = Field::from_f64(grid, &data, device)?;
	Ok((grid, field))
}
