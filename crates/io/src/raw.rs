use std::io::Write;
use std::path::Path;

use fluxion_core::Grid;
use fluxion_runtime::Field;

/// Write a field as raw binary f64 with a small header.
///
/// Format: [rows:u64][cols:u64][dx:f64][dy:f64][data:f64*n]
///
/// # Errors
/// Returns an I/O error if writing fails.
pub fn write_raw(
	path: &Path,
	field: &Field,
) -> std::io::Result<()> {
	let grid = field.grid();
	let data = field.to_vec_f64();

	let mut f = std::fs::File::create(path)?;
	f.write_all(&(grid.rows as u64).to_le_bytes())?;
	f.write_all(&(grid.cols as u64).to_le_bytes())?;
	f.write_all(&grid.dx.to_le_bytes())?;
	f.write_all(&grid.dy.to_le_bytes())?;

	for &val in &data {
		f.write_all(&val.to_le_bytes())?;
	}

	Ok(())
}

/// Read a field from raw binary format.
///
/// # Errors
/// Returns an I/O error if reading fails.
pub fn read_raw(
	path: &Path,
	device: fluxion_runtime::Device,
) -> std::io::Result<Field> {
	use std::io::Read;
	let mut f = std::fs::File::open(path)?;

	let mut buf8 = [0u8; 8];

	f.read_exact(&mut buf8)?;
	let rows = u64::from_le_bytes(buf8) as usize;
	f.read_exact(&mut buf8)?;
	let cols = u64::from_le_bytes(buf8) as usize;
	f.read_exact(&mut buf8)?;
	let dx = f64::from_le_bytes(buf8);
	f.read_exact(&mut buf8)?;
	let dy = f64::from_le_bytes(buf8);

	let grid = Grid::new(rows, cols, dx, dy).map_err(
		|e| {
			std::io::Error::new(
				std::io::ErrorKind::InvalidData,
				e.to_string(),
			)
		},
	)?;

	let n = rows * cols;
	let mut data = vec![0.0_f64; n];
	for val in &mut data {
		f.read_exact(&mut buf8)?;
		*val = f64::from_le_bytes(buf8);
	}

	Field::from_f64(grid, &data, device).map_err(|e| {
		std::io::Error::new(
			std::io::ErrorKind::InvalidData,
			e.to_string(),
		)
	})
}
