use std::io::Write;
use std::path::Path;

use fluxion_core::Grid;
use fluxion_runtime::Field;

/// Write a field to VTK Legacy format (`.vtk`).
///
/// Produces a `STRUCTURED_POINTS` dataset viewable in `ParaView`.
///
/// # Errors
/// Returns an I/O error if writing fails.
pub fn write_vtk(
	path: &Path,
	field: &Field,
	name: &str,
) -> std::io::Result<()> {
	let grid = field.grid();
	let data = field.to_vec_f64();
	write_vtk_data(path, grid, &data, name)
}

/// Write raw f64 data to VTK Legacy format.
///
/// # Errors
/// Returns an I/O error if writing fails.
pub fn write_vtk_data(
	path: &Path,
	grid: &Grid,
	data: &[f64],
	name: &str,
) -> std::io::Result<()> {
	let mut f = std::fs::File::create(path)?;

	writeln!(f, "# vtk DataFile Version 3.0")?;
	writeln!(f, "Fluxion {name}")?;
	writeln!(f, "ASCII")?;
	writeln!(f, "DATASET STRUCTURED_POINTS")?;
	writeln!(
		f,
		"DIMENSIONS {} {} 1",
		grid.cols, grid.rows
	)?;
	writeln!(f, "ORIGIN 0 0 0")?;
	writeln!(
		f,
		"SPACING {} {} 1",
		grid.dx, grid.dy
	)?;
	writeln!(
		f,
		"POINT_DATA {}",
		grid.rows * grid.cols
	)?;
	writeln!(f, "SCALARS {name} double 1")?;
	writeln!(f, "LOOKUP_TABLE default")?;

	for &val in data {
		writeln!(f, "{val:.10e}")?;
	}

	Ok(())
}

/// Write a numbered VTK file for time-series animation.
///
/// Generates filenames like `prefix_000100.vtk`.
///
/// # Errors
/// Returns an I/O error if writing fails.
pub fn write_vtk_timestep(
	dir: &Path,
	prefix: &str,
	step: usize,
	field: &Field,
	name: &str,
) -> std::io::Result<()> {
	let filename =
		format!("{prefix}_{step:06}.vtk");
	let path = dir.join(filename);
	write_vtk(path.as_path(), field, name)
}
