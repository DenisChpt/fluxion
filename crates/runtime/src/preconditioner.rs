use crate::error::Result;
use crate::field::Field;

/// Preconditioner interface for iterative solvers.
///
/// Applied as `M⁻¹·r` in the PCG inner loop. The solve
/// method on `CgSolver` is generic over `P: Preconditioner`,
/// so no vtable / trait object overhead — the compiler
/// monomorphises each variant.
pub trait Preconditioner {
	/// Apply the preconditioner: solve `M·z ≈ r` approximately.
	///
	/// `z` is the output (modified in place), `r` is the input.
	///
	/// # Errors
	/// Returns an error on backend failure.
	fn apply(
		&mut self,
		z: &mut Field,
		r: &Field,
	) -> Result<()>;
}

/// Identity preconditioner (no-op: z = r).
#[derive(Debug)]
pub struct Identity;

impl Preconditioner for Identity {
	#[inline]
	fn apply(
		&mut self,
		z: &mut Field,
		r: &Field,
	) -> Result<()> {
		z.copy_from(r)
	}
}
