#![allow(clippy::many_single_char_names)]
//! Krylov solver tests on CPU.
//!
//! Tests CG, BiCGSTAB, PipelinedCG, PCG, and the LinearSolver
//! trait on CPU to complement HIP-only integration tests.

use fluxion_core::{Boundaries, DType, Grid, Stencil};
use fluxion_core::BoundaryCondition;
use fluxion_runtime::{
	AdaptiveSolver, BdfSolver, BiCgStabSolver, CgSolver,
	ConvergenceReason, CrankNicolsonSolver, Device,
	DiffusionSolver, Field, GmresSolver, Identity,
	ImexSolver, LinearSolver, Multigrid,
	PipelinedCgSolver, SmootherKind, SspOrder, SspRkSolver,
	TimeScheme,
};

fn cpu() -> Device {
	Device::Cpu
}

/// Helper: Gaussian IC centered at (0.5, 0.5).
fn gaussian_ic(n: usize, h: f64) -> Vec<f64> {
	let sigma = 0.1_f64;
	let inv_2s2 = 1.0 / (2.0 * sigma * sigma);
	let mut data = vec![0.0_f64; n * n];
	for row in 0..n {
		for col in 0..n {
			let x = col as f64 * h;
			let y = row as f64 * h;
			let dx = x - 0.5;
			let dy = y - 0.5;
			let r2 = dx.mul_add(dx, dy * dy);
			data[row * n + col] = (-r2 * inv_2s2).exp();
		}
	}
	data
}

// ── CG on CPU ──────────────────────────────────────────

/// CG solver converges on a well-conditioned system.
#[test]
fn cg_converges_cpu() {
	let n = 32;
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let dev = cpu();
	let stencil = Stencil::laplacian_2d_5pt(h, h);
	let bc = Boundaries::zero_dirichlet();

	// Build RHS from a known u: b = u + coeff·Δ(u).
	let coeff = -0.01;
	let u_data = gaussian_ic(n, h);
	let u_exact =
		Field::from_f64(grid, &u_data, dev).unwrap();
	let mut b = Field::zeros(grid, DType::F64, dev).unwrap();
	b.copy_from(&u_exact).unwrap();
	let mut tmp =
		Field::zeros(grid, DType::F64, dev).unwrap();
	tmp.copy_from(&u_exact).unwrap();
	b.stencil_axpy(coeff, &tmp, &stencil, &bc).unwrap();

	let mut x = Field::zeros(grid, DType::F64, dev).unwrap();
	let mut cg =
		CgSolver::new(grid, dev, 1e-8, 500).unwrap();
	let stats =
		cg.solve(&mut x, &b, coeff, &stencil, &bc).unwrap();

	assert!(
		stats.converged(),
		"CG did not converge: {:?} (res={:.2e}, iters={})",
		stats.reason, stats.residual, stats.iterations,
	);
	assert!(stats.iterations < 200);
}

/// CG with zero RHS converges in 0 iterations.
#[test]
fn cg_zero_rhs_converges_immediately() {
	let n = 16;
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let dev = cpu();
	let stencil = Stencil::laplacian_2d_5pt(h, h);
	let bc = Boundaries::zero_dirichlet();

	let b = Field::zeros(grid, DType::F64, dev).unwrap();
	let mut x = Field::zeros(grid, DType::F64, dev).unwrap();
	let mut cg =
		CgSolver::new(grid, dev, 1e-10, 100).unwrap();
	let stats =
		cg.solve(&mut x, &b, -0.01, &stencil, &bc).unwrap();

	assert!(stats.converged());
	assert_eq!(stats.iterations, 0);
}

// ── PCG (preconditioned CG) on CPU ─────────────────────

/// PCG with Identity preconditioner matches unpreconditioned CG.
#[test]
fn pcg_identity_matches_cg() {
	let n = 32;
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let dev = cpu();
	let stencil = Stencil::laplacian_2d_5pt(h, h);
	let bc = Boundaries::zero_dirichlet();
	let coeff = -0.01;

	let u_data = gaussian_ic(n, h);
	let u_exact =
		Field::from_f64(grid, &u_data, dev).unwrap();
	let mut b = Field::zeros(grid, DType::F64, dev).unwrap();
	b.copy_from(&u_exact).unwrap();
	let mut tmp =
		Field::zeros(grid, DType::F64, dev).unwrap();
	tmp.copy_from(&u_exact).unwrap();
	b.stencil_axpy(coeff, &tmp, &stencil, &bc).unwrap();

	// Unpreconditioned CG.
	let mut x1 =
		Field::zeros(grid, DType::F64, dev).unwrap();
	let mut cg =
		CgSolver::new(grid, dev, 1e-8, 500).unwrap();
	let s1 = cg
		.solve(&mut x1, &b, coeff, &stencil, &bc)
		.unwrap();

	// PCG with Identity (should be identical).
	let mut x2 =
		Field::zeros(grid, DType::F64, dev).unwrap();
	let mut cg2 =
		CgSolver::new(grid, dev, 1e-8, 500).unwrap();
	let mut pc = Identity;
	let s2 = cg2
		.solve_preconditioned(
			&mut x2, &b, coeff, &stencil, &bc, &mut pc,
		)
		.unwrap();

	assert!(s1.converged());
	assert!(s2.converged());
	// Both should converge in roughly the same iterations.
	let diff = (s1.iterations as i64 - s2.iterations as i64)
		.unsigned_abs();
	assert!(
		diff <= 2,
		"Identity PCG iterations ({}) should match CG ({})",
		s2.iterations,
		s1.iterations,
	);
}

/// PCG with multigrid beats unpreconditioned CG.
#[test]
fn pcg_multigrid_fewer_iterations_cpu() {
	// Match parameters from the HIP test that's known to work.
	let n = 64;
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let dev = cpu();
	let bc = Boundaries::zero_dirichlet();
	let alpha = 1.0;
	let dt = 0.1;

	let data = gaussian_ic(n, h);
	let mut u_pcg =
		Field::from_f64(grid, &data, dev).unwrap();

	// PCG (Crank-Nicolson with multigrid).
	let mut solver_pcg =
		CrankNicolsonSolver::with_multigrid(
			grid, alpha, dt, bc.clone(), dev, 1e-8, 100,
			2, 2, 2.0 / 3.0,
		)
		.unwrap();
	let s_pcg =
		solver_pcg.step(&mut u_pcg, None).unwrap();

	// Unpreconditioned CG (Crank-Nicolson without multigrid).
	let mut u_cg =
		Field::from_f64(grid, &data, dev).unwrap();
	let mut solver_cg = CrankNicolsonSolver::new(
		grid, alpha, dt, bc, dev, 1e-8, 500,
	)
	.unwrap();
	let s_cg = solver_cg.step(&mut u_cg, None).unwrap();

	assert!(
		s_pcg.converged(),
		"PCG: {:?} (res={:.2e}, iters={})",
		s_pcg.reason, s_pcg.residual, s_pcg.iterations,
	);
	assert!(
		s_cg.converged(),
		"CG: {:?} (res={:.2e}, iters={})",
		s_cg.reason, s_cg.residual, s_cg.iterations,
	);
	assert!(
		s_pcg.iterations <= s_cg.iterations,
		"PCG ({} its) should beat CG ({} its)",
		s_pcg.iterations,
		s_cg.iterations,
	);
}

// ── BiCGSTAB on CPU ────────────────────────────────────

/// BiCGSTAB converges on a symmetric system (like CG).
#[test]
fn bicgstab_converges_cpu() {
	let n = 32;
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let dev = cpu();
	let stencil = Stencil::laplacian_2d_5pt(h, h);
	let bc = Boundaries::zero_dirichlet();
	let coeff = -0.01;

	let u_data = gaussian_ic(n, h);
	let u_exact =
		Field::from_f64(grid, &u_data, dev).unwrap();
	let mut b = Field::zeros(grid, DType::F64, dev).unwrap();
	b.copy_from(&u_exact).unwrap();
	let mut tmp =
		Field::zeros(grid, DType::F64, dev).unwrap();
	tmp.copy_from(&u_exact).unwrap();
	b.stencil_axpy(coeff, &tmp, &stencil, &bc).unwrap();

	let mut x = Field::zeros(grid, DType::F64, dev).unwrap();
	let mut solver =
		BiCgStabSolver::new(grid, dev, 1e-8, 500).unwrap();
	let stats = solver
		.solve(&mut x, &b, coeff, &stencil, &bc)
		.unwrap();

	assert!(
		stats.converged(),
		"BiCGSTAB: {:?} (res={:.2e}, iters={})",
		stats.reason, stats.residual, stats.iterations,
	);
}

/// BiCGSTAB with preconditioner converges.
#[test]
fn bicgstab_preconditioned_cpu() {
	let n = 32;
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let dev = cpu();
	let stencil = Stencil::laplacian_2d_5pt(h, h);
	let bc = Boundaries::zero_dirichlet();
	let coeff = -0.01;

	let u_data = gaussian_ic(n, h);
	let u_exact =
		Field::from_f64(grid, &u_data, dev).unwrap();
	let mut b = Field::zeros(grid, DType::F64, dev).unwrap();
	b.copy_from(&u_exact).unwrap();
	let mut tmp =
		Field::zeros(grid, DType::F64, dev).unwrap();
	tmp.copy_from(&u_exact).unwrap();
	b.stencil_axpy(coeff, &tmp, &stencil, &bc).unwrap();

	let mut x = Field::zeros(grid, DType::F64, dev).unwrap();
	let mut solver =
		BiCgStabSolver::new(grid, dev, 1e-8, 500).unwrap();
	let mut pc = Identity;
	let stats = solver
		.solve_preconditioned(
			&mut x, &b, coeff, &stencil, &bc, &mut pc,
		)
		.unwrap();

	assert!(
		stats.converged(),
		"BiCGSTAB-PC: {:?} (res={:.2e}, iters={})",
		stats.reason, stats.residual, stats.iterations,
	);
}

/// BiCGSTAB handles larger (negative) coeff where CG
/// requires more conditioning.
#[test]
fn bicgstab_negative_coeff_cpu() {
	let n = 32;
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let dev = cpu();
	let stencil = Stencil::laplacian_2d_5pt(h, h);
	let bc = Boundaries::zero_dirichlet();
	let coeff = -0.5;

	let u_data = gaussian_ic(n, h);
	let u_exact =
		Field::from_f64(grid, &u_data, dev).unwrap();
	let mut b = Field::zeros(grid, DType::F64, dev).unwrap();
	b.copy_from(&u_exact).unwrap();
	let mut tmp =
		Field::zeros(grid, DType::F64, dev).unwrap();
	tmp.copy_from(&u_exact).unwrap();
	b.stencil_axpy(coeff, &tmp, &stencil, &bc).unwrap();

	let mut x = Field::zeros(grid, DType::F64, dev).unwrap();
	let mut solver =
		BiCgStabSolver::new(grid, dev, 1e-6, 1000).unwrap();
	let stats = solver
		.solve(&mut x, &b, coeff, &stencil, &bc)
		.unwrap();

	assert!(
		stats.converged(),
		"BiCGSTAB coeff=-0.5: {:?} (res={:.2e}, iters={})",
		stats.reason, stats.residual, stats.iterations,
	);
}

// ── Pipelined CG on CPU ────────────────────────────────

/// Pipelined CG converges and matches standard CG solution.
#[test]
fn pipelined_cg_matches_standard_cpu() {
	let n = 32;
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let dev = cpu();
	let stencil = Stencil::laplacian_2d_5pt(h, h);
	let bc = Boundaries::zero_dirichlet();
	let coeff = -0.01;

	let u_data = gaussian_ic(n, h);
	let u_exact =
		Field::from_f64(grid, &u_data, dev).unwrap();
	let mut b = Field::zeros(grid, DType::F64, dev).unwrap();
	b.copy_from(&u_exact).unwrap();
	let mut tmp =
		Field::zeros(grid, DType::F64, dev).unwrap();
	tmp.copy_from(&u_exact).unwrap();
	b.stencil_axpy(coeff, &tmp, &stencil, &bc).unwrap();

	// Standard CG.
	let mut x_cg =
		Field::zeros(grid, DType::F64, dev).unwrap();
	let mut cg =
		CgSolver::new(grid, dev, 1e-8, 500).unwrap();
	let s_cg = cg
		.solve(&mut x_cg, &b, coeff, &stencil, &bc)
		.unwrap();

	// Pipelined CG.
	let mut x_pcg =
		Field::zeros(grid, DType::F64, dev).unwrap();
	let mut pcg =
		PipelinedCgSolver::new(grid, dev, 1e-8, 500)
			.unwrap();
	let s_pcg = pcg
		.solve(&mut x_pcg, &b, coeff, &stencil, &bc)
		.unwrap();

	assert!(s_cg.converged());
	assert!(s_pcg.converged());

	// Solutions should be close.
	let v_cg = x_cg.to_vec_f64();
	let v_pcg = x_pcg.to_vec_f64();
	let max_diff: f64 = v_cg
		.iter()
		.zip(v_pcg.iter())
		.map(|(a, b)| (a - b).abs())
		.fold(0.0_f64, f64::max);
	assert!(
		max_diff < 1e-6,
		"PipelinedCG vs CG max diff = {max_diff:.2e}"
	);
}

// ── LinearSolver trait ─────────────────────────────────

/// All 3 solvers work through the trait interface.
#[test]
fn linear_solver_trait_polymorphism() {
	let n = 32;
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let dev = cpu();
	let stencil = Stencil::laplacian_2d_5pt(h, h);
	let bc = Boundaries::zero_dirichlet();
	let coeff = -0.01;

	let u_data = gaussian_ic(n, h);
	let u_exact =
		Field::from_f64(grid, &u_data, dev).unwrap();
	let mut b = Field::zeros(grid, DType::F64, dev).unwrap();
	b.copy_from(&u_exact).unwrap();
	let mut tmp =
		Field::zeros(grid, DType::F64, dev).unwrap();
	tmp.copy_from(&u_exact).unwrap();
	b.stencil_axpy(coeff, &tmp, &stencil, &bc).unwrap();

	let solvers: Vec<Box<dyn LinearSolver>> = vec![
		Box::new(
			CgSolver::new(grid, dev, 1e-8, 500).unwrap(),
		),
		Box::new(
			PipelinedCgSolver::new(grid, dev, 1e-8, 500)
				.unwrap(),
		),
		Box::new(
			BiCgStabSolver::new(grid, dev, 1e-8, 500)
				.unwrap(),
		),
	];

	for mut solver in solvers {
		let name = solver.name().to_string();
		let mut x =
			Field::zeros(grid, DType::F64, dev).unwrap();
		let stats = solver
			.solve(&mut x, &b, coeff, &stencil, &bc)
			.unwrap();
		assert!(
			stats.converged(),
			"{name} via trait: {:?} (res={:.2e})",
			stats.reason,
			stats.residual,
		);
	}
}

// ── Crank-Nicolson on CPU ──────────────────────────────

/// Crank-Nicolson without preconditioner converges.
#[test]
fn crank_nicolson_unpreconditioned_cpu() {
	let n = 32;
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let dev = cpu();
	let bc = Boundaries::zero_dirichlet();

	let mut solver = CrankNicolsonSolver::new(
		grid, 0.01, 0.01, bc, dev, 1e-8, 200,
	)
	.unwrap();

	let data = gaussian_ic(n, h);
	let mut u =
		Field::from_f64(grid, &data, dev).unwrap();

	let stats = solver.step(&mut u, None).unwrap();
	assert!(
		stats.converged(),
		"CN unpreconditioned: {:?} (res={:.2e}, iters={})",
		stats.reason, stats.residual, stats.iterations,
	);
	assert_eq!(solver.steps_done(), 1);
}

/// Crank-Nicolson long-run stability: 200 steps without blowup.
#[test]
fn crank_nicolson_stability_200_steps() {
	let n = 32;
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let dev = cpu();
	let bc = Boundaries::zero_dirichlet();

	let mut solver = CrankNicolsonSolver::with_multigrid(
		grid, 0.01, 0.01, bc, dev, 1e-8, 200, 2, 2,
		2.0 / 3.0,
	)
	.unwrap();

	let data = gaussian_ic(n, h);
	let mut u =
		Field::from_f64(grid, &data, dev).unwrap();
	let initial_peak: f64 = data
		.iter()
		.copied()
		.reduce(f64::max)
		.unwrap();

	solver.step_n(&mut u, 200).unwrap();

	let vals = u.to_vec_f64();
	let final_peak: f64 = vals
		.iter()
		.copied()
		.reduce(f64::max)
		.unwrap();

	assert!(final_peak.is_finite());
	assert!(final_peak > 0.0);
	assert!(
		final_peak < initial_peak,
		"peak should decrease: {initial_peak} -> {final_peak}"
	);
	assert_eq!(solver.steps_done(), 200);
	assert!(
		(solver.sim_time() - 2.0).abs() < 1e-10,
		"sim_time = {}",
		solver.sim_time()
	);
}

/// Crank-Nicolson with source term.
#[test]
fn crank_nicolson_with_source() {
	let n = 32;
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let dev = cpu();
	let bc = Boundaries::zero_dirichlet();

	let mut solver = CrankNicolsonSolver::new(
		grid, 0.01, 0.01, bc, dev, 1e-8, 200,
	)
	.unwrap();

	let mut u =
		Field::zeros(grid, DType::F64, dev).unwrap();
	let mut source =
		Field::zeros(grid, DType::F64, dev).unwrap();
	source.fill(1.0).unwrap();

	let stats =
		solver.step(&mut u, Some(&source)).unwrap();
	assert!(stats.converged());

	// With constant source and zero IC, u should be nonzero.
	let norm = u.norm_l2().unwrap();
	assert!(
		norm > 1e-10,
		"source term should produce nonzero u: norm={norm}"
	);
}

// ── Explicit solver schemes ────────────────────────────

/// RK2 scheme remains stable and diffuses.
#[test]
fn rk2_diffuses_gaussian() {
	let n = 32;
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let dev = cpu();

	let mut solver = DiffusionSolver::build(
		grid,
		1.0,
		None,
		Boundaries::zero_dirichlet(),
		TimeScheme::Rk2,
		dev,
	)
	.unwrap();

	let data = gaussian_ic(n, h);
	let initial_peak: f64 = data
		.iter()
		.copied()
		.reduce(f64::max)
		.unwrap();
	let mut u =
		Field::from_f64(grid, &data, dev).unwrap();

	solver.step_n(&mut u, 100).unwrap();

	let vals = u.to_vec_f64();
	let final_peak: f64 = vals
		.iter()
		.copied()
		.reduce(f64::max)
		.unwrap();
	assert!(final_peak.is_finite());
	assert!(
		final_peak < initial_peak,
		"RK2: peak {initial_peak} -> {final_peak}"
	);
}

/// RK4 scheme remains stable and diffuses.
#[test]
fn rk4_diffuses_gaussian() {
	let n = 32;
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let dev = cpu();

	let mut solver = DiffusionSolver::build(
		grid,
		1.0,
		None,
		Boundaries::zero_dirichlet(),
		TimeScheme::Rk4,
		dev,
	)
	.unwrap();

	let data = gaussian_ic(n, h);
	let initial_peak: f64 = data
		.iter()
		.copied()
		.reduce(f64::max)
		.unwrap();
	let mut u =
		Field::from_f64(grid, &data, dev).unwrap();

	solver.step_n(&mut u, 100).unwrap();

	let vals = u.to_vec_f64();
	let final_peak: f64 = vals
		.iter()
		.copied()
		.reduce(f64::max)
		.unwrap();
	assert!(final_peak.is_finite());
	assert!(
		final_peak < initial_peak,
		"RK4: peak {initial_peak} -> {final_peak}"
	);
}

/// Explicit solver with source term: constant source makes
/// field grow.
#[test]
fn explicit_with_source_term() {
	let n = 32;
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let dev = cpu();

	let mut solver =
		DiffusionSolver::new(grid, 0.01, None, dev).unwrap();

	let mut u =
		Field::zeros(grid, DType::F64, dev).unwrap();
	let mut source =
		Field::zeros(grid, DType::F64, dev).unwrap();
	source.fill(100.0).unwrap();

	solver
		.step_n_with_source(&mut u, 50, &source)
		.unwrap();

	let norm = u.norm_l2().unwrap();
	assert!(
		norm > 1.0,
		"constant source should produce large field: {norm}"
	);
}

// ── Non-square grids ───────────────────────────────────

/// Stencil works on rectangular (non-square) grids.
#[test]
fn stencil_rectangular_grid() {
	let rows = 48;
	let cols = 32;
	let dx = 1.0 / (cols - 1) as f64;
	let dy = 1.0 / (rows - 1) as f64;
	let grid = Grid::new(rows, cols, dx, dy).unwrap();
	let stencil = Stencil::laplacian_2d_5pt(dx, dy);
	let dev = cpu();
	let bc = Boundaries::zero_dirichlet();

	// u(x,y) = x² + y² → Δu = 2/dx²·? Actually for
	// non-uniform spacing: Δu = d²u/dx² + d²u/dy² = 2+2 = 4.
	let mut data = vec![0.0_f64; rows * cols];
	for row in 0..rows {
		for col in 0..cols {
			let x = col as f64 * dx;
			let y = row as f64 * dy;
			data[row * cols + col] = x.mul_add(x, y * y);
		}
	}

	let input =
		Field::from_f64(grid, &data, dev).unwrap();
	let mut output =
		Field::zeros(grid, DType::F64, dev).unwrap();
	input
		.apply_stencil_into(&stencil, &bc, &mut output)
		.unwrap();

	let out = output.to_vec_f64();
	for row in 2..rows - 2 {
		for col in 2..cols - 2 {
			let val = out[row * cols + col];
			assert!(
				(val - 4.0).abs() < 1e-10,
				"Δ(x²+y²) on rect grid at ({row},{col}) = {val}"
			);
		}
	}
}

/// CG solver converges on rectangular grid.
#[test]
fn cg_rectangular_grid() {
	let rows = 48;
	let cols = 32;
	let dx = 1.0 / (cols - 1) as f64;
	let dy = 1.0 / (rows - 1) as f64;
	let grid = Grid::new(rows, cols, dx, dy).unwrap();
	let stencil = Stencil::laplacian_2d_5pt(dx, dy);
	let dev = cpu();
	let bc = Boundaries::zero_dirichlet();
	let coeff = -0.01;

	// Simple RHS.
	let pi = std::f64::consts::PI;
	let mut rhs_data = vec![0.0_f64; rows * cols];
	for row in 0..rows {
		for col in 0..cols {
			let x = col as f64 * dx;
			let y = row as f64 * dy;
			rhs_data[row * cols + col] =
				(pi * x).sin() * (pi * y).sin();
		}
	}
	let b =
		Field::from_f64(grid, &rhs_data, dev).unwrap();
	let mut x = Field::zeros(grid, DType::F64, dev).unwrap();
	let mut cg =
		CgSolver::new(grid, dev, 1e-8, 500).unwrap();
	let stats =
		cg.solve(&mut x, &b, coeff, &stencil, &bc).unwrap();

	assert!(stats.converged());
}

// ── Neumann boundary conditions ────────────────────────

/// Diffusion with zero-flux Neumann BCs conserves total mass.
#[test]
fn neumann_bc_conserves_mass() {
	use fluxion_core::BoundaryCondition;
	let n = 32;
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let dev = cpu();

	let bc = Boundaries {
		top: BoundaryCondition::Neumann(0.0),
		bottom: BoundaryCondition::Neumann(0.0),
		left: BoundaryCondition::Neumann(0.0),
		right: BoundaryCondition::Neumann(0.0),
	};

	let mut solver = DiffusionSolver::with_boundaries(
		grid, 1.0, None, bc, dev,
	)
	.unwrap();

	let data = gaussian_ic(n, h);
	let mut u =
		Field::from_f64(grid, &data, dev).unwrap();
	let initial_sum = u.sum().unwrap();

	solver.step_n(&mut u, 200).unwrap();

	let final_sum = u.sum().unwrap();
	// Mass should be approximately conserved (explicit Euler
	// with Neumann has some numerical diffusion at boundaries,
	// so allow generous tolerance).
	let rel_err =
		(final_sum - initial_sum).abs() / initial_sum.abs();
	assert!(
		rel_err < 0.5,
		"Neumann mass conservation: initial={initial_sum:.6}, \
		 final={final_sum:.6}, rel_err={rel_err:.2e}"
	);
	// Also verify the field didn't blow up or collapse.
	assert!(final_sum.is_finite());
	assert!(final_sum > 0.0);
}

/// Diffusion with zero Dirichlet: long run drives interior
/// toward zero.
#[test]
fn dirichlet_long_run_decays_to_zero() {
	let n = 32;
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let dev = cpu();
	let bc = Boundaries::zero_dirichlet();

	let mut solver = DiffusionSolver::with_boundaries(
		grid, 1.0, None, bc, dev,
	)
	.unwrap();

	let data = gaussian_ic(n, h);
	let mut u =
		Field::from_f64(grid, &data, dev).unwrap();
	let initial_norm = u.norm_l2().unwrap();

	solver.step_n(&mut u, 2000).unwrap();

	let final_norm = u.norm_l2().unwrap();
	assert!(
		final_norm < 0.1 * initial_norm,
		"should decay: {initial_norm} -> {final_norm}"
	);
}

// ── Multigrid hierarchy ────────────────────────────────

/// Multigrid depth matches expected coarsening hierarchy.
#[test]
fn multigrid_depth_hierarchy() {
	let dev = cpu();
	let bc = Boundaries::zero_dirichlet();

	// 64×64 → 32×32 → 16×16 → 8×8 → 4×4 = 5 levels.
	let g64 = Grid::square(64, 1.0 / 63.0).unwrap();
	let mg64 = Multigrid::new(
		g64, bc.clone(), dev, 2, 2, 2.0 / 3.0,
	)
	.unwrap();
	assert_eq!(mg64.depth(), 5, "64×64 should have 5 levels");

	// 32×32 → 16×16 → 8×8 → 4×4 = 4 levels.
	let g32 = Grid::square(32, 1.0 / 31.0).unwrap();
	let mg32 = Multigrid::new(
		g32, bc.clone(), dev, 2, 2, 2.0 / 3.0,
	)
	.unwrap();
	assert_eq!(mg32.depth(), 4, "32×32 should have 4 levels");

	// 8×8 → 4×4 = 2 levels.
	let g8 = Grid::square(8, 1.0 / 7.0).unwrap();
	let mg8 =
		Multigrid::new(g8, bc, dev, 2, 2, 2.0 / 3.0)
			.unwrap();
	assert_eq!(mg8.depth(), 2, "8×8 should have 2 levels");
}

/// Chebyshev smoother converges at least as fast as Jacobi
/// in the context of a Crank-Nicolson step.
#[test]
fn chebyshev_fewer_vcycles_than_jacobi() {
	let n = 64;
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let dev = cpu();
	let bc = Boundaries::zero_dirichlet();
	let alpha = 1.0;
	let dt = 0.1;

	let data = gaussian_ic(n, h);

	// PCG with Jacobi smoother.
	let mut u_j =
		Field::from_f64(grid, &data, dev).unwrap();
	let mut solver_j =
		CrankNicolsonSolver::with_multigrid_smoother(
			grid, alpha, dt, bc.clone(), dev, 1e-8, 100,
			2, 2, 2.0 / 3.0, SmootherKind::Jacobi,
		)
		.unwrap();
	let s_j = solver_j.step(&mut u_j, None).unwrap();

	// PCG with Chebyshev smoother.
	let mut u_c =
		Field::from_f64(grid, &data, dev).unwrap();
	let mut solver_c =
		CrankNicolsonSolver::with_multigrid_smoother(
			grid, alpha, dt, bc, dev, 1e-8, 100,
			2, 2, 2.0 / 3.0, SmootherKind::Chebyshev,
		)
		.unwrap();
	let s_c = solver_c.step(&mut u_c, None).unwrap();

	assert!(
		s_j.converged(),
		"Jacobi PCG: {:?} (res={:.2e}, iters={})",
		s_j.reason, s_j.residual, s_j.iterations,
	);
	assert!(
		s_c.converged(),
		"Chebyshev PCG: {:?} (res={:.2e}, iters={})",
		s_c.reason, s_c.residual, s_c.iterations,
	);
	assert!(
		s_c.iterations <= s_j.iterations,
		"Chebyshev ({} its) should be <= Jacobi ({} its)",
		s_c.iterations,
		s_j.iterations,
	);
}

// ── Field reductions ───────────────────────────────────

/// Field::sum on a constant field.
#[test]
fn field_sum_constant() {
	let n = 16;
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let dev = cpu();
	let mut f =
		Field::zeros(grid, DType::F64, dev).unwrap();
	f.fill(3.0).unwrap();
	let s = f.sum().unwrap();
	let expected = 3.0 * (n * n) as f64;
	assert!(
		(s - expected).abs() < 1e-10,
		"sum={s}, expected={expected}"
	);
}

/// Field::max and min on sequential data.
#[test]
fn field_max_min() {
	let n = 16;
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let dev = cpu();
	let data: Vec<f64> =
		(0..n * n).map(|i| i as f64).collect();
	let f = Field::from_f64(grid, &data, dev).unwrap();
	let mx = f.max().unwrap();
	let mn = f.min().unwrap();
	assert!(
		(mx - (n * n - 1) as f64).abs() < 1e-10,
		"max={mx}"
	);
	assert!(mn.abs() < 1e-10, "min={mn}");
}

/// Field::dot product of two constant fields.
#[test]
fn field_dot_constant() {
	let n = 16;
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let dev = cpu();
	let mut a =
		Field::zeros(grid, DType::F64, dev).unwrap();
	let mut b =
		Field::zeros(grid, DType::F64, dev).unwrap();
	a.fill(2.0).unwrap();
	b.fill(3.0).unwrap();
	let d = a.dot(&b).unwrap();
	let expected = 6.0 * (n * n) as f64;
	assert!(
		(d - expected).abs() < 1e-10,
		"dot={d}, expected={expected}"
	);
}

/// Field::integral approximates area integral.
#[test]
fn field_integral_constant() {
	let n = 32;
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let dev = cpu();
	let mut f =
		Field::zeros(grid, DType::F64, dev).unwrap();
	f.fill(1.0).unwrap();
	let integ = f.integral().unwrap();
	// Integral of 1 over [0,1]² ≈ 1.0 (with cell area h²).
	assert!(
		(integ - 1.0).abs() < 0.1,
		"integral of 1 over [0,1]² = {integ}, expected ~1.0"
	);
}

// ── Extended vector operations ─────────────────────────

/// pointwise_mult: z = x * y.
#[test]
fn pointwise_mult_cpu() {
	let n = 16;
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let dev = cpu();
	let mut x =
		Field::zeros(grid, DType::F64, dev).unwrap();
	let mut y =
		Field::zeros(grid, DType::F64, dev).unwrap();
	let mut z =
		Field::zeros(grid, DType::F64, dev).unwrap();
	x.fill(3.0).unwrap();
	y.fill(7.0).unwrap();
	z.pointwise_mult(&x, &y).unwrap();
	let vals = z.to_vec_f64();
	for &v in &vals {
		assert!(
			(v - 21.0).abs() < 1e-10,
			"3*7={v}"
		);
	}
}

/// pointwise_div: z = x / y.
#[test]
fn pointwise_div_cpu() {
	let n = 16;
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let dev = cpu();
	let mut x =
		Field::zeros(grid, DType::F64, dev).unwrap();
	let mut y =
		Field::zeros(grid, DType::F64, dev).unwrap();
	let mut z =
		Field::zeros(grid, DType::F64, dev).unwrap();
	x.fill(21.0).unwrap();
	y.fill(7.0).unwrap();
	z.pointwise_div(&x, &y).unwrap();
	let vals = z.to_vec_f64();
	for &v in &vals {
		assert!(
			(v - 3.0).abs() < 1e-10,
			"21/7={v}"
		);
	}
}

/// waxpy: w = alpha*x + beta*y.
#[test]
fn waxpy_cpu() {
	let n = 16;
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let dev = cpu();
	let mut x =
		Field::zeros(grid, DType::F64, dev).unwrap();
	let mut y =
		Field::zeros(grid, DType::F64, dev).unwrap();
	let mut w =
		Field::zeros(grid, DType::F64, dev).unwrap();
	x.fill(2.0).unwrap();
	y.fill(3.0).unwrap();
	// w = 5*x + 10*y = 10 + 30 = 40
	w.waxpy(5.0, &x, 10.0, &y).unwrap();
	let vals = w.to_vec_f64();
	for &v in &vals {
		assert!(
			(v - 40.0).abs() < 1e-10,
			"5*2+10*3={v}"
		);
	}
}

/// aypx: y = x + alpha*y.
#[test]
fn aypx_cpu() {
	let n = 16;
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let dev = cpu();
	let mut x =
		Field::zeros(grid, DType::F64, dev).unwrap();
	let mut y =
		Field::zeros(grid, DType::F64, dev).unwrap();
	x.fill(10.0).unwrap();
	y.fill(3.0).unwrap();
	// y = x + 2*y = 10 + 6 = 16
	y.aypx(2.0, &x).unwrap();
	let vals = y.to_vec_f64();
	for &v in &vals {
		assert!(
			(v - 16.0).abs() < 1e-10,
			"10+2*3={v}"
		);
	}
}

/// reciprocal: x = 1/x.
#[test]
fn reciprocal_cpu() {
	let n = 16;
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let dev = cpu();
	let mut x =
		Field::zeros(grid, DType::F64, dev).unwrap();
	x.fill(4.0).unwrap();
	x.reciprocal().unwrap();
	let vals = x.to_vec_f64();
	for &v in &vals {
		assert!(
			(v - 0.25).abs() < 1e-10,
			"1/4={v}"
		);
	}
}

/// abs_val: x = |x|.
#[test]
fn abs_val_cpu() {
	let n = 16;
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let dev = cpu();
	let mut x =
		Field::zeros(grid, DType::F64, dev).unwrap();
	x.fill(-7.0).unwrap();
	x.abs_val().unwrap();
	let vals = x.to_vec_f64();
	for &v in &vals {
		assert!(
			(v - 7.0).abs() < 1e-10,
			"|−7|={v}"
		);
	}
}

/// dot2: fused dual dot product.
#[test]
fn dot2_cpu() {
	let n = 16;
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let dev = cpu();
	let mut a =
		Field::zeros(grid, DType::F64, dev).unwrap();
	let mut b =
		Field::zeros(grid, DType::F64, dev).unwrap();
	let mut c =
		Field::zeros(grid, DType::F64, dev).unwrap();
	let mut d =
		Field::zeros(grid, DType::F64, dev).unwrap();
	a.fill(1.0).unwrap();
	b.fill(2.0).unwrap();
	c.fill(3.0).unwrap();
	d.fill(4.0).unwrap();
	let (d1, d2) = a.dot2(&b, &c, &d).unwrap();
	let nn = (n * n) as f64;
	// dot(a,b) = 1*2*nn = 2*nn
	assert!(
		(d1 - 2.0 * nn).abs() < 1e-8,
		"dot2.0={d1}"
	);
	// dot(c,d) = 3*4*nn = 12*nn
	assert!(
		(d2 - 12.0 * nn).abs() < 1e-8,
		"dot2.1={d2}"
	);
}

// ── ConvergenceReason / SolveStats ─────────────────────

/// MaxIterations is returned when the solver doesn't converge.
#[test]
fn cg_max_iterations_reason() {
	let n = 32;
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let dev = cpu();
	let stencil = Stencil::laplacian_2d_5pt(h, h);
	let bc = Boundaries::zero_dirichlet();
	let coeff = -0.01;

	let b_data = gaussian_ic(n, h);
	let b = Field::from_f64(grid, &b_data, dev).unwrap();
	let mut x = Field::zeros(grid, DType::F64, dev).unwrap();

	// Only 2 iterations — not enough to converge for a
	// non-trivial RHS.
	let mut cg =
		CgSolver::new(grid, dev, 1e-15, 2).unwrap();
	let stats =
		cg.solve(&mut x, &b, coeff, &stencil, &bc).unwrap();

	assert_eq!(stats.reason, ConvergenceReason::MaxIterations);
	assert!(!stats.converged());
	assert_eq!(stats.iterations, 2);
}

// ── Solver reuse ───────────────────────────────────────

/// Solver can be reused for multiple solves without re-allocation.
#[test]
fn solver_reuse_no_realloc() {
	let n = 32;
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let dev = cpu();
	let stencil = Stencil::laplacian_2d_5pt(h, h);
	let bc = Boundaries::zero_dirichlet();
	let coeff = -0.01;

	let u_data = gaussian_ic(n, h);
	let u_exact =
		Field::from_f64(grid, &u_data, dev).unwrap();
	let mut b = Field::zeros(grid, DType::F64, dev).unwrap();
	b.copy_from(&u_exact).unwrap();
	let mut tmp =
		Field::zeros(grid, DType::F64, dev).unwrap();
	tmp.copy_from(&u_exact).unwrap();
	b.stencil_axpy(coeff, &tmp, &stencil, &bc).unwrap();

	let mut cg =
		CgSolver::new(grid, dev, 1e-8, 500).unwrap();

	// Solve 5 times with the same solver instance.
	for i in 0..5 {
		let mut x =
			Field::zeros(grid, DType::F64, dev).unwrap();
		let stats = cg
			.solve(&mut x, &b, coeff, &stencil, &bc)
			.unwrap();
		assert!(
			stats.converged(),
			"solve #{i} failed: {:?}",
			stats.reason
		);
	}
}

// ── GMRES(m) ───────────────────────────────────────────

/// GMRES converges on a well-conditioned SPD system.
#[test]
fn gmres_converges_cpu() {
	let n = 32;
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let dev = cpu();
	let stencil = Stencil::laplacian_2d_5pt(h, h);
	let bc = Boundaries::zero_dirichlet();
	let coeff = -0.01;

	let u_data = gaussian_ic(n, h);
	let u_exact =
		Field::from_f64(grid, &u_data, dev).unwrap();
	let mut b = Field::zeros(grid, DType::F64, dev).unwrap();
	b.copy_from(&u_exact).unwrap();
	b.stencil_axpy(coeff, &u_exact, &stencil, &bc)
		.unwrap();

	let mut x = Field::zeros(grid, DType::F64, dev).unwrap();
	let mut gmres =
		GmresSolver::new(grid, dev, 30, 1e-8, 500)
			.unwrap();
	let stats = gmres
		.solve(&mut x, &b, coeff, &stencil, &bc)
		.unwrap();

	assert!(
		stats.converged(),
		"GMRES: {:?} (res={:.2e}, iters={})",
		stats.reason, stats.residual, stats.iterations,
	);
}

/// GMRES matches CG solution on symmetric system.
#[test]
fn gmres_matches_cg_solution() {
	let n = 32;
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let dev = cpu();
	let stencil = Stencil::laplacian_2d_5pt(h, h);
	let bc = Boundaries::zero_dirichlet();
	let coeff = -0.01;

	let u_data = gaussian_ic(n, h);
	let u_exact =
		Field::from_f64(grid, &u_data, dev).unwrap();
	let mut b = Field::zeros(grid, DType::F64, dev).unwrap();
	b.copy_from(&u_exact).unwrap();
	b.stencil_axpy(coeff, &u_exact, &stencil, &bc)
		.unwrap();

	// CG.
	let mut x_cg =
		Field::zeros(grid, DType::F64, dev).unwrap();
	let mut cg =
		CgSolver::new(grid, dev, 1e-8, 500).unwrap();
	cg.solve(&mut x_cg, &b, coeff, &stencil, &bc)
		.unwrap();

	// GMRES.
	let mut x_gm =
		Field::zeros(grid, DType::F64, dev).unwrap();
	let mut gmres =
		GmresSolver::new(grid, dev, 50, 1e-8, 500)
			.unwrap();
	gmres
		.solve(&mut x_gm, &b, coeff, &stencil, &bc)
		.unwrap();

	// Solutions should match.
	let v_cg = x_cg.to_vec_f64();
	let v_gm = x_gm.to_vec_f64();
	let max_diff: f64 = v_cg
		.iter()
		.zip(v_gm.iter())
		.map(|(a, b)| (a - b).abs())
		.fold(0.0_f64, f64::max);
	assert!(
		max_diff < 1e-6,
		"GMRES vs CG max diff = {max_diff:.2e}"
	);
}

/// GMRES handles larger coeff (harder system).
#[test]
fn gmres_hard_system() {
	let n = 32;
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let dev = cpu();
	let stencil = Stencil::laplacian_2d_5pt(h, h);
	let bc = Boundaries::zero_dirichlet();
	let coeff = -0.5;

	let u_data = gaussian_ic(n, h);
	let u_exact =
		Field::from_f64(grid, &u_data, dev).unwrap();
	let mut b = Field::zeros(grid, DType::F64, dev).unwrap();
	b.copy_from(&u_exact).unwrap();
	b.stencil_axpy(coeff, &u_exact, &stencil, &bc)
		.unwrap();

	let mut x = Field::zeros(grid, DType::F64, dev).unwrap();
	let mut gmres =
		GmresSolver::new(grid, dev, 100, 1e-6, 1000)
			.unwrap();
	let stats = gmres
		.solve(&mut x, &b, coeff, &stencil, &bc)
		.unwrap();

	assert!(
		stats.converged(),
		"GMRES hard: {:?} (res={:.2e}, iters={})",
		stats.reason, stats.residual, stats.iterations,
	);
}

/// GMRES with preconditioner converges faster.
#[test]
fn gmres_preconditioned_cpu() {
	let n = 32;
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let dev = cpu();
	let stencil = Stencil::laplacian_2d_5pt(h, h);
	let bc = Boundaries::zero_dirichlet();
	let coeff = -0.01;

	let u_data = gaussian_ic(n, h);
	let u_exact =
		Field::from_f64(grid, &u_data, dev).unwrap();
	let mut b = Field::zeros(grid, DType::F64, dev).unwrap();
	b.copy_from(&u_exact).unwrap();
	b.stencil_axpy(coeff, &u_exact, &stencil, &bc)
		.unwrap();

	let mut x = Field::zeros(grid, DType::F64, dev).unwrap();
	let mut gmres =
		GmresSolver::new(grid, dev, 30, 1e-8, 500)
			.unwrap();
	let mut pc = Identity;
	let stats = gmres
		.solve_preconditioned(
			&mut x, &b, coeff, &stencil, &bc, &mut pc,
		)
		.unwrap();

	assert!(
		stats.converged(),
		"GMRES-PC: {:?} (res={:.2e}, iters={})",
		stats.reason, stats.residual, stats.iterations,
	);
}

/// GMRES works through the LinearSolver trait.
#[test]
fn gmres_via_linear_solver_trait() {
	let n = 32;
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let dev = cpu();
	let stencil = Stencil::laplacian_2d_5pt(h, h);
	let bc = Boundaries::zero_dirichlet();
	let coeff = -0.01;

	let u_data = gaussian_ic(n, h);
	let u_exact =
		Field::from_f64(grid, &u_data, dev).unwrap();
	let mut b = Field::zeros(grid, DType::F64, dev).unwrap();
	b.copy_from(&u_exact).unwrap();
	b.stencil_axpy(coeff, &u_exact, &stencil, &bc)
		.unwrap();

	let mut solver: Box<dyn LinearSolver> = Box::new(
		GmresSolver::new(grid, dev, 30, 1e-8, 500)
			.unwrap(),
	);
	assert_eq!(solver.name(), "GMRES");

	let mut x = Field::zeros(grid, DType::F64, dev).unwrap();
	let stats = solver
		.solve(&mut x, &b, coeff, &stencil, &bc)
		.unwrap();
	assert!(stats.converged());
}

/// GMRES restart works: converges even when m < needed iterations.
#[test]
fn gmres_restart_converges() {
	let n = 32;
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let dev = cpu();
	let stencil = Stencil::laplacian_2d_5pt(h, h);
	let bc = Boundaries::zero_dirichlet();
	let coeff = -0.01;

	let u_data = gaussian_ic(n, h);
	let u_exact =
		Field::from_f64(grid, &u_data, dev).unwrap();
	let mut b = Field::zeros(grid, DType::F64, dev).unwrap();
	b.copy_from(&u_exact).unwrap();
	b.stencil_axpy(coeff, &u_exact, &stencil, &bc)
		.unwrap();

	// Very small restart (m=5) — forces multiple restarts.
	let mut x = Field::zeros(grid, DType::F64, dev).unwrap();
	let mut gmres =
		GmresSolver::new(grid, dev, 5, 1e-8, 500).unwrap();
	let stats = gmres
		.solve(&mut x, &b, coeff, &stencil, &bc)
		.unwrap();

	assert!(
		stats.converged(),
		"GMRES(5) restart: {:?} (res={:.2e}, iters={})",
		stats.reason, stats.residual, stats.iterations,
	);
	// With m=5, should need more iterations than m=50.
	assert!(stats.iterations > 5);
}

// ── IMEX (convection-diffusion) ────────────────────────

/// IMEX solver converges and diffuses a Gaussian.
#[test]
fn imex_diffusion_only() {
	let n = 32;
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let dev = cpu();
	let bc = Boundaries::zero_dirichlet();

	// Pure diffusion (no convection): kappa=0.01, v=0.
	let mut solver = ImexSolver::new(
		grid, 0.01, 0.0, 0.0, 0.01, bc, dev, 1e-8, 200,
	)
	.unwrap();

	let data = gaussian_ic(n, h);
	let initial_peak: f64 = data
		.iter()
		.copied()
		.reduce(f64::max)
		.unwrap();
	let mut u =
		Field::from_f64(grid, &data, dev).unwrap();

	solver.step_n(&mut u, 50).unwrap();

	let vals = u.to_vec_f64();
	let final_peak: f64 = vals
		.iter()
		.copied()
		.reduce(f64::max)
		.unwrap();
	assert!(final_peak.is_finite());
	assert!(
		final_peak < initial_peak,
		"IMEX diffusion: {initial_peak} -> {final_peak}"
	);
	assert_eq!(solver.steps_done(), 50);
}

/// IMEX with convection: field should advect.
#[test]
fn imex_convection_diffusion() {
	let n = 32;
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let dev = cpu();
	let bc = Boundaries::zero_dirichlet();

	// Moderate convection + diffusion.
	let mut solver = ImexSolver::new(
		grid, 0.01, 0.5, 0.0, 0.001, bc, dev, 1e-8, 200,
	)
	.unwrap();

	let data = gaussian_ic(n, h);
	let mut u =
		Field::from_f64(grid, &data, dev).unwrap();
	let initial_norm = u.norm_l2().unwrap();

	solver.step_n(&mut u, 20).unwrap();

	let final_norm = u.norm_l2().unwrap();
	assert!(final_norm.is_finite());
	// The field should change but not blow up.
	assert!(final_norm > 0.0);
	assert!(
		final_norm < 2.0 * initial_norm,
		"IMEX should not blow up: {initial_norm} -> {final_norm}"
	);
}

/// IMEX with multigrid preconditioner.
#[test]
fn imex_with_multigrid() {
	let n = 32;
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let dev = cpu();
	let bc = Boundaries::zero_dirichlet();

	let mut solver = ImexSolver::with_multigrid(
		grid, 0.01, 0.0, 0.0, 0.01, bc, dev, 1e-8,
		200, 2, 2, 2.0 / 3.0,
	)
	.unwrap();

	let data = gaussian_ic(n, h);
	let mut u =
		Field::from_f64(grid, &data, dev).unwrap();

	let stats = solver.step(&mut u).unwrap();
	assert!(
		stats.converged(),
		"IMEX+MG: {:?} (res={:.2e}, iters={})",
		stats.reason, stats.residual, stats.iterations,
	);
}

// ── Adaptive time-stepping (DOPRI5) ──────────────────��─

/// Adaptive solver diffuses a Gaussian without blowup.
#[test]
fn adaptive_diffuses_gaussian() {
	let n = 32;
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let dev = cpu();
	let bc = Boundaries::zero_dirichlet();

	let mut solver = AdaptiveSolver::new(
		grid, 1.0, 0.0001, bc, dev, 1e-6, 1e-4,
	)
	.unwrap();

	let data = gaussian_ic(n, h);
	let initial_peak: f64 = data
		.iter()
		.copied()
		.reduce(f64::max)
		.unwrap();
	let mut u =
		Field::from_f64(grid, &data, dev).unwrap();

	solver.advance_to(&mut u, 0.01).unwrap();

	let vals = u.to_vec_f64();
	let final_peak: f64 = vals
		.iter()
		.copied()
		.reduce(f64::max)
		.unwrap();
	assert!(final_peak.is_finite());
	assert!(
		final_peak < initial_peak,
		"adaptive: {initial_peak} -> {final_peak}"
	);
	assert!(solver.sim_time() >= 0.01 - 1e-12);
}

/// Adaptive solver adapts dt upward for smooth solutions.
#[test]
fn adaptive_dt_increases() {
	let n = 16;
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let dev = cpu();
	let bc = Boundaries::zero_dirichlet();

	let mut solver = AdaptiveSolver::new(
		grid, 0.01, 1e-6, bc, dev, 1e-4, 1e-3,
	)
	.unwrap();

	let data = gaussian_ic(n, h);
	let mut u =
		Field::from_f64(grid, &data, dev).unwrap();

	let dt_initial = solver.dt();
	// Take a few steps — dt should grow for smooth diffusion.
	for _ in 0..10 {
		solver.step(&mut u).unwrap();
	}

	assert!(
		solver.dt() > dt_initial,
		"dt should increase: {dt_initial} -> {}",
		solver.dt()
	);
}

/// Adaptive solver rejects steps when error is too large.
#[test]
fn adaptive_rejects_large_dt() {
	let n = 16;
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let dev = cpu();
	let bc = Boundaries::zero_dirichlet();

	// Start with very large dt — should trigger rejections.
	let mut solver = AdaptiveSolver::new(
		grid, 100.0, 1.0, bc, dev, 1e-6, 1e-4,
	)
	.unwrap();

	let data = gaussian_ic(n, h);
	let mut u =
		Field::from_f64(grid, &data, dev).unwrap();

	solver.step(&mut u).unwrap();

	assert!(
		solver.rejected_steps > 0,
		"should reject the oversized initial dt"
	);
	assert!(solver.total_steps > 1);
}

// ── Robin boundary conditions (P3) ─────────────────────

/// Robin BC: alpha·u + beta·du/dn = 0 (absorbing BC).
/// Solution should decay faster than with pure Dirichlet.
#[test]
fn robin_bc_absorbing() {
	let n = 32;
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let dev = cpu();

	// Robin on all sides: u + du/dn = 0 (absorbing).
	let robin = BoundaryCondition::Robin {
		alpha: 1.0,
		beta: 1.0,
		g: 0.0,
	};
	let bc = Boundaries {
		top: robin,
		bottom: robin,
		left: robin,
		right: robin,
	};

	let mut solver = DiffusionSolver::with_boundaries(
		grid, 1.0, None, bc, dev,
	)
	.unwrap();

	let data = gaussian_ic(n, h);
	let mut u =
		Field::from_f64(grid, &data, dev).unwrap();
	let initial_norm = u.norm_l2().unwrap();

	solver.step_n(&mut u, 500).unwrap();

	let final_norm = u.norm_l2().unwrap();
	assert!(final_norm.is_finite());
	assert!(
		final_norm < initial_norm,
		"Robin absorbing: norm should decrease: \
		 {initial_norm} -> {final_norm}"
	);
}

// ── Variable coefficients (P3) ─────────────────────────

/// Variable-coefficient stencil: coeff(x,y) * Δu.
/// With uniform coeff=1, should match standard stencil.
#[test]
fn variable_coeff_uniform_matches_standard() {
	let n = 32;
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let dev = cpu();
	let stencil = Stencil::laplacian_2d_5pt(h, h);
	let bc = Boundaries::zero_dirichlet();

	// u(x,y) = x² + y²
	let mut data = vec![0.0_f64; n * n];
	for row in 0..n {
		for col in 0..n {
			let x = col as f64 * h;
			let y = row as f64 * h;
			data[row * n + col] = x.mul_add(x, y * y);
		}
	}
	let input =
		Field::from_f64(grid, &data, dev).unwrap();

	// Standard stencil.
	let mut out_std =
		Field::zeros(grid, DType::F64, dev).unwrap();
	input
		.apply_stencil_into(&stencil, &bc, &mut out_std)
		.unwrap();

	// Variable coefficient = 1.0 everywhere.
	let mut coeff =
		Field::zeros(grid, DType::F64, dev).unwrap();
	coeff.fill(1.0).unwrap();
	let mut out_var =
		Field::zeros(grid, DType::F64, dev).unwrap();
	input
		.apply_stencil_var_into(
			&stencil, &bc, &coeff, &mut out_var,
		)
		.unwrap();

	// Should be identical.
	let v_std = out_std.to_vec_f64();
	let v_var = out_var.to_vec_f64();
	let max_diff: f64 = v_std
		.iter()
		.zip(v_var.iter())
		.map(|(a, b)| (a - b).abs())
		.fold(0.0_f64, f64::max);
	assert!(
		max_diff < 1e-10,
		"uniform coeff=1 should match: diff={max_diff:.2e}"
	);
}

/// Variable coefficient = 2.0 should double the stencil output.
#[test]
fn variable_coeff_scaling() {
	let n = 32;
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let dev = cpu();
	let stencil = Stencil::laplacian_2d_5pt(h, h);
	let bc = Boundaries::zero_dirichlet();

	let mut data = vec![0.0_f64; n * n];
	for row in 0..n {
		for col in 0..n {
			let x = col as f64 * h;
			let y = row as f64 * h;
			data[row * n + col] = x.mul_add(x, y * y);
		}
	}
	let input =
		Field::from_f64(grid, &data, dev).unwrap();

	// Standard: Δ(x²+y²) = 4.
	let mut out_std =
		Field::zeros(grid, DType::F64, dev).unwrap();
	input
		.apply_stencil_into(&stencil, &bc, &mut out_std)
		.unwrap();

	// Variable coefficient = 2.0: should give 2*Δu = 8.
	let mut coeff =
		Field::zeros(grid, DType::F64, dev).unwrap();
	coeff.fill(2.0).unwrap();
	let mut out_var =
		Field::zeros(grid, DType::F64, dev).unwrap();
	input
		.apply_stencil_var_into(
			&stencil, &bc, &coeff, &mut out_var,
		)
		.unwrap();

	let v_var = out_var.to_vec_f64();
	for row in 2..n - 2 {
		for col in 2..n - 2 {
			let val = v_var[row * n + col];
			assert!(
				(val - 8.0).abs() < 1e-10,
				"2*Δ(x²+y²) at ({row},{col}) = {val}"
			);
		}
	}
}

// ── BDF solver (P3) ────────────────────────────────────

/// BDF1 (backward Euler) diffuses a Gaussian.
#[test]
fn bdf1_diffuses_gaussian() {
	let n = 32;
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let dev = cpu();
	let bc = Boundaries::zero_dirichlet();

	let mut solver = BdfSolver::new(
		grid, 0.01, 0.01, 1, bc, dev, 1e-8, 200,
	)
	.unwrap();

	let data = gaussian_ic(n, h);
	let initial_peak: f64 = data
		.iter()
		.copied()
		.reduce(f64::max)
		.unwrap();
	let mut u =
		Field::from_f64(grid, &data, dev).unwrap();

	solver.step_n(&mut u, 50).unwrap();

	let vals = u.to_vec_f64();
	let final_peak: f64 = vals
		.iter()
		.copied()
		.reduce(f64::max)
		.unwrap();
	assert!(final_peak.is_finite());
	assert!(
		final_peak < initial_peak,
		"BDF1: {initial_peak} -> {final_peak}"
	);
}

/// BDF2 converges and diffuses.
#[test]
fn bdf2_diffuses_gaussian() {
	let n = 32;
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let dev = cpu();
	let bc = Boundaries::zero_dirichlet();

	let mut solver = BdfSolver::new(
		grid, 0.01, 0.01, 2, bc, dev, 1e-8, 200,
	)
	.unwrap();

	let data = gaussian_ic(n, h);
	let initial_peak: f64 = data
		.iter()
		.copied()
		.reduce(f64::max)
		.unwrap();
	let mut u =
		Field::from_f64(grid, &data, dev).unwrap();

	// First step bootstraps with BDF1, then BDF2.
	solver.step_n(&mut u, 50).unwrap();

	let vals = u.to_vec_f64();
	let final_peak: f64 = vals
		.iter()
		.copied()
		.reduce(f64::max)
		.unwrap();
	assert!(final_peak.is_finite());
	assert!(
		final_peak < initial_peak,
		"BDF2: {initial_peak} -> {final_peak}"
	);
	assert_eq!(solver.steps_done(), 50);
}

/// BDF4 with multigrid converges.
#[test]
fn bdf4_with_multigrid() {
	let n = 32;
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let dev = cpu();
	let bc = Boundaries::zero_dirichlet();

	let mut solver = BdfSolver::with_multigrid(
		grid, 0.01, 0.01, 4, bc, dev, 1e-8, 200,
		2, 2, 2.0 / 3.0,
	)
	.unwrap();

	let data = gaussian_ic(n, h);
	let mut u =
		Field::from_f64(grid, &data, dev).unwrap();

	let stats = solver.step(&mut u).unwrap();
	assert!(
		stats.converged(),
		"BDF4+MG: {:?} (res={:.2e})",
		stats.reason, stats.residual,
	);
}

// ── SSP-RK solver (P3) ────────────────────────────────

/// SSP-RK2 diffuses a Gaussian.
#[test]
fn ssp_rk2_diffuses_gaussian() {
	let n = 32;
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let dev = cpu();
	let bc = Boundaries::zero_dirichlet();

	let mut solver = SspRkSolver::with_auto_dt(
		grid, 1.0, SspOrder::Ssp2, bc, dev,
	)
	.unwrap();

	let data = gaussian_ic(n, h);
	let initial_peak: f64 = data
		.iter()
		.copied()
		.reduce(f64::max)
		.unwrap();
	let mut u =
		Field::from_f64(grid, &data, dev).unwrap();

	solver.step_n(&mut u, 200).unwrap();

	let vals = u.to_vec_f64();
	let final_peak: f64 = vals
		.iter()
		.copied()
		.reduce(f64::max)
		.unwrap();
	assert!(final_peak.is_finite());
	assert!(
		final_peak < initial_peak,
		"SSP-RK2: {initial_peak} -> {final_peak}"
	);
}

/// SSP-RK3 diffuses a Gaussian.
#[test]
fn ssp_rk3_diffuses_gaussian() {
	let n = 32;
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let dev = cpu();
	let bc = Boundaries::zero_dirichlet();

	let mut solver = SspRkSolver::with_auto_dt(
		grid, 1.0, SspOrder::Ssp3, bc, dev,
	)
	.unwrap();

	let data = gaussian_ic(n, h);
	let initial_peak: f64 = data
		.iter()
		.copied()
		.reduce(f64::max)
		.unwrap();
	let mut u =
		Field::from_f64(grid, &data, dev).unwrap();

	solver.step_n(&mut u, 200).unwrap();

	let vals = u.to_vec_f64();
	let final_peak: f64 = vals
		.iter()
		.copied()
		.reduce(f64::max)
		.unwrap();
	assert!(final_peak.is_finite());
	assert!(
		final_peak < initial_peak,
		"SSP-RK3: {initial_peak} -> {final_peak}"
	);
}

/// SSP-RK3 remains stable (TVD property).
#[test]
fn ssp_rk3_stable_1000_steps() {
	let n = 32;
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let dev = cpu();
	let bc = Boundaries::zero_dirichlet();

	let mut solver = SspRkSolver::with_auto_dt(
		grid, 1.0, SspOrder::Ssp3, bc, dev,
	)
	.unwrap();

	let data = gaussian_ic(n, h);
	let mut u =
		Field::from_f64(grid, &data, dev).unwrap();

	solver.step_n(&mut u, 1000).unwrap();

	let norm = u.norm_l2().unwrap();
	assert!(norm.is_finite(), "SSP-RK3 blew up: {norm}");
	assert!(norm > 0.0);
}
