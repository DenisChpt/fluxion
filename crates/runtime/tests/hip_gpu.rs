//! Integration tests for the HIP backend on real AMD GPUs.
//!
//! IMPORTANT: Run sequentially to avoid stream interleaving:
//!   cargo test --features hip --test hip_gpu -- --test-threads=1

#![cfg(feature = "hip")]

use fluxion_core::{Boundaries, DType, Grid, Stencil};
use fluxion_runtime::{
	AdaptiveSolver, BiCgStabSolver, CgSolver,
	CrankNicolsonSolver, Device, Field, GmresSolver,
	ImexSolver, PipelinedCgSolver,
};

fn hip_device() -> Device {
	Device::Hip { ordinal: 0 }
}

fn gaussian(n: usize, h: f64) -> Vec<f64> {
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

#[test]
fn hip_allocate_and_readback() {
	let grid = Grid::square(16, 1.0 / 15.0).unwrap();
	let device = hip_device();

	let f =
		Field::zeros(grid, DType::F64, device).unwrap();
	let data = f.to_vec_f64();
	assert_eq!(data.len(), 16 * 16);
	assert!(data.iter().all(|&v| v == 0.0));
}

#[test]
fn hip_upload_and_readback() {
	let n = 32;
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let device = hip_device();

	let src: Vec<f64> =
		(0..n * n).map(|i| i as f64).collect();
	let f =
		Field::from_f64(grid, &src, device).unwrap();
	let data = f.to_vec_f64();
	for (i, (&a, &b)) in
		src.iter().zip(data.iter()).enumerate()
	{
		assert!(
			(a - b).abs() < 1e-12,
			"mismatch at {i}: {a} vs {b}"
		);
	}
}

#[test]
fn hip_fill_and_norm() {
	let n = 64;
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let device = hip_device();

	let mut f =
		Field::zeros(grid, DType::F64, device).unwrap();
	f.fill(3.0).unwrap();
	let norm = f.norm_l2().unwrap();
	// norm = sqrt(n*n * 9) = 3*n
	let expected = 3.0 * (n as f64);
	assert!(
		(norm - expected).abs() < 1e-8,
		"norm = {norm}, expected {expected}"
	);
}

#[test]
fn hip_axpy() {
	let n = 128;
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let device = hip_device();

	let ones = vec![1.0_f64; n * n];
	let x =
		Field::from_f64(grid, &ones, device).unwrap();
	let mut y =
		Field::from_f64(grid, &vec![10.0; n * n], device)
			.unwrap();
	y.axpy(2.0, &x).unwrap();

	let data = y.to_vec_f64();
	assert!(data.iter().all(|&v| (v - 12.0).abs() < 1e-12));
}

#[test]
fn hip_stencil_quadratic() {
	let n = 32;
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let stencil = Stencil::laplacian_2d_5pt(h, h);
	let bcs = Boundaries::zero_dirichlet();
	let device = hip_device();

	let mut data = vec![0.0_f64; n * n];
	for row in 0..n {
		for col in 0..n {
			let x = col as f64 * h;
			let y = row as f64 * h;
			data[row * n + col] = x * x + y * y;
		}
	}

	let input =
		Field::from_f64(grid, &data, device).unwrap();
	let mut output =
		Field::zeros(grid, DType::F64, device).unwrap();
	input
		.apply_stencil_into(&stencil, &bcs, &mut output)
		.unwrap();

	let out = output.to_vec_f64();
	// Interior points: Δ(x²+y²) = 4.
	for row in 2..n - 2 {
		for col in 2..n - 2 {
			let val = out[row * n + col];
			assert!(
				(val - 4.0).abs() < 1e-8,
				"Δu at ({row},{col}) = {val}, expected 4.0"
			);
		}
	}
}

#[test]
fn hip_diffusion_decreases_peak() {
	use fluxion_runtime::DiffusionSolver;

	let n = 64;
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let device = hip_device();
	let sigma = 0.1_f64;

	let mut data = vec![0.0_f64; n * n];
	for row in 0..n {
		for col in 0..n {
			let x = col as f64 * h;
			let y = row as f64 * h;
			let dx = x - 0.5;
			let dy = y - 0.5;
			let r2 = dx.mul_add(dx, dy * dy);
			data[row * n + col] =
				(-r2 / (2.0 * sigma * sigma)).exp();
		}
	}

	let initial_peak: f64 = data
		.iter()
		.copied()
		.reduce(f64::max)
		.unwrap();

	let mut u =
		Field::from_f64(grid, &data, device).unwrap();
	let mut solver = DiffusionSolver::new(
		grid, 0.01, None, device,
	)
	.unwrap();

	solver.step_n(&mut u, 100).unwrap();
	let final_peak = u.max().unwrap();

	assert!(
		final_peak < initial_peak,
		"peak should decrease: {final_peak} >= {initial_peak}"
	);
	assert!(final_peak > 0.0);
}

#[test]
fn hip_dot_product() {
	let n = 256;
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let device = hip_device();

	let ones = vec![1.0_f64; n * n];
	let threes = vec![3.0_f64; n * n];

	let a =
		Field::from_f64(grid, &ones, device).unwrap();
	let b =
		Field::from_f64(grid, &threes, device).unwrap();

	let dot = a.dot(&b).unwrap();
	let expected = 3.0 * (n * n) as f64;
	assert!(
		(dot - expected).abs() < 1e-6,
		"dot = {dot}, expected {expected}"
	);
}

#[test]
fn hip_scale() {
	let n = 64;
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let device = hip_device();

	let mut f = Field::from_f64(
		grid,
		&vec![4.0; n * n],
		device,
	)
	.unwrap();
	f.scale(0.5).unwrap();

	let data = f.to_vec_f64();
	assert!(data.iter().all(|&v| (v - 2.0).abs() < 1e-12));
}

#[test]
fn hip_reduce_sum_max_min() {
	let n = 64;
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let device = hip_device();

	let data: Vec<f64> =
		(0..n * n).map(|i| i as f64).collect();
	let f =
		Field::from_f64(grid, &data, device).unwrap();

	let sum = f.sum().unwrap();
	let expected_sum =
		(n * n - 1) as f64 * (n * n) as f64 / 2.0;
	assert!(
		(sum - expected_sum).abs() < 1e-4,
		"sum = {sum}, expected {expected_sum}"
	);

	let max = f.max().unwrap();
	assert!(
		(max - (n * n - 1) as f64).abs() < 1e-12,
		"max = {max}"
	);

	let min = f.min().unwrap();
	assert!((min - 0.0).abs() < 1e-12, "min = {min}");
}

#[test]
fn hip_crank_nicolson_pcg_multigrid() {
	let n = 64;
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let device = hip_device();
	let bcs = Boundaries::zero_dirichlet();
	let sigma = 0.1_f64;

	// Gaussian initial condition.
	let mut data = vec![0.0_f64; n * n];
	for row in 0..n {
		for col in 0..n {
			let x = col as f64 * h;
			let y = row as f64 * h;
			let dx = x - 0.5;
			let dy = y - 0.5;
			let r2 = dx.mul_add(dx, dy * dy);
			data[row * n + col] =
				(-r2 / (2.0 * sigma * sigma)).exp();
		}
	}

	let mut u =
		Field::from_f64(grid, &data, device).unwrap();
	let initial_peak = u.max().unwrap();

	// PCG with multigrid preconditioner.
	// Large dt + alpha to make the system stiff enough
	// that multigrid preconditioning actually helps.
	let alpha = 1.0;
	let dt = 0.1;
	let mut solver = CrankNicolsonSolver::with_multigrid(
		grid, alpha, dt, bcs, device, 1e-8, 100,
		2, 2, 2.0 / 3.0,
	)
	.unwrap();

	let stats = solver.step(&mut u, None).unwrap();
	assert!(
		stats.converged(),
		"PCG did not converge: {:?}", stats
	);

	// Also run unpreconditioned CG to compare iterations.
	let mut u2 =
		Field::from_f64(grid, &data, device).unwrap();
	let mut solver_plain = CrankNicolsonSolver::new(
		grid, alpha, dt, Boundaries::zero_dirichlet(),
		device, 1e-8, 200,
	)
	.unwrap();
	let stats_plain =
		solver_plain.step(&mut u2, None).unwrap();

	eprintln!(
		"  PCG: {} its (res {:.2e})  |  CG: {} its (res {:.2e})",
		stats.iterations, stats.residual,
		stats_plain.iterations, stats_plain.residual,
	);

	// PCG should need fewer iterations than plain CG.
	assert!(
		stats.iterations <= stats_plain.iterations,
		"PCG ({} its) should beat CG ({} its)",
		stats.iterations, stats_plain.iterations
	);

	// Run a few more steps.
	solver.step_n(&mut u, 4).unwrap();
	let final_peak = u.max().unwrap();

	assert!(
		final_peak < initial_peak,
		"peak should decrease: {final_peak} >= {initial_peak}"
	);
	assert!(final_peak > 0.0);
}

#[test]
fn hip_multigrid_v_cycle() {
	use fluxion_runtime::Multigrid;

	let n = 32;
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let device = hip_device();
	let bcs = Boundaries::zero_dirichlet();

	let mut mg = Multigrid::new(
		grid, bcs, device, 2, 2, 2.0 / 3.0,
	)
	.unwrap();

	// RHS = 1 everywhere.
	let rhs_data = vec![1.0_f64; n * n];
	let rhs =
		Field::from_f64(grid, &rhs_data, device).unwrap();
	let mut x =
		Field::zeros(grid, DType::F64, device).unwrap();

	// V-cycle should produce a non-zero correction.
	mg.v_cycle(&mut x, &rhs).unwrap();
	let norm = x.norm_l2().unwrap();
	assert!(
		norm > 1e-10,
		"V-cycle produced zero output: norm = {norm}"
	);
}

#[test]
fn hip_vecops() {
	let n = 64;
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let device = hip_device();

	let twos = vec![2.0_f64; n * n];
	let threes = vec![3.0_f64; n * n];

	let x =
		Field::from_f64(grid, &twos, device).unwrap();
	let y =
		Field::from_f64(grid, &threes, device).unwrap();
	let mut z =
		Field::zeros(grid, DType::F64, device).unwrap();

	// pointwise_mult: z = x * y = 6
	z.pointwise_mult(&x, &y).unwrap();
	let data = z.to_vec_f64();
	assert!(data.iter().all(|&v| (v - 6.0).abs() < 1e-12));

	// pointwise_div: z = y / x = 1.5
	z.pointwise_div(&y, &x).unwrap();
	let data = z.to_vec_f64();
	assert!(data.iter().all(|&v| (v - 1.5).abs() < 1e-12));

	// waxpy: z = 2*x + 3*y = 4 + 9 = 13
	z.waxpy(2.0, &x, 3.0, &y).unwrap();
	let data = z.to_vec_f64();
	assert!(data.iter().all(|&v| (v - 13.0).abs() < 1e-12));

	// aypx: z = x + 0.5*z = 2 + 6.5 = 8.5
	z.aypx(0.5, &x).unwrap();
	let data = z.to_vec_f64();
	assert!(data.iter().all(|&v| (v - 8.5).abs() < 1e-12));

	// reciprocal: z = 1/z = 1/8.5
	z.reciprocal().unwrap();
	let data = z.to_vec_f64();
	assert!(data
		.iter()
		.all(|&v| (v - 1.0 / 8.5).abs() < 1e-12));

	// abs_val on negative values
	let neg = vec![-5.0_f64; n * n];
	let mut a =
		Field::from_f64(grid, &neg, device).unwrap();
	a.abs_val().unwrap();
	let data = a.to_vec_f64();
	assert!(data.iter().all(|&v| (v - 5.0).abs() < 1e-12));
}

#[test]
fn hip_bicgstab_solves_laplacian() {
	let n = 64;
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let stencil = Stencil::laplacian_2d_5pt(h, h);
	let bcs = Boundaries::zero_dirichlet();
	let device = hip_device();

	// Set up RHS from a Gaussian.
	let sigma = 0.1_f64;
	let mut rhs_data = vec![0.0_f64; n * n];
	for row in 0..n {
		for col in 0..n {
			let x = col as f64 * h;
			let y = row as f64 * h;
			let dx = x - 0.5;
			let dy = y - 0.5;
			let r2 = dx.mul_add(dx, dy * dy);
			rhs_data[row * n + col] =
				(-r2 / (2.0 * sigma * sigma)).exp();
		}
	}

	let b =
		Field::from_f64(grid, &rhs_data, device).unwrap();
	let coeff = -0.5;

	// BiCGSTAB solve.
	let mut x_bcgs =
		Field::zeros(grid, DType::F64, device).unwrap();
	let mut bcgs = BiCgStabSolver::new(
		grid, device, 1e-8, 500,
	)
	.unwrap();
	let stats_bcgs = bcgs
		.solve(
			&mut x_bcgs, &b, coeff, &stencil, &bcs,
		)
		.unwrap();

	assert!(
		stats_bcgs.converged(),
		"BiCGSTAB did not converge: {:?}",
		stats_bcgs
	);

	// Verify: compute A*x and check it matches b.
	let mut ax =
		Field::zeros(grid, DType::F64, device).unwrap();
	ax.copy_from(&x_bcgs).unwrap();
	let mut tmp =
		Field::zeros(grid, DType::F64, device).unwrap();
	tmp.copy_from(&x_bcgs).unwrap();
	ax.stencil_axpy(coeff, &tmp, &stencil, &bcs)
		.unwrap();

	// residual = b - A*x
	ax.axpy(-1.0, &b).unwrap();
	ax.scale(-1.0).unwrap();
	let res_norm = ax.norm_l2().unwrap();
	let b_norm = b.norm_l2().unwrap();
	let rel_res = res_norm / b_norm;

	eprintln!(
		"BiCGSTAB: {} its, rel residual {:.2e}",
		stats_bcgs.iterations, rel_res
	);
	assert!(
		rel_res < 1e-4,
		"relative residual too large: {rel_res}"
	);
}

#[test]
fn hip_pipelined_cg_matches_standard() {
	let n = 64;
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let stencil = Stencil::laplacian_2d_5pt(h, h);
	let bcs = Boundaries::zero_dirichlet();
	let device = hip_device();
	// Crank-Nicolson-like coefficient: -(dt/2)*alpha.
	let coeff = -0.005;

	let sigma = 0.1_f64;
	let mut rhs_data = vec![0.0_f64; n * n];
	for row in 0..n {
		for col in 0..n {
			let x = col as f64 * h;
			let y = row as f64 * h;
			let dx = x - 0.5;
			let dy = y - 0.5;
			let r2 = dx.mul_add(dx, dy * dy);
			rhs_data[row * n + col] =
				(-r2 / (2.0 * sigma * sigma)).exp();
		}
	}
	let b =
		Field::from_f64(grid, &rhs_data, device).unwrap();

	// Standard CG.
	let mut x_cg =
		Field::zeros(grid, DType::F64, device).unwrap();
	let mut cg =
		CgSolver::new(grid, device, 1e-8, 500).unwrap();
	let stats_cg = cg
		.solve(&mut x_cg, &b, coeff, &stencil, &bcs)
		.unwrap();

	// Pipelined CG.
	let mut x_pcg =
		Field::zeros(grid, DType::F64, device).unwrap();
	let mut pcg = PipelinedCgSolver::new(
		grid, device, 1e-8, 500,
	)
	.unwrap();
	let stats_pcg = pcg
		.solve(&mut x_pcg, &b, coeff, &stencil, &bcs)
		.unwrap();

	eprintln!(
		"CG: {} its (res {:.2e}) | Pipelined: {} its (res {:.2e})",
		stats_cg.iterations, stats_cg.residual,
		stats_pcg.iterations, stats_pcg.residual,
	);

	assert!(
		stats_pcg.converged(),
		"Pipelined CG did not converge: {:?}",
		stats_pcg
	);

	// Both should produce similar solutions.
	let d_cg = x_cg.to_vec_f64();
	let d_pcg = x_pcg.to_vec_f64();
	let max_diff: f64 = d_cg
		.iter()
		.zip(d_pcg.iter())
		.map(|(a, b)| (a - b).abs())
		.reduce(f64::max)
		.unwrap();

	eprintln!("max diff CG vs pipelined: {max_diff:.2e}");
	assert!(
		max_diff < 1e-4,
		"solutions differ too much: {max_diff}"
	);
}

// ── GMRES on GPU ────────────────────────────────────

#[test]
fn hip_gmres_solves_laplacian() {
	let n = 64;
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let dev = hip_device();
	let stencil = Stencil::laplacian_2d_5pt(h, h);
	let bc = Boundaries::zero_dirichlet();
	let coeff = -0.01;

	let data = gaussian(n, h);
	let u_exact =
		Field::from_f64(grid, &data, dev).unwrap();
	let mut b =
		Field::zeros(grid, DType::F64, dev).unwrap();
	b.copy_from(&u_exact).unwrap();
	b.stencil_axpy(coeff, &u_exact, &stencil, &bc)
		.unwrap();

	let mut x =
		Field::zeros(grid, DType::F64, dev).unwrap();
	let mut gmres =
		GmresSolver::new(grid, dev, 30, 1e-8, 500)
			.unwrap();
	let stats = gmres
		.solve(&mut x, &b, coeff, &stencil, &bc)
		.unwrap();

	eprintln!(
		"GMRES HIP: {} its, res={:.2e}",
		stats.iterations, stats.residual,
	);
	assert!(
		stats.converged(),
		"GMRES HIP: {:?}",
		stats.reason
	);
}

// ── IMEX on GPU ─────────────────────────────────────

#[test]
fn hip_imex_convection_diffusion() {
	let n = 32;
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let dev = hip_device();
	let bc = Boundaries::zero_dirichlet();

	let mut solver = ImexSolver::new(
		grid, 0.01, 0.5, 0.0, 0.001, bc, dev, 1e-8, 200,
	)
	.unwrap();

	let data = gaussian(n, h);
	let mut u =
		Field::from_f64(grid, &data, dev).unwrap();
	let initial_norm = u.norm_l2().unwrap();

	solver.step_n(&mut u, 10).unwrap();

	let final_norm = u.norm_l2().unwrap();
	assert!(final_norm.is_finite());
	assert!(final_norm > 0.0);
	assert!(
		final_norm < 2.0 * initial_norm,
		"IMEX HIP blew up: {initial_norm} -> {final_norm}"
	);
	eprintln!(
		"IMEX HIP: norm {initial_norm:.4} -> {final_norm:.4}"
	);
}

// ── Adaptive on GPU ─────────────────────────────────

#[test]
fn hip_adaptive_diffusion() {
	let n = 32;
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let dev = hip_device();
	let bc = Boundaries::zero_dirichlet();

	let mut solver = AdaptiveSolver::new(
		grid, 1.0, 0.0001, bc, dev, 1e-6, 1e-4,
	)
	.unwrap();

	let data = gaussian(n, h);
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
		"adaptive HIP: {initial_peak} -> {final_peak}"
	);
	eprintln!(
		"Adaptive HIP: peak {initial_peak:.4} -> \
		 {final_peak:.4}, {} steps ({} rejected)",
		solver.total_steps, solver.rejected_steps,
	);
}

// ── 9-point stencil on GPU ──────────────────────────

#[test]
fn hip_stencil_9pt_quadratic() {
	let n = 64;
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let dev = hip_device();
	let stencil = Stencil::laplacian_2d_9pt(h, h);
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
	let mut output =
		Field::zeros(grid, DType::F64, dev).unwrap();
	input
		.apply_stencil_into(&stencil, &bc, &mut output)
		.unwrap();

	let out = output.to_vec_f64();
	for row in 2..n - 2 {
		for col in 2..n - 2 {
			let val = out[row * n + col];
			assert!(
				(val - 4.0).abs() < 1e-10,
				"9pt HIP Δ(x²+y²) at ({row},{col}) = {val}"
			);
		}
	}
}

