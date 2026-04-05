//! Comprehensive Fluxion HIP benchmark suite.
//! Matches petsc_full_bench.c scenarios exactly.
//!
//! Run: cargo test --features hip --test hip_full_bench
//!      --release -- --nocapture

#![cfg(feature = "hip")]

use std::time::Instant;

use fluxion_core::{Boundaries, DType, Grid, Stencil};
use fluxion_runtime::{
	CrankNicolsonSolver, Device, DiffusionSolver, Field,
	TimeScheme,
};

fn hip() -> Device {
	Device::Hip { ordinal: 0 }
}

fn gaussian(n: usize, h: f64) -> Vec<f64> {
	let sigma = 0.1_f64;
	(0..n * n)
		.map(|i| {
			let (row, col) = (i / n, i % n);
			let dx = col as f64 * h - 0.5;
			let dy = row as f64 * h - 0.5;
			(-dx.mul_add(dx, dy * dy) / (2.0 * sigma * sigma))
				.exp()
		})
		.collect()
}

// ── Test 1: Sustained Euler ─────────────────────────

fn bench_euler(n: usize, steps: usize) {
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let data = gaussian(n, h);

	let mut u =
		Field::from_f64(grid, &data, hip()).unwrap();
	let mut solver = DiffusionSolver::build(
		grid,
		0.01,
		None,
		Boundaries::zero_dirichlet(),
		TimeScheme::Euler,
		hip(),
	)
	.unwrap();

	solver.step_n(&mut u, 10).unwrap(); // warmup
	let start = Instant::now();
	solver.step_n(&mut u, steps).unwrap();
	let _pk = u.max().unwrap(); // sync
	let elapsed = start.elapsed();

	let ms = elapsed.as_secs_f64() * 1000.0;
	println!(
		"  Fluxion Euler   | {n:5}x{n:<5} | {steps:6} steps | \
		 {ms:8.1} ms | {:8.1} us/step",
		ms * 1000.0 / steps as f64
	);
}

// ── Test 2: RK4 ────────────────────────────────────

fn bench_rk4(n: usize, steps: usize) {
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let data = gaussian(n, h);

	let mut u =
		Field::from_f64(grid, &data, hip()).unwrap();
	let mut solver = DiffusionSolver::build(
		grid,
		0.01,
		None,
		Boundaries::zero_dirichlet(),
		TimeScheme::Rk4,
		hip(),
	)
	.unwrap();

	solver.step_n(&mut u, 5).unwrap(); // warmup
	let start = Instant::now();
	solver.step_n(&mut u, steps).unwrap();
	let _pk = u.max().unwrap(); // sync
	let elapsed = start.elapsed();

	let ms = elapsed.as_secs_f64() * 1000.0;
	println!(
		"  Fluxion RK4     | {n:5}x{n:<5} | {steps:6} steps | \
		 {ms:8.1} ms | {:8.1} us/step",
		ms * 1000.0 / steps as f64
	);
}

// ── Test 3: CG (Crank-Nicolson) ────────────────────

fn bench_cg(n: usize, steps: usize) {
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let data = gaussian(n, h);

	let mut u =
		Field::from_f64(grid, &data, hip()).unwrap();
	let mut solver = CrankNicolsonSolver::new(
		grid,
		0.01,
		0.001,
		Boundaries::zero_dirichlet(),
		hip(),
		1e-6,
		500,
	)
	.unwrap();

	// warmup
	for _ in 0..2 {
		let _ = solver.step(&mut u, None).unwrap();
	}

	let start = Instant::now();
	let mut last_stats = None;
	for _ in 0..steps {
		last_stats =
			Some(solver.step(&mut u, None).unwrap());
	}
	let _pk = u.max().unwrap(); // sync
	let elapsed = start.elapsed();

	let ms = elapsed.as_secs_f64() * 1000.0;
	let its = last_stats
		.map(|s| s.iterations)
		.unwrap_or(0);
	println!(
		"  Fluxion CG      | {n:5}x{n:<5} | {steps:6} steps | \
		 {ms:8.1} ms | {:8.1} us/step (last: {its} its)",
		ms * 1000.0 / steps as f64
	);
}

// ── Test 4: Raw stencil throughput ──────────────────

fn bench_stencil(n: usize, iters: usize) {
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let stencil = Stencil::laplacian_2d_5pt(h, h);
	let bcs = Boundaries::zero_dirichlet();
	let data = gaussian(n, h);

	let input =
		Field::from_f64(grid, &data, hip()).unwrap();
	let mut output =
		Field::zeros(grid, DType::F64, hip()).unwrap();

	// warmup
	for _ in 0..10 {
		input
			.apply_stencil_into(
				&stencil, &bcs, &mut output,
			)
			.unwrap();
	}

	let start = Instant::now();
	for _ in 0..iters {
		input
			.apply_stencil_into(
				&stencil, &bcs, &mut output,
			)
			.unwrap();
	}
	let _nrm = output.norm_l2().unwrap(); // sync
	let elapsed = start.elapsed();

	let ms = elapsed.as_secs_f64() * 1000.0;
	let gflops = iters as f64 * (n * n) as f64 * 9.0
		/ (elapsed.as_secs_f64() * 1e9);
	// Stencil-apply: read 5 neighbors (40B) + write 1 (8B) = 48B/point
	let bw = iters as f64 * (n * n) as f64 * 48.0
		/ (elapsed.as_secs_f64() * 1e9);
	println!(
		"  Fluxion Stencil | {n:5}x{n:<5} | {iters:6} iters | \
		 {:8.1} us/iter | {bw:6.1} GB/s | {gflops:.1} GFLOP/s",
		ms * 1000.0 / iters as f64
	);
}

// ── Test 5: Pure AXPY bandwidth ─────────────────────

fn bench_axpy(n: usize, iters: usize) {
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();

	let x = Field::from_f64(
		grid,
		&gaussian(n, h),
		hip(),
	)
	.unwrap();
	let mut y =
		Field::from_f64(grid, &vec![1.0; n * n], hip())
			.unwrap();

	// warmup
	for _ in 0..20 {
		y.axpy(2.0, &x).unwrap();
	}
	let _ = y.norm_l2().unwrap(); // sync

	let start = Instant::now();
	for _ in 0..iters {
		y.axpy(2.0, &x).unwrap();
	}
	let _ = y.norm_l2().unwrap(); // sync
	let elapsed = start.elapsed();

	let ms = elapsed.as_secs_f64() * 1000.0;
	// 2 reads + 1 write of f64 per element per iter
	let bw = iters as f64 * (n * n) as f64 * 24.0
		/ (elapsed.as_secs_f64() * 1e9);
	println!(
		"  Fluxion AXPY    | {n:5}x{n:<5} | {iters:6} iters | \
		 {ms:8.1} ms | {:8.1} us/iter | {bw:.1} GB/s",
		ms * 1000.0 / iters as f64
	);
}

// ── Test 6: Reduction (norm_l2) ─────────────────────

fn bench_norm(n: usize, iters: usize) {
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let u = Field::from_f64(
		grid,
		&gaussian(n, h),
		hip(),
	)
	.unwrap();

	for _ in 0..20 {
		let _ = u.norm_l2().unwrap();
	}

	let start = Instant::now();
	let mut nrm = 0.0;
	for _ in 0..iters {
		nrm = u.norm_l2().unwrap();
	}
	let _ = nrm;
	let elapsed = start.elapsed();

	let ms = elapsed.as_secs_f64() * 1000.0;
	let bw = iters as f64 * (n * n) as f64 * 8.0
		/ (elapsed.as_secs_f64() * 1e9);
	println!(
		"  Fluxion Norm    | {n:5}x{n:<5} | {iters:6} iters | \
		 {:8.1} us/iter | {bw:6.1} GB/s",
		ms * 1000.0 / iters as f64
	);
}

// ── Test 7: hipGraph Euler ──────────────────────────

fn bench_euler_graph(n: usize, steps: usize) {
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let data = gaussian(n, h);
	let bcs = Boundaries::zero_dirichlet();

	// Without graph.
	let mut u_no =
		Field::from_f64(grid, &data, hip()).unwrap();
	let mut solver_no = DiffusionSolver::build(
		grid, 0.01, None, bcs.clone(),
		TimeScheme::Euler, hip(),
	)
	.unwrap();
	solver_no.step_n(&mut u_no, 10).unwrap(); // warmup
	let start = Instant::now();
	solver_no.step_n(&mut u_no, steps).unwrap();
	let _ = u_no.max().unwrap();
	let no_elapsed = start.elapsed();

	// With graph.
	let mut u_gr =
		Field::from_f64(grid, &data, hip()).unwrap();
	let mut solver_gr = DiffusionSolver::build(
		grid, 0.01, None, bcs,
		TimeScheme::Euler, hip(),
	)
	.unwrap();
	solver_gr.enable_hip_graph();
	solver_gr.step_n(&mut u_gr, 10).unwrap(); // warmup + capture
	let start = Instant::now();
	solver_gr.step_n(&mut u_gr, steps).unwrap();
	let _ = u_gr.max().unwrap();
	let gr_elapsed = start.elapsed();

	let no_us =
		no_elapsed.as_secs_f64() * 1e6 / steps as f64;
	let gr_us =
		gr_elapsed.as_secs_f64() * 1e6 / steps as f64;
	let speedup = no_us / gr_us;

	println!(
		"  {n:5}x{n:<5}  regular: {:7.1} us/step | \
		 graph: {:7.1} us/step | {speedup:.2}x",
		no_us, gr_us,
	);
}

// ── Test 8: Pipelined CG vs Standard CG ────────────

fn bench_pipelined_cg(n: usize, steps: usize) {
	use fluxion_runtime::{CgSolver, PipelinedCgSolver};

	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let stencil = Stencil::laplacian_2d_5pt(h, h);
	let bcs = Boundaries::zero_dirichlet();
	let data = gaussian(n, h);
	let coeff = -0.005; // Crank-Nicolson-like

	let b =
		Field::from_f64(grid, &data, hip()).unwrap();

	// Standard CG
	let mut x_cg =
		Field::zeros(grid, DType::F64, hip()).unwrap();
	let mut cg =
		CgSolver::new(grid, hip(), 1e-8, 500).unwrap();
	// warmup
	let _ = cg.solve(
		&mut x_cg, &b, coeff, &stencil, &bcs,
	);
	let start = Instant::now();
	let mut cg_its = 0;
	for _ in 0..steps {
		x_cg.fill(0.0).unwrap();
		let s = cg
			.solve(
				&mut x_cg, &b, coeff, &stencil, &bcs,
			)
			.unwrap();
		cg_its = s.iterations;
	}
	let _ = x_cg.max().unwrap();
	let cg_elapsed = start.elapsed();

	// Pipelined CG
	let mut x_pcg =
		Field::zeros(grid, DType::F64, hip()).unwrap();
	let mut pcg = PipelinedCgSolver::new(
		grid, hip(), 1e-8, 500,
	)
	.unwrap();
	// warmup
	let _ = pcg.solve(
		&mut x_pcg, &b, coeff, &stencil, &bcs,
	);
	let start = Instant::now();
	let mut pcg_its = 0;
	for _ in 0..steps {
		x_pcg.fill(0.0).unwrap();
		let s = pcg
			.solve(
				&mut x_pcg, &b, coeff, &stencil, &bcs,
			)
			.unwrap();
		pcg_its = s.iterations;
	}
	let _ = x_pcg.max().unwrap();
	let pcg_elapsed = start.elapsed();

	let cg_us =
		cg_elapsed.as_secs_f64() * 1e6 / steps as f64;
	let pcg_us =
		pcg_elapsed.as_secs_f64() * 1e6 / steps as f64;
	let speedup = cg_us / pcg_us;

	println!(
		"  {n:5}x{n:<5}  CG: {:8.0} us ({cg_its:3} its) | \
		 Pipelined: {:8.0} us ({pcg_its:3} its) | \
		 {speedup:.2}x",
		cg_us, pcg_us,
	);
}

// ── Test 9: BiCGSTAB vs CG ─────────────────────────

fn bench_bicgstab_vs_cg(n: usize, steps: usize) {
	use fluxion_runtime::{BiCgStabSolver, CgSolver};

	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let stencil = Stencil::laplacian_2d_5pt(h, h);
	let bcs = Boundaries::zero_dirichlet();
	let data = gaussian(n, h);
	let coeff = -0.5;

	let b =
		Field::from_f64(grid, &data, hip()).unwrap();

	// CG
	let mut x_cg =
		Field::zeros(grid, DType::F64, hip()).unwrap();
	let mut cg =
		CgSolver::new(grid, hip(), 1e-8, 500).unwrap();
	// warmup
	let _ = cg.solve(
		&mut x_cg, &b, coeff, &stencil, &bcs,
	);
	x_cg.fill(0.0).unwrap();
	let start = Instant::now();
	let mut cg_its = 0;
	for _ in 0..steps {
		x_cg.fill(0.0).unwrap();
		let s = cg
			.solve(
				&mut x_cg, &b, coeff, &stencil, &bcs,
			)
			.unwrap();
		cg_its = s.iterations;
	}
	let _ = x_cg.max().unwrap();
	let cg_elapsed = start.elapsed();

	// BiCGSTAB
	let mut x_bc =
		Field::zeros(grid, DType::F64, hip()).unwrap();
	let mut bcgs = BiCgStabSolver::new(
		grid, hip(), 1e-8, 500,
	)
	.unwrap();
	// warmup
	let _ = bcgs.solve(
		&mut x_bc, &b, coeff, &stencil, &bcs,
	);
	x_bc.fill(0.0).unwrap();
	let start = Instant::now();
	let mut bc_its = 0;
	for _ in 0..steps {
		x_bc.fill(0.0).unwrap();
		let s = bcgs
			.solve(
				&mut x_bc, &b, coeff, &stencil, &bcs,
			)
			.unwrap();
		bc_its = s.iterations;
	}
	let _ = x_bc.max().unwrap();
	let bc_elapsed = start.elapsed();

	let cg_us =
		cg_elapsed.as_secs_f64() * 1e6 / steps as f64;
	let bc_us =
		bc_elapsed.as_secs_f64() * 1e6 / steps as f64;

	println!(
		"  {n:5}x{n:<5}  CG: {:8.0} us ({cg_its:3} its) | \
		 BiCGSTAB: {:8.0} us ({bc_its:3} its)",
		cg_us, bc_us,
	);
}

// ── Test 9: PCG vs CG ──────────────────────────────

fn bench_pcg(n: usize, steps: usize) {
	let h = 1.0 / (n - 1) as f64;
	let grid = Grid::square(n, h).unwrap();
	let data = gaussian(n, h);
	let bcs = Boundaries::zero_dirichlet();
	let alpha = 1.0;
	let dt = 0.01;

	// CG (unpreconditioned)
	let mut u_cg =
		Field::from_f64(grid, &data, hip()).unwrap();
	let mut solver_cg = CrankNicolsonSolver::new(
		grid, alpha, dt, bcs.clone(), hip(), 1e-8, 500,
	)
	.unwrap();
	// warmup
	for _ in 0..2 {
		let _ = solver_cg.step(&mut u_cg, None).unwrap();
	}
	let start = Instant::now();
	let mut cg_its = 0;
	for _ in 0..steps {
		let s =
			solver_cg.step(&mut u_cg, None).unwrap();
		cg_its = s.iterations;
	}
	let _ = u_cg.max().unwrap();
	let cg_elapsed = start.elapsed();

	// PCG (multigrid-preconditioned)
	let mut u_pcg =
		Field::from_f64(grid, &data, hip()).unwrap();
	let mut solver_pcg =
		CrankNicolsonSolver::with_multigrid(
			grid, alpha, dt, bcs, hip(), 1e-8, 500,
			2, 2, 2.0 / 3.0,
		)
		.unwrap();
	// warmup
	for _ in 0..2 {
		let _ =
			solver_pcg.step(&mut u_pcg, None).unwrap();
	}
	let start = Instant::now();
	let mut pcg_its = 0;
	for _ in 0..steps {
		let s =
			solver_pcg.step(&mut u_pcg, None).unwrap();
		pcg_its = s.iterations;
	}
	let _ = u_pcg.max().unwrap();
	let pcg_elapsed = start.elapsed();

	let cg_us =
		cg_elapsed.as_secs_f64() * 1e6 / steps as f64;
	let pcg_us =
		pcg_elapsed.as_secs_f64() * 1e6 / steps as f64;
	let speedup = cg_us / pcg_us;

	println!(
		"  {n:5}x{n:<5}  CG: {:7.1} us/step ({cg_its:3} its) | \
		 PCG: {:7.1} us/step ({pcg_its:3} its) | \
		 speedup {speedup:.1}x",
		cg_us, pcg_us
	);
}

// ── Main test ───────────────────────────────────────

#[test]
fn full_benchmark_suite() {
	println!();
	println!(
		"========================================\
		 ============================"
	);
	println!(
		" Fluxion HIP Benchmark Suite (RX 7800XT)"
	);
	println!(
		"========================================\
		 ============================"
	);

	println!("\n--- Test 1: Sustained Euler diffusion ---");
	bench_euler(256, 5000);
	bench_euler(512, 2000);
	bench_euler(1024, 500);
	bench_euler(2048, 100);

	println!("\n--- Test 2: RK4 integration ---");
	bench_rk4(256, 1000);
	bench_rk4(512, 500);
	bench_rk4(1024, 200);

	println!("\n--- Test 3: Crank-Nicolson CG ---");
	bench_cg(256, 50);
	bench_cg(512, 50);
	bench_cg(1024, 20);

	println!(
		"\n--- Test 4: Raw stencil throughput ---"
	);
	bench_stencil(256, 10000);
	bench_stencil(512, 5000);
	bench_stencil(1024, 2000);
	bench_stencil(2048, 500);

	println!(
		"\n--- Test 5: Pure BLAS-1 AXPY bandwidth ---"
	);
	bench_axpy(512, 10000);
	bench_axpy(1024, 5000);
	bench_axpy(2048, 2000);

	println!(
		"\n--- Test 6: Reduction (norm_l2) ---"
	);
	bench_norm(512, 10000);
	bench_norm(1024, 5000);
	bench_norm(2048, 2000);

	println!(
		"\n--- Test 7: hipGraph Euler (regular vs graph) ---"
	);
	bench_euler_graph(64, 5000);
	bench_euler_graph(128, 5000);
	bench_euler_graph(256, 5000);
	bench_euler_graph(512, 2000);
	bench_euler_graph(1024, 500);

	println!(
		"\n--- Test 8: Pipelined CG vs Standard CG ---"
	);
	bench_pipelined_cg(64, 20);
	bench_pipelined_cg(128, 10);
	bench_pipelined_cg(256, 5);
	bench_pipelined_cg(512, 5);

	println!(
		"\n--- Test 9: BiCGSTAB vs CG (same SPD system) ---"
	);
	bench_bicgstab_vs_cg(64, 20);
	bench_bicgstab_vs_cg(128, 10);
	bench_bicgstab_vs_cg(256, 5);

	println!(
		"\n--- Test 10: PCG (MG) vs CG ---"
	);
	bench_pcg(64, 50);
	bench_pcg(128, 20);
	bench_pcg(256, 20);
	bench_pcg(512, 10);

	println!();
}
