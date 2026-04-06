# Fluxion

GPU-first PDE solver in Rust. Zero-allocation hot path, enum dispatch (no trait objects, no generics in user API), automatic backend selection.

## Backends

| Backend | Technology | Status |
|---------|-----------|--------|
| CPU | faer + rayon | Done |
| wgpu | WGSL compute shaders (Vulkan/Metal/DX12) | Done |
| HIP | ROCm FFI | In progress |
| CUDA | CUDA FFI | In progress |

Fallback chain: CUDA/ROCm -> wgpu -> CPU. The runtime picks the best available device automatically.

## Solvers

**Explicit time-stepping** -- `DiffusionSolver`
- Forward Euler, Heun (RK2), Classical RK4
- SSP Runge-Kutta (order 2 & 3)
- Adaptive time-stepping (error-based dt control)

**Implicit** -- `CrankNicolsonSolver`, `BdfSolver`, `ImexSolver`
- Crank-Nicolson (no CFL constraint)
- BDF (backward differentiation)
- IMEX (implicit-explicit splitting)

**Linear solvers**
- Conjugate Gradient (`CgSolver`) -- matrix-free
- Pipelined CG (`PipelinedCgSolver`) -- overlapped reductions
- BiCGSTAB (`BiCgStabSolver`)
- GMRES (`GmresSolver`)

**Preconditioners**
- Multigrid V-cycle (restriction, prolongation, weighted Jacobi smoothing)
- Identity (passthrough)

## Backend operations

Stencil apply, fused stencil+axpy, AXPY, WAXPY, AYPX, scale, dot, dual-dot, L2 norm, reductions (sum/max/min), pointwise multiply/divide/max/min, reciprocal, abs, restrict, prolong, weighted Jacobi, convection-diffusion operator (with upwind), fused conv-diff+axpy.

Kahan compensated summation on CPU reductions. Bind group caching and reduction buffer pre-allocation on wgpu.

## License

MIT
