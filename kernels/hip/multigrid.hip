#include <hip/hip_runtime.h>

// Multigrid transfer operators and Jacobi smoother.

constexpr int TILE_X = 16;
constexpr int TILE_Y = 16;

// ── Full-weighting restriction: fine → coarse ───────

extern "C" __global__ void restrict_f64(
    const double* __restrict__ fine,
    double* __restrict__ coarse,
    int fine_rows, int fine_cols,
    int coarse_rows, int coarse_cols)
{
    int cc = blockIdx.x * blockDim.x + threadIdx.x;
    int cr = blockIdx.y * blockDim.y + threadIdx.y;

    if (cr >= coarse_rows || cc >= coarse_cols) return;

    int fr = cr * 2;
    int fc = cc * 2;
    int fr1 = min(fr + 1, fine_rows - 1);
    int fc1 = min(fc + 1, fine_cols - 1);

    coarse[cr * coarse_cols + cc] = 0.25 * (
        fine[fr  * fine_cols + fc ] +
        fine[fr1 * fine_cols + fc ] +
        fine[fr  * fine_cols + fc1] +
        fine[fr1 * fine_cols + fc1]);
}

// ── Prolongation: coarse → fine (additive) ──────────

extern "C" __global__ void prolong_f64(
    const double* __restrict__ coarse,
    double* __restrict__ fine,
    int fine_rows, int fine_cols,
    int coarse_cols)
{
    int fc = blockIdx.x * blockDim.x + threadIdx.x;
    int fr = blockIdx.y * blockDim.y + threadIdx.y;

    if (fr >= fine_rows || fc >= fine_cols) return;

    fine[fr * fine_cols + fc] += coarse[(fr / 2) * coarse_cols + (fc / 2)];
}

// ── Weighted Jacobi smoother ────────────────────────
// x = x + omega * D^{-1} * (b - A*x)

extern "C" __global__ void weighted_jacobi_f64(
    double* __restrict__ x,
    const double* __restrict__ b,
    double omega,
    int rows, int cols,
    double inv_dx2, double inv_dy2)
{
    __shared__ double tile[TILE_Y + 2][TILE_X + 2];

    int col = blockIdx.x * TILE_X + threadIdx.x;
    int row = blockIdx.y * TILE_Y + threadIdx.y;
    int tx = threadIdx.x + 1;
    int ty = threadIdx.y + 1;

    // Load x into shared memory with halo.
    if (row < rows && col < cols)
        tile[ty][tx] = x[row * cols + col];
    else
        tile[ty][tx] = 0.0;

    if (threadIdx.y == 0 && row > 0)
        tile[0][tx] = x[(row - 1) * cols + col];
    if (threadIdx.y == TILE_Y - 1 && row + 1 < rows)
        tile[TILE_Y + 1][tx] = x[(row + 1) * cols + col];
    if (threadIdx.x == 0 && col > 0)
        tile[ty][0] = x[row * cols + (col - 1)];
    if (threadIdx.x == TILE_X - 1 && col + 1 < cols)
        tile[ty][TILE_X + 1] = x[row * cols + (col + 1)];

    __syncthreads();

    if (row <= 0 || row >= rows - 1 || col <= 0 || col >= cols - 1)
        return;

    double center = -2.0 * (inv_dx2 + inv_dy2);
    double ax = fma(center, tile[ty][tx],
               fma(inv_dy2, tile[ty - 1][tx],
               fma(inv_dy2, tile[ty + 1][tx],
               fma(inv_dx2, tile[ty][tx - 1],
                   inv_dx2 * tile[ty][tx + 1]))));

    double residual = b[row * cols + col] - ax;
    double inv_diag = 1.0 / center;

    int idx = row * cols + col;
    x[idx] = fma(omega * inv_diag, residual, x[idx]);
}
