#include <hip/hip_runtime.h>

// Fused stencil + axpy: y += alpha * laplacian(x).
// LDS-tiled for bandwidth efficiency. Single pass.

constexpr int TILE_X = 16;
constexpr int TILE_Y = 16;

extern "C" __global__ void fused_stencil_axpy_f64(
    double alpha,
    const double* __restrict__ x,
    double* __restrict__ y,
    int rows, int cols,
    double inv_dx2, double inv_dy2)
{
    __shared__ double tile[TILE_Y + 2][TILE_X + 2];

    int col = blockIdx.x * TILE_X + threadIdx.x;
    int row = blockIdx.y * TILE_Y + threadIdx.y;
    int tx = threadIdx.x + 1;
    int ty = threadIdx.y + 1;

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

    // Interior only — boundaries untouched.
    if (row <= 0 || row >= rows - 1 || col <= 0 || col >= cols - 1)
        return;

    double center = -2.0 * (inv_dx2 + inv_dy2);
    double lap = fma(center, tile[ty][tx],
                 fma(inv_dy2, tile[ty - 1][tx],
                 fma(inv_dy2, tile[ty + 1][tx],
                 fma(inv_dx2, tile[ty][tx - 1],
                     inv_dx2 * tile[ty][tx + 1]))));

    int idx = row * cols + col;
    y[idx] = fma(alpha, lap, y[idx]);
}
