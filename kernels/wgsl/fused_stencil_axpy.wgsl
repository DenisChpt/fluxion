// Fused stencil + axpy: y[i] += alpha * laplacian(x)[i]
//
// Single memory pass — reads x (5-point stencil), reads y,
// writes y. Halves memory traffic vs separate stencil + axpy.

struct Params {
    rows: u32,
    cols: u32,
    inv_dx2: f32,
    inv_dy2: f32,
    alpha: f32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;

@compute
@workgroup_size(16, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    let row = gid.y;

    if (row >= params.rows || col >= params.cols) {
        return;
    }

    // Boundary: no contribution (Dirichlet zero stencil).
    if (row == 0u || row >= params.rows - 1u ||
        col == 0u || col >= params.cols - 1u) {
        return;
    }

    let c = row * params.cols + col;
    let center_coeff = -2.0 * (params.inv_dx2 + params.inv_dy2);

    let lap = x[c] * center_coeff
        + x[(row - 1u) * params.cols + col] * params.inv_dy2
        + x[(row + 1u) * params.cols + col] * params.inv_dy2
        + x[row * params.cols + (col - 1u)] * params.inv_dx2
        + x[row * params.cols + (col + 1u)] * params.inv_dx2;

    y[c] = fma(params.alpha, lap, y[c]);
}
