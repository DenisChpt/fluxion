// Weighted Jacobi smoother: x = x + omega * D^{-1} * (b - A*x)
// For the Laplacian, D = diag(A) = -2*(inv_dx2 + inv_dy2).
// So D^{-1} = 1 / (-2*(inv_dx2 + inv_dy2)).
//
// Interior points only; boundaries untouched.

struct Params {
    rows: u32,
    cols: u32,
    inv_dx2: f32,
    inv_dy2: f32,
    omega: f32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> x: array<f32>;

@compute
@workgroup_size(16, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    let row = gid.y;

    if (row >= params.rows || col >= params.cols) {
        return;
    }
    if (row == 0u || row >= params.rows - 1u ||
        col == 0u || col >= params.cols - 1u) {
        return;
    }

    let c = row * params.cols + col;
    let center_coeff = -2.0 * (params.inv_dx2 + params.inv_dy2);

    // A*x at this point.
    let ax = x[c] * center_coeff
        + x[(row - 1u) * params.cols + col] * params.inv_dy2
        + x[(row + 1u) * params.cols + col] * params.inv_dy2
        + x[row * params.cols + (col - 1u)] * params.inv_dx2
        + x[row * params.cols + (col + 1u)] * params.inv_dx2;

    let residual = b[c] - ax;
    let inv_diag = 1.0 / center_coeff;

    x[c] = x[c] + params.omega * inv_diag * residual;
}
