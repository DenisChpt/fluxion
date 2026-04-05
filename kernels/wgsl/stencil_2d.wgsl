struct Grid {
    rows: u32,
    cols: u32,
    inv_dx2: f32,
    inv_dy2: f32,
}

@group(0) @binding(0) var<uniform> grid: Grid;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute
@workgroup_size(16, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    let row = gid.y;

    if (row >= grid.rows || col >= grid.cols) {
        return;
    }

    // Boundary: Dirichlet zero.
    if (row == 0u || row >= grid.rows - 1u ||
        col == 0u || col >= grid.cols - 1u) {
        output[row * grid.cols + col] = 0.0;
        return;
    }

    let c = row * grid.cols + col;
    let center_coeff = -2.0 * (grid.inv_dx2 + grid.inv_dy2);

    let val = input[c] * center_coeff
        + input[(row - 1u) * grid.cols + col] * grid.inv_dy2
        + input[(row + 1u) * grid.cols + col] * grid.inv_dy2
        + input[row * grid.cols + (col - 1u)] * grid.inv_dx2
        + input[row * grid.cols + (col + 1u)] * grid.inv_dx2;

    output[c] = val;
}
