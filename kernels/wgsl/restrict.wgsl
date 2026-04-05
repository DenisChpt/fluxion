// Full-weighting restriction: fine grid → coarse grid.
// coarse[i,j] = 0.25*(fine[2i,2j] + fine[2i+1,2j]
//                    + fine[2i,2j+1] + fine[2i+1,2j+1])

struct Params {
    fine_rows: u32,
    fine_cols: u32,
    coarse_rows: u32,
    coarse_cols: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> fine: array<f32>;
@group(0) @binding(2) var<storage, read_write> coarse: array<f32>;

@compute
@workgroup_size(16, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cc = gid.x;
    let cr = gid.y;

    if (cr >= params.coarse_rows || cc >= params.coarse_cols) {
        return;
    }

    let fr = cr * 2u;
    let fc = cc * 2u;

    // Clamp to fine grid bounds.
    let fr1 = min(fr + 1u, params.fine_rows - 1u);
    let fc1 = min(fc + 1u, params.fine_cols - 1u);

    let v00 = fine[fr  * params.fine_cols + fc];
    let v10 = fine[fr1 * params.fine_cols + fc];
    let v01 = fine[fr  * params.fine_cols + fc1];
    let v11 = fine[fr1 * params.fine_cols + fc1];

    coarse[cr * params.coarse_cols + cc] =
        0.25 * (v00 + v10 + v01 + v11);
}
