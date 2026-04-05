// Bilinear prolongation: coarse grid → fine grid (additive).
// fine[i,j] += coarse[i/2, j/2] (nearest-neighbor for now,
// bilinear interpolation for interior points).

struct Params {
    fine_rows: u32,
    fine_cols: u32,
    coarse_cols: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> coarse: array<f32>;
@group(0) @binding(2) var<storage, read_write> fine: array<f32>;

@compute
@workgroup_size(16, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let fc = gid.x;
    let fr = gid.y;

    if (fr >= params.fine_rows || fc >= params.fine_cols) {
        return;
    }

    // Map fine coords to coarse coords with bilinear weights.
    let cr = fr / 2u;
    let cc = fc / 2u;

    // Simple injection: just add the coarse value.
    fine[fr * params.fine_cols + fc] +=
        coarse[cr * params.coarse_cols + cc];
}
