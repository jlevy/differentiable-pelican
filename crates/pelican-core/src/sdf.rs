use burn::prelude::*;
use burn::tensor::activation::sigmoid;

/// SDF for a circle: negative inside, positive outside.
pub fn sdf_circle<B: Backend>(
    grid: &Tensor<B, 3>, // [H, W, 2]
    cx: Tensor<B, 1>,    // [1]
    cy: Tensor<B, 1>,    // [1]
    radius: Tensor<B, 1>, // [1]
) -> Tensor<B, 2> {
    let gx = grid.clone().slice([None, None, Some((0, 1))]).squeeze(2); // [H, W]
    let gy = grid.clone().slice([None, None, Some((1, 2))]).squeeze(2); // [H, W]

    let cx = cx.unsqueeze::<2>(); // [1, 1]
    let cy = cy.unsqueeze::<2>();
    let radius = radius.unsqueeze::<2>();

    let dx = gx - cx;
    let dy = gy - cy;
    // Epsilon prevents gradient singularity at exact center (sqrt(0) has infinite gradient)
    let dist = (dx.clone() * dx + dy.clone() * dy + 1e-10).sqrt();
    dist - radius
}

/// SDF for an ellipse using scaled normalized distance.
///
/// Uses the normalized-distance approximation scaled by geometric mean radius:
///     sdf ≈ (‖p_normalized‖ - 1) · √(rx·ry)
///
/// This is differentiable everywhere with bounded gradients, making it ideal
/// for gradient-based optimization. The approximation error vs the exact
/// Quilez solution is sub-pixel at typical rendering resolutions.
pub fn sdf_ellipse<B: Backend>(
    grid: &Tensor<B, 3>,
    cx: Tensor<B, 1>,
    cy: Tensor<B, 1>,
    rx: Tensor<B, 1>,
    ry: Tensor<B, 1>,
    rotation: Tensor<B, 1>,
) -> Tensor<B, 2> {
    let gx = grid.clone().slice([None, None, Some((0, 1))]).squeeze(2);
    let gy = grid.clone().slice([None, None, Some((1, 2))]).squeeze(2);

    let cx = cx.unsqueeze::<2>();
    let cy = cy.unsqueeze::<2>();

    let dx = gx - cx;
    let dy = gy - cy;

    let neg_rot = rotation.neg();
    let cos_theta = neg_rot.clone().cos().unsqueeze::<2>();
    let sin_theta = neg_rot.sin().unsqueeze::<2>();

    let x_rot = dx.clone() * cos_theta.clone() - dy.clone() * sin_theta.clone();
    let y_rot = dx * sin_theta + dy * cos_theta;

    // Epsilon prevents division by near-zero radii
    let rx_2d = rx.clone().unsqueeze::<2>() + 1e-10;
    let ry_2d = ry.clone().unsqueeze::<2>() + 1e-10;
    let x_norm = x_rot / rx_2d;
    let y_norm = y_rot / ry_2d;

    // Epsilon inside sqrt prevents gradient singularity at exact center
    let dist_norm = (x_norm.clone() * x_norm + y_norm.clone() * y_norm + 1e-10).sqrt();

    // Scale by geometric mean radius for proper distance units
    let scale = (rx.unsqueeze::<2>() * ry.unsqueeze::<2>() + 1e-10).sqrt();
    (dist_norm - 1.0) * scale
}

/// SDF for a triangle using edge-distance + cross-product sign test.
pub fn sdf_triangle<B: Backend>(
    grid: &Tensor<B, 3>, // [H, W, 2]
    v0: Tensor<B, 1>,    // [2]
    v1: Tensor<B, 1>,    // [2]
    v2: Tensor<B, 1>,    // [2]
) -> Tensor<B, 2> {
    let gx = grid.clone().slice([None, None, Some((0, 1))]).squeeze(2);
    let gy = grid.clone().slice([None, None, Some((1, 2))]).squeeze(2);

    let v0x = v0.clone().slice([0..1]).unsqueeze::<2>();
    let v0y = v0.slice([1..2]).unsqueeze::<2>();
    let v1x = v1.clone().slice([0..1]).unsqueeze::<2>();
    let v1y = v1.slice([1..2]).unsqueeze::<2>();
    let v2x = v2.clone().slice([0..1]).unsqueeze::<2>();
    let v2y = v2.slice([1..2]).unsqueeze::<2>();

    // Edge vectors
    let e0x = v1x.clone() - v0x.clone();
    let e0y = v1y.clone() - v0y.clone();
    let e1x = v2x.clone() - v1x.clone();
    let e1y = v2y.clone() - v1y.clone();
    let e2x = v0x.clone() - v2x.clone();
    let e2y = v0y.clone() - v2y.clone();

    // Point-to-vertex
    let p0x = gx.clone() - v0x;
    let p0y = gy.clone() - v0y;
    let p1x = gx.clone() - v1x;
    let p1y = gy.clone() - v1y;
    let p2x = gx.clone() - v2x;
    let p2y = gy.clone() - v2y;

    // Edge distances
    let d0 = edge_distance(&e0x, &e0y, &p0x, &p0y);
    let d1 = edge_distance(&e1x, &e1y, &p1x, &p1y);
    let d2 = edge_distance(&e2x, &e2y, &p2x, &p2y);

    // Element-wise min of three distance tensors using mask_where
    let min_d01 = d0.clone().mask_where(d1.clone().lower(d0.clone()), d1);
    let min_dist = min_d01.clone().mask_where(d2.clone().lower(min_d01.clone()), d2);

    // 2D cross products for inside/outside
    let c0 = cross_2d(&e0x, &e0y, &p0x, &p0y);
    let c1 = cross_2d(&e1x, &e1y, &p1x, &p1y);
    let c2 = cross_2d(&e2x, &e2y, &p2x, &p2y);

    // Inside if all cross products same sign (handles CW and CCW)
    let zeros = Tensor::<B, 2>::zeros_like(&c0);
    let c0_pos = c0.clone().greater(zeros.clone());
    let c1_pos = c1.clone().greater(zeros.clone());
    let c2_pos = c2.clone().greater(zeros.clone());
    let c0_neg = c0.lower(zeros.clone());
    let c1_neg = c1.lower(zeros.clone());
    let c2_neg = c2.lower(zeros);

    // Multiply float masks: 1.0 where condition holds, 0.0 where not
    let all_pos = c0_pos.float() * c1_pos.float() * c2_pos.float();
    let all_neg = c0_neg.float() * c1_neg.float() * c2_neg.float();
    let inside_float = all_pos + all_neg; // 1 where inside, 0 where outside

    // sign: inside -> -1, outside -> +1
    // SDF = min_dist * (1 - 2*inside)
    let sign = Tensor::<B, 2>::ones_like(&min_dist) - inside_float * 2.0;
    min_dist * sign
}

fn edge_distance<B: Backend>(
    ex: &Tensor<B, 2>,
    ey: &Tensor<B, 2>,
    px: &Tensor<B, 2>,
    py: &Tensor<B, 2>,
) -> Tensor<B, 2> {
    let edge_len_sq = ex.clone() * ex.clone() + ey.clone() * ey.clone() + 1e-8;
    let dot = px.clone() * ex.clone() + py.clone() * ey.clone();
    let t = (dot / edge_len_sq).clamp(0.0, 1.0);
    let proj_x = px.clone() - ex.clone() * t.clone();
    let proj_y = py.clone() - ey.clone() * t;
    (proj_x.clone() * proj_x + proj_y.clone() * proj_y).sqrt()
}

fn cross_2d<B: Backend>(
    ax: &Tensor<B, 2>,
    ay: &Tensor<B, 2>,
    bx: &Tensor<B, 2>,
    by: &Tensor<B, 2>,
) -> Tensor<B, 2> {
    ax.clone() * by.clone() - ay.clone() * bx.clone()
}

/// Convert SDF to soft coverage via sigmoid.
pub fn coverage_from_sdf<B: Backend>(sdf: Tensor<B, 2>, tau: f32) -> Tensor<B, 2> {
    let scaled = (sdf / tau).clamp(-10.0, 10.0).neg();
    sigmoid(scaled)
}
