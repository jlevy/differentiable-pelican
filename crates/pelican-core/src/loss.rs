use burn::prelude::*;

use crate::geometry::{PelicanModel, ShapeKind};

/// MSE loss between rendered and target images.
pub fn mse_loss<B: Backend>(rendered: Tensor<B, 2>, target: &Tensor<B, 2>) -> Tensor<B, 1> {
    let diff = rendered - target.clone();
    (diff.clone() * diff).mean().unsqueeze()
}

/// Edge loss using Sobel filters (via shifted-tensor differences, not conv2d).
pub fn edge_loss<B: Backend>(rendered: &Tensor<B, 2>, target: &Tensor<B, 2>) -> Tensor<B, 1> {
    let r_mag = sobel_magnitude(rendered);
    let t_mag = sobel_magnitude(target);
    let diff = r_mag - t_mag;
    (diff.clone() * diff).mean().unsqueeze()
}

/// Compute Sobel gradient magnitude using shifted-tensor differences.
fn sobel_magnitude<B: Backend>(img: &Tensor<B, 2>) -> Tensor<B, 2> {
    let [h, w] = img.dims();
    if h < 3 || w < 3 {
        return Tensor::zeros_like(img);
    }

    // Sobel X kernel effect via shifts:
    // gx = (-1*[r-1,c-1] + 1*[r-1,c+1] - 2*[r,c-1] + 2*[r,c+1] - 1*[r+1,c-1] + 1*[r+1,c+1])
    // We compute on the interior [1..h-1, 1..w-1]
    let tl = img.clone().slice([Some((0, h as i64 - 2)), Some((0, w as i64 - 2))]); // top-left
    let tr = img.clone().slice([Some((0, h as i64 - 2)), Some((2, w as i64))]); // top-right
    let ml = img.clone().slice([Some((1, h as i64 - 1)), Some((0, w as i64 - 2))]); // mid-left
    let mr = img.clone().slice([Some((1, h as i64 - 1)), Some((2, w as i64))]); // mid-right
    let bl = img.clone().slice([Some((2, h as i64)), Some((0, w as i64 - 2))]); // bottom-left
    let br = img.clone().slice([Some((2, h as i64)), Some((2, w as i64))]); // bottom-right
    let tm = img.clone().slice([Some((0, h as i64 - 2)), Some((1, w as i64 - 1))]); // top-mid
    let bm = img.clone().slice([Some((2, h as i64)), Some((1, w as i64 - 1))]); // bottom-mid

    let gx = tr.clone() - tl.clone() + mr.clone() * 2.0 - ml.clone() * 2.0 + br.clone()
        - bl.clone();
    let gy = bl + bm * 2.0 + br - tl - tm * 2.0 - tr;

    (gx.clone() * gx + gy.clone() * gy + 1e-8).sqrt()
}

/// Perimeter prior: penalizes large shapes.
pub fn perimeter_prior<B: Backend>(model: &PelicanModel<B>) -> Tensor<B, 1> {
    let device = model.devices()[0].clone();
    let mut total = Tensor::<B, 1>::zeros([1], &device);

    for shape in &model.shapes {
        let p = match shape {
            ShapeKind::Circle(c) => {
                // 2 * pi * radius
                c.radius() * (2.0 * std::f32::consts::PI)
            }
            ShapeKind::Ellipse(e) => {
                // Ramanujan approximation
                let a = e.rx();
                let b = e.ry();
                let sum = a.clone() + b.clone();
                let diff = a.clone() - b.clone();
                let h = (diff.clone() * diff) / (sum.clone() * sum.clone() + 1e-8);
                let denom = h.clone() * (-3.0) + 4.0;
                let bracket = h * 3.0 / (denom.sqrt() + 10.0) + 1.0;
                sum * bracket * std::f32::consts::PI
            }
            ShapeKind::Triangle(t) => {
                let v0 = t.v0();
                let v1 = t.v1();
                let v2 = t.v2();
                let e0 = v1.clone() - v0.clone();
                let e1 = v2.clone() - v1;
                let e2 = v0 - v2;
                let l0 = (e0.clone() * e0).sum().sqrt();
                let l1 = (e1.clone() * e1).sum().sqrt();
                let l2 = (e2.clone() * e2).sum().sqrt();
                l0 + l1 + l2
            }
        };
        total = total + p;
    }

    total
}

/// Triangle degeneracy penalty: penalizes edges shorter than min_edge_length.
pub fn triangle_degeneracy_penalty<B: Backend>(
    model: &PelicanModel<B>,
    min_edge_length: f32,
) -> Tensor<B, 1> {
    let device = model.devices()[0].clone();
    let mut total = Tensor::<B, 1>::zeros([1], &device);

    for shape in &model.shapes {
        if let ShapeKind::Triangle(t) = shape {
            let v0 = t.v0();
            let v1 = t.v1();
            let v2 = t.v2();
            let e0 = v1.clone() - v0.clone();
            let e1 = v2.clone() - v1;
            let e2 = v0 - v2;
            let l0 = (e0.clone() * e0).sum().sqrt();
            let l1 = (e1.clone() * e1).sum().sqrt();
            let l2 = (e2.clone() * e2).sum().sqrt();
            // relu(min_edge_length - edge_length)
            total = total + (l0.neg() + min_edge_length).clamp_min(0.0);
            total = total + (l1.neg() + min_edge_length).clamp_min(0.0);
            total = total + (l2.neg() + min_edge_length).clamp_min(0.0);
        }
    }

    total
}

/// On-canvas penalty: penalizes shapes that extend beyond [margin, 1-margin].
pub fn on_canvas_penalty<B: Backend>(model: &PelicanModel<B>, margin: f32) -> Tensor<B, 1> {
    let device = model.devices()[0].clone();
    let mut total = Tensor::<B, 1>::zeros([1], &device);
    let upper = 1.0 - margin;

    for shape in &model.shapes {
        match shape {
            ShapeKind::Circle(c) => {
                let cx = c.cx();
                let cy = c.cy();
                let r = c.radius();
                // right: cx + r > upper
                total =
                    total + (cx.clone() + r.clone() - upper).clamp_min(0.0);
                // left: cx - r < margin
                total = total
                    + (cx.clone().neg() + r.clone().neg() + margin)
                        .clamp_min(0.0);
                // bottom: cy + r > upper
                total =
                    total + (cy.clone() + r.clone() - upper).clamp_min(0.0);
                // top: cy - r < margin
                total = total + (cy.neg() + r.neg() + margin).clamp_min(0.0);
            }
            ShapeKind::Ellipse(e) => {
                let cx = e.cx();
                let cy = e.cy();
                let rx = e.rx();
                let ry = e.ry();
                let max_r = rx.clone().mask_where(ry.clone().greater(rx.clone()), ry);
                total = total
                    + (cx.clone() + max_r.clone() - upper).clamp_min(0.0);
                total = total
                    + (cx.neg() + max_r.clone().neg() + margin).clamp_min(0.0);
                total = total
                    + (cy.clone() + max_r.clone() - upper).clamp_min(0.0);
                total =
                    total + (cy.neg() + max_r.neg() + margin).clamp_min(0.0);
            }
            ShapeKind::Triangle(t) => {
                for v in [t.v0(), t.v1(), t.v2()] {
                    let x = v.clone().slice([0..1]);
                    let y = v.slice([1..2]);
                    total = total + (x.clone() - upper).clamp_min(0.0);
                    total = total + (x.neg() + margin).clamp_min(0.0);
                    total = total + (y.clone() - upper).clamp_min(0.0);
                    total = total + (y.neg() + margin).clamp_min(0.0);
                }
            }
        }
    }

    total
}

/// Loss weights configuration.
pub struct LossWeights {
    pub perimeter: f32,
    pub degeneracy: f32,
    pub canvas: f32,
    pub edge: f32,
}

impl Default for LossWeights {
    fn default() -> Self {
        Self {
            perimeter: 0.001,
            degeneracy: 0.1,
            canvas: 1.0,
            edge: 0.1,
        }
    }
}

/// Compute total loss with all components.
/// For the spike, we use MSE + edge + priors (skip SSIM for now).
pub fn total_loss<B: Backend>(
    rendered: Tensor<B, 2>,
    target: &Tensor<B, 2>,
    model: &PelicanModel<B>,
    weights: &LossWeights,
) -> Tensor<B, 1> {
    let l_mse = mse_loss(rendered.clone(), target);
    let l_edge = edge_loss(&rendered, target);
    let l_perimeter = perimeter_prior(model);
    let l_degeneracy = triangle_degeneracy_penalty(model, 0.01);
    let l_canvas = on_canvas_penalty(model, 0.05);

    l_mse
        + l_edge * weights.edge
        + l_perimeter * weights.perimeter
        + l_degeneracy * weights.degeneracy
        + l_canvas * weights.canvas
}
