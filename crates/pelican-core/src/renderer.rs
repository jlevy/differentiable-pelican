use burn::prelude::*;

use crate::geometry::{PelicanModel, ShapeKind};
use crate::sdf::{coverage_from_sdf, sdf_circle, sdf_ellipse, sdf_triangle};

/// Create a coordinate grid with (x, y) at each pixel center, both in (0, 1).
/// Uses pixel-center coordinates: (i + 0.5) / N, so tau = 1/N corresponds
/// to exactly one pixel width in normalized coordinates.
pub fn make_grid<B: Backend>(height: usize, width: usize, device: &B::Device) -> Tensor<B, 3> {
    let h = height as f32;
    let w = width as f32;

    let y_vals: Vec<f32> = (0..height).map(|i| (i as f32 + 0.5) / h).collect();
    let x_vals: Vec<f32> = (0..width).map(|i| (i as f32 + 0.5) / w).collect();

    let y_tensor = Tensor::<B, 1>::from_floats(y_vals.as_slice(), device); // [H]
    let x_tensor = Tensor::<B, 1>::from_floats(x_vals.as_slice(), device); // [W]

    // meshgrid: yy[i,j] = y[i], xx[i,j] = x[j]
    let yy = y_tensor.unsqueeze_dim::<2>(1).repeat_dim(1, width); // [H, W]
    let xx = x_tensor.unsqueeze_dim::<2>(0).repeat_dim(0, height); // [H, W]

    // Stack as [H, W, 2] with channel 0 = x, channel 1 = y
    Tensor::stack(vec![xx, yy], 2)
}

/// Render the model onto a canvas using painter's algorithm (back-to-front).
/// Returns a grayscale image tensor [H, W] with values in [0, 1].
pub fn render<B: Backend>(
    model: &PelicanModel<B>,
    height: usize,
    width: usize,
    tau: f32,
    device: &B::Device,
    grid: Option<&Tensor<B, 3>>,
) -> Tensor<B, 2> {
    let owned_grid;
    let grid = match grid {
        Some(g) => g,
        None => {
            owned_grid = make_grid(height, width, device);
            &owned_grid
        }
    };

    // Start with white background
    let mut canvas = Tensor::<B, 2>::ones([height, width], device);

    for shape in &model.shapes {
        let (sdf, intensity) = match shape {
            ShapeKind::Circle(c) => {
                let sdf = sdf_circle(grid, c.cx(), c.cy(), c.radius());
                (sdf, c.intensity())
            }
            ShapeKind::Ellipse(e) => {
                let sdf = sdf_ellipse(grid, e.cx(), e.cy(), e.rx(), e.ry(), e.rotation());
                (sdf, e.intensity())
            }
            ShapeKind::Triangle(t) => {
                let sdf = sdf_triangle(grid, t.v0(), t.v1(), t.v2());
                (sdf, t.intensity())
            }
        };

        let coverage = coverage_from_sdf(sdf, tau);
        // Intensity is [1] — unsqueeze to [1, 1] for broadcast
        let color = intensity.unsqueeze::<2>();

        // Porter-Duff alpha-over: canvas = (1 - coverage) * canvas + coverage * color
        canvas = (Tensor::ones_like(&coverage) - coverage.clone()) * canvas + coverage * color;
    }

    canvas
}

/// Render to a Vec<u8> of grayscale pixel values.
pub fn render_to_pixels<B: Backend>(
    model: &PelicanModel<B>,
    height: usize,
    width: usize,
    tau: f32,
    device: &B::Device,
) -> Vec<u8> {
    let rendered = render(model, height, width, tau, device, None);
    let data = rendered.into_data();
    let floats: Vec<f32> = data.to_vec().unwrap();
    floats
        .iter()
        .map(|&v| (v.clamp(0.0, 1.0) * 255.0) as u8)
        .collect()
}
