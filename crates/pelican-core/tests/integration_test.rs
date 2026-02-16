use burn::backend::ndarray::NdArray;
use burn::backend::Autodiff;

use pelican_core::geometry::create_initial_pelican;
use pelican_core::optimizer::{optimize, OptimConfig};
use pelican_core::renderer::render;
use pelican_core::svg_export::shapes_to_svg;

type MyBackend = Autodiff<NdArray>;

#[test]
fn test_render_initial_pelican() {
    let device = Default::default();
    let model = create_initial_pelican::<NdArray>(&device);

    let rendered = render(&model, 64, 64, 0.01, &device, None);
    let data = rendered.into_data();
    let pixels: Vec<f32> = data.to_vec().unwrap();

    assert_eq!(pixels.len(), 64 * 64);
    // All values should be in [0, 1]
    for &v in &pixels {
        assert!(v >= 0.0 && v <= 1.0, "pixel value out of range: {}", v);
    }
    // Not all white (should have some shape coverage)
    let non_white = pixels.iter().filter(|&&v| v < 0.99).count();
    assert!(non_white > 100, "expected visible shapes, got {} non-white pixels", non_white);
}

#[test]
fn test_svg_export() {
    let device = Default::default();
    let model = create_initial_pelican::<NdArray>(&device);

    let svg = shapes_to_svg(&model, 128, 128);
    assert!(svg.contains("<svg"));
    assert!(svg.contains("<circle"));
    assert!(svg.contains("<ellipse"));
    assert!(svg.contains("<polygon"));
    assert!(svg.contains("</svg>"));
}

#[test]
fn test_optimize_gradients_flow() {
    let device = Default::default();

    // Use a simple gray target (0.5 everywhere) — different from the initial pelican,
    // so the optimizer has something to work towards
    let target = burn::tensor::Tensor::<MyBackend, 2>::zeros([32, 32], &device) + 0.5;

    let model = create_initial_pelican::<MyBackend>(&device);
    let config = OptimConfig::new(32, 10);

    let result = optimize(model, &target, &config, 0, None);
    let loss_history = result.loss_history;

    assert_eq!(loss_history.len(), 10, "should complete all 10 steps");

    // All losses should be finite (gradients flow without NaN)
    for (i, &l) in loss_history.iter().enumerate() {
        assert!(!l.is_nan(), "loss at step {} is NaN", i);
        assert!(l.is_finite(), "loss at step {} is not finite", i);
    }

    // Loss should change (not stuck) — proves gradients are flowing
    let first = loss_history[0];
    let last = loss_history[loss_history.len() - 1];
    assert!(
        (first - last).abs() > 1e-6,
        "loss should change during optimization: first={:.6}, last={:.6}",
        first,
        last
    );

    println!("Optimization test passed: loss went from {:.6} to {:.6}", first, last);
}
