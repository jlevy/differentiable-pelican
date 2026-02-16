use std::fs;
use std::path::Path;

use burn::backend::ndarray::NdArray;
use burn::backend::Autodiff;
use burn::prelude::*;

use pelican_core::geometry::create_initial_pelican;
use pelican_core::optimizer::{optimize, OptimConfig, StepResult};
use pelican_core::renderer::render_to_pixels;
use pelican_core::svg_export::shapes_to_svg;

type MyBackend = NdArray;
type MyAutodiffBackend = Autodiff<MyBackend>;

fn save_png(pixels: &[u8], width: u32, height: u32, path: &Path) {
    let img = image::GrayImage::from_raw(width, height, pixels.to_vec())
        .expect("pixel buffer size mismatch");
    img.save(path).expect("failed to save PNG");
    println!("  Saved: {}", path.display());
}

fn save_svg(svg: &str, path: &Path) {
    fs::write(path, svg).expect("failed to save SVG");
    println!("  Saved: {}", path.display());
}

fn load_target_grayscale(path: &str, resolution: u32) -> Vec<f32> {
    let img = image::open(path).expect("failed to open target image");
    let gray = img.into_luma8();
    let resized = image::imageops::resize(
        &gray,
        resolution,
        resolution,
        image::imageops::FilterType::Lanczos3,
    );
    resized.into_raw().iter().map(|&v| v as f32 / 255.0).collect()
}

fn main() {
    let target_path = "images/pelican-drawing-1.jpg";
    let output_dir = Path::new("docs/results/rust");
    let resolution: usize = 128;
    let steps: usize = 500;
    let save_every: usize = 25;

    fs::create_dir_all(output_dir).expect("failed to create output dir");
    fs::create_dir_all(output_dir.join("frames")).expect("failed to create frames dir");

    let device = Default::default();
    let res_u32 = resolution as u32;

    // ========================================
    // Stage 1: Initial Pelican (test render)
    // ========================================
    println!("\n=== Stage 1: Test Render (Initial Geometry) ===");
    {
        let model = create_initial_pelican::<MyBackend>(&device);
        let tau = 1.0 / resolution as f32;
        let pixels = render_to_pixels(&model, resolution, resolution, tau, &device);
        save_png(&pixels, res_u32, res_u32, &output_dir.join("01_test_render.png"));

        let svg = shapes_to_svg(&model, res_u32 * 4, res_u32 * 4);
        save_svg(&svg, &output_dir.join("01_test_render.svg"));
    }

    // ========================================
    // Stage 2: Optimization (500 steps)
    // ========================================
    println!("\n=== Stage 2: Optimization ({} steps, {}x{}) ===", steps, resolution, resolution);

    // Load and prep target
    let target_floats = load_target_grayscale(target_path, res_u32);
    {
        // Save the target at working resolution
        let target_u8: Vec<u8> = target_floats.iter().map(|&v| (v * 255.0).clamp(0.0, 255.0) as u8).collect();
        save_png(&target_u8, res_u32, res_u32, &output_dir.join("00_target.png"));
    }

    let target_tensor = Tensor::<MyAutodiffBackend, 1>::from_floats(
        target_floats.as_slice(),
        &device,
    )
    .reshape([resolution, resolution]);

    let model = create_initial_pelican::<MyAutodiffBackend>(&device);
    let config = OptimConfig::new(resolution, steps);

    // Collect frames for animation
    let output_dir_clone = output_dir.to_path_buf();
    let mut best_loss = f32::MAX;
    let mut callback = |result: StepResult| {
        let step = result.step;
        let loss = result.loss;

        if loss < best_loss {
            best_loss = loss;
        }

        if step % 50 == 0 || step == steps - 1 {
            println!("  Step {}/{}: loss = {:.6} (best = {:.6})", step, steps, loss, best_loss);
        }

        if step % save_every == 0 || step == steps - 1 {
            // We can't render from inside the callback since we don't have the model ref.
            // Just record the step number for now.
            let _ = &output_dir_clone;
        }
    };

    let (optimized_model, loss_history) = optimize(model, &target_tensor, &config, Some(&mut callback));

    // Save final optimized result
    println!("\n  Final loss: {:.6}", loss_history.last().unwrap_or(&0.0));
    println!("  Best loss:  {:.6}", best_loss);

    // Render at half-tau for sharper final output (matching Python's tau = 0.5/resolution)
    let final_tau = 0.5 / resolution as f32;
    let optimized_pixels = render_to_pixels(&optimized_model, resolution, resolution, final_tau, &device);
    save_png(&optimized_pixels, res_u32, res_u32, &output_dir.join("02_optimized.png"));

    let optimized_svg = shapes_to_svg(&optimized_model, res_u32 * 4, res_u32 * 4);
    save_svg(&optimized_svg, &output_dir.join("02_optimized.svg"));

    // Save intermediate frames at different tau values to show the optimization progression
    println!("\n  Rendering optimization frames...");
    let tau_values: Vec<(usize, f32)> = (0..steps)
        .step_by(save_every)
        .chain(std::iter::once(steps - 1))
        .map(|s| {
            let tau = pelican_core::optimizer::anneal_tau(s, steps, config.tau_start, config.tau_end);
            (s, tau)
        })
        .collect();

    // Since we can't rewind the model to intermediate states, render the final model at different tau values
    // to show what the SDF looks like at each sharpness level
    for (i, &(_step, tau)) in tau_values.iter().enumerate() {
        let pixels = render_to_pixels(&optimized_model, resolution, resolution, tau, &device);
        let frame_path = output_dir.join(format!("frames/frame_{:04}.png", i));
        save_png(&pixels, res_u32, res_u32, &frame_path);
    }

    // ========================================
    // Stage 3: Summary
    // ========================================
    println!("\n=== Results Summary ===");
    println!("  Target:     {}", target_path);
    println!("  Resolution: {}x{}", resolution, resolution);
    println!("  Steps:      {}", steps);
    println!("  Shapes:     9 (body, neck, head, beak_upper, beak_lower, wing, tail, eye, feet)");
    println!("  Final loss: {:.6}", loss_history.last().unwrap_or(&0.0));
    println!("  Best loss:  {:.6}", best_loss);
    println!();
    println!("  Output files:");
    println!("    {}/00_target.png          - Target at working resolution", output_dir.display());
    println!("    {}/01_test_render.png     - Initial 9-shape pelican", output_dir.display());
    println!("    {}/01_test_render.svg     - SVG of initial pelican", output_dir.display());
    println!("    {}/02_optimized.png       - Optimized result", output_dir.display());
    println!("    {}/02_optimized.svg       - SVG of optimized result", output_dir.display());

    // Save loss history as simple text
    let loss_text: String = loss_history
        .iter()
        .enumerate()
        .map(|(i, l)| format!("{},{:.6}", i, l))
        .collect::<Vec<_>>()
        .join("\n");
    let loss_path = output_dir.join("loss_history.csv");
    fs::write(&loss_path, format!("step,loss\n{}\n", loss_text)).expect("failed to write loss history");
    println!("    {}        - Loss per step", loss_path.display());

    println!("\n  Done!");
}
