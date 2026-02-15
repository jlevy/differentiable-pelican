use wasm_bindgen::prelude::*;

use burn::backend::ndarray::NdArray;
use burn::backend::Autodiff;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;

use pelican_core::geometry::{create_initial_pelican, PelicanModel};
use pelican_core::loss::{total_loss, LossWeights};
use pelican_core::optimizer::{anneal_tau, cosine_lr};
use pelican_core::renderer::{make_grid, render, render_to_pixels};
use pelican_core::svg_export::shapes_to_svg;

type B = Autodiff<NdArray>;

/// Trait to erase the concrete optimizer type (which uses private burn internals).
trait Stepper {
    fn step(&mut self) -> f32;
    fn get_rendered_pixels(&self, width: usize, height: usize) -> Vec<u8>;
    fn get_svg(&self, width: u32, height: u32) -> String;
    fn current_step(&self) -> usize;
    fn total_steps(&self) -> usize;
    fn is_done(&self) -> bool;
}

/// Concrete stepper that holds the optimizer, model, and config.
/// Generic over `O` so we never need to name the private Adam adaptor type.
struct ConcreteStepper<O: Optimizer<PelicanModel<B>, B>> {
    model: PelicanModel<B>,
    optim: O,
    target: Tensor<B, 2>,
    resolution: usize,
    total_steps: usize,
    current_step: usize,
    lr_start: f32,
    lr_end: f32,
    tau_start: f32,
    tau_end: f32,
    loss_weights: LossWeights,
}

impl<O: Optimizer<PelicanModel<B>, B>> Stepper for ConcreteStepper<O> {
    fn step(&mut self) -> f32 {
        let device = Default::default();

        let tau = anneal_tau(self.current_step, self.total_steps, self.tau_start, self.tau_end);
        let lr = cosine_lr(self.current_step, self.total_steps, self.lr_start, self.lr_end);

        let grid = make_grid(self.resolution, self.resolution, &device);
        let rendered = render(
            &self.model,
            self.resolution,
            self.resolution,
            tau,
            &device,
            Some(&grid),
        );
        let loss = total_loss(rendered, &self.target, &self.model, &self.loss_weights);
        let loss_val: f32 = loss.clone().into_data().to_vec::<f32>().unwrap()[0];

        if !loss_val.is_nan() {
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &self.model);
            self.model = self.optim.step(lr, self.model.clone(), grads);
        }

        self.current_step += 1;
        loss_val
    }

    fn get_rendered_pixels(&self, width: usize, height: usize) -> Vec<u8> {
        let device = Default::default();
        let tau = anneal_tau(
            self.current_step.saturating_sub(1),
            self.total_steps,
            self.tau_start,
            self.tau_end,
        );
        render_to_pixels(&self.model, height, width, tau, &device)
    }

    fn get_svg(&self, width: u32, height: u32) -> String {
        shapes_to_svg(&self.model, width, height)
    }

    fn current_step(&self) -> usize {
        self.current_step
    }

    fn total_steps(&self) -> usize {
        self.total_steps
    }

    fn is_done(&self) -> bool {
        self.current_step >= self.total_steps
    }
}

/// Helper to create a ConcreteStepper without naming the optimizer type.
fn create_stepper(
    target: Tensor<B, 2>,
    resolution: usize,
    total_steps: usize,
) -> Box<dyn Stepper> {
    let device = Default::default();
    let model = create_initial_pelican::<B>(&device);

    let lr_start = 0.02_f32;
    let lr_end = lr_start / 20.0;
    let tau_start = 1.0 / resolution as f32;
    let tau_end = 0.2 / resolution as f32;

    let optim = AdamConfig::new().init::<B, PelicanModel<B>>();

    Box::new(ConcreteStepper {
        model,
        optim,
        target,
        resolution,
        total_steps,
        current_step: 0,
        lr_start,
        lr_end,
        tau_start,
        tau_end,
        loss_weights: LossWeights::default(),
    })
}

/// The main optimizer exposed to JavaScript via wasm-bindgen.
#[wasm_bindgen]
pub struct PelicanOptimizer {
    stepper: Box<dyn Stepper>,
}

#[wasm_bindgen]
impl PelicanOptimizer {
    /// Create a new optimizer from a target image (grayscale u8 pixels, row-major).
    #[wasm_bindgen(constructor)]
    pub fn new(target_pixels: &[u8], width: u32, height: u32, total_steps: u32) -> Self {
        let device = Default::default();
        let h = height as usize;
        let w = width as usize;

        let floats: Vec<f32> = target_pixels.iter().map(|&v| v as f32 / 255.0).collect();
        let target = Tensor::<B, 1>::from_floats(floats.as_slice(), &device).reshape([h, w]);

        Self {
            stepper: create_stepper(target, w, total_steps as usize),
        }
    }

    /// Run one optimization step. Returns the loss value.
    pub fn step(&mut self) -> f32 {
        self.stepper.step()
    }

    /// Run multiple optimization steps. Returns array of loss values.
    pub fn step_n(&mut self, n: u32) -> Vec<f32> {
        let mut losses = Vec::with_capacity(n as usize);
        for _ in 0..n {
            if self.stepper.is_done() {
                break;
            }
            losses.push(self.stepper.step());
        }
        losses
    }

    /// Render the current model state as grayscale u8 pixels.
    pub fn get_rendered_pixels(&self, width: u32, height: u32) -> Vec<u8> {
        self.stepper.get_rendered_pixels(width as usize, height as usize)
    }

    /// Get SVG representation of the current model.
    pub fn get_svg(&self, width: u32, height: u32) -> String {
        self.stepper.get_svg(width, height)
    }

    /// Get the current step number.
    pub fn current_step(&self) -> u32 {
        self.stepper.current_step() as u32
    }

    /// Check if optimization is complete.
    pub fn is_done(&self) -> bool {
        self.stepper.is_done()
    }

    /// Get total steps.
    pub fn total_steps(&self) -> u32 {
        self.stepper.total_steps() as u32
    }
}

/// Render the initial pelican as grayscale u8 pixels (no optimization).
#[wasm_bindgen]
pub fn render_initial_pelican(width: u32, height: u32) -> Vec<u8> {
    let device = Default::default();
    let model = create_initial_pelican::<NdArray>(&device);
    let tau = 1.0 / width as f32;
    render_to_pixels(&model, height as usize, width as usize, tau, &device)
}

/// Get SVG of the initial pelican.
#[wasm_bindgen]
pub fn initial_pelican_svg(width: u32, height: u32) -> String {
    let device = Default::default();
    let model = create_initial_pelican::<NdArray>(&device);
    shapes_to_svg(&model, width, height)
}
