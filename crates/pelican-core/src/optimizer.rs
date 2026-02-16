use burn::grad_clipping::GradientClippingConfig;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;

use crate::geometry::{PelicanModel, PelicanModelRecord};
use crate::loss::{total_loss, LossWeights};
use crate::renderer::{make_grid, render, render_to_pixels};

/// Exponential tau annealing (log-linear interpolation).
pub fn anneal_tau(step: usize, total_steps: usize, tau_start: f32, tau_end: f32) -> f32 {
    if total_steps == 0 {
        return tau_end;
    }
    let progress = step as f32 / total_steps as f32;
    tau_start * (tau_end / tau_start).powf(progress)
}

/// Cosine learning rate annealing.
pub fn cosine_lr(step: usize, total_steps: usize, lr_start: f32, lr_end: f32) -> f64 {
    if total_steps == 0 {
        return lr_end as f64;
    }
    let progress = step as f32 / total_steps as f32;
    let lr = lr_end + 0.5 * (lr_start - lr_end) * (1.0 + (std::f32::consts::PI * progress).cos());
    lr as f64
}

/// Optimization configuration.
pub struct OptimConfig {
    pub resolution: usize,
    pub steps: usize,
    pub lr: f32,
    pub lr_end: f32,
    pub tau_start: f32,
    pub tau_end: f32,
    pub loss_weights: LossWeights,
    pub grad_clip_norm: f32,
}

impl OptimConfig {
    pub fn new(resolution: usize, steps: usize) -> Self {
        let lr = 0.02;
        Self {
            resolution,
            steps,
            lr,
            lr_end: lr / 20.0,
            tau_start: 1.0 / resolution as f32,
            tau_end: 0.2 / resolution as f32,
            loss_weights: LossWeights::default(),
            grad_clip_norm: 1.0,
        }
    }
}

/// Result of a single optimization step.
pub struct StepResult {
    pub loss: f32,
    pub step: usize,
}

/// A rendered frame snapshot from during optimization.
pub struct Snapshot {
    pub step: usize,
    pub loss: f32,
    pub pixels: Vec<u8>,
}

/// Full optimization result including snapshots.
pub struct OptimResult<B: Backend> {
    pub model: PelicanModel<B>,
    pub loss_history: Vec<f32>,
    pub snapshots: Vec<Snapshot>,
}

/// Run the full optimization loop.
///
/// Includes NaN safety: detects NaN loss before backward pass and NaN
/// gradients before parameter update. Tracks and restores best model state.
/// Saves model snapshots every `save_every` steps (0 = no snapshots).
pub fn optimize<B: AutodiffBackend>(
    model: PelicanModel<B>,
    target: &Tensor<B, 2>,
    config: &OptimConfig,
    save_every: usize,
    mut progress_callback: Option<&mut dyn FnMut(StepResult)>,
) -> OptimResult<B> {
    let device = model.devices()[0].clone();
    let grid = make_grid(config.resolution, config.resolution, &device);

    let adam_config = AdamConfig::new()
        .with_grad_clipping(Some(GradientClippingConfig::Norm(config.grad_clip_norm)));
    let mut optim = adam_config.init::<B, PelicanModel<B>>();

    let mut model = model;
    let mut loss_history = Vec::with_capacity(config.steps);
    let mut best_loss = f32::INFINITY;
    let mut best_record: Option<PelicanModelRecord<B>> = None;
    let mut snapshots = Vec::new();

    for step in 0..config.steps {
        let tau = anneal_tau(step, config.steps, config.tau_start, config.tau_end);
        let lr = cosine_lr(step, config.steps, config.lr, config.lr_end);

        // Forward pass
        let rendered = render(&model, config.resolution, config.resolution, tau, &device, Some(&grid));
        let loss = total_loss(rendered, target, &model, &config.loss_weights);

        // Record loss value
        let loss_val: f32 = loss.clone().into_data().to_vec::<f32>().unwrap()[0];
        loss_history.push(loss_val);

        if let Some(ref mut cb) = progress_callback {
            cb(StepResult { loss: loss_val, step });
        }

        // Check for NaN before backward pass (NaN gradients corrupt params irreversibly)
        if loss_val.is_nan() {
            eprintln!("Warning: NaN loss at step {step}, stopping");
            break;
        }

        // Save rendered frame snapshot at regular intervals
        if save_every > 0 && (step % save_every == 0 || step == config.steps - 1) {
            let pixels = render_to_pixels(&model, config.resolution, config.resolution, tau, &device);
            snapshots.push(Snapshot {
                step,
                loss: loss_val,
                pixels,
            });
        }

        // Track best model state
        if loss_val < best_loss {
            best_loss = loss_val;
            best_record = Some(model.clone().into_record());
        }

        // Backward + update
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        model = optim.step(lr, model, grads);
    }

    // Restore best parameters if we have them
    if let Some(record) = best_record {
        model = model.load_record(record);
    }

    OptimResult {
        model,
        loss_history,
        snapshots,
    }
}
