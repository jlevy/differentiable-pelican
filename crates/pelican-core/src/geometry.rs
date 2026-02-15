use burn::module::{Module, Param, ParamId};
use burn::prelude::*;
use burn::tensor::activation::sigmoid;

/// Inverse sigmoid: maps (0,1) to unconstrained reals.
pub fn logit_param(x: f32, eps: f32) -> f32 {
    let x = x.clamp(eps, 1.0 - eps);
    (x / (1.0 - x)).ln()
}

/// Inverse softplus: maps (0, inf) to unconstrained reals.
pub fn inv_softplus(x: f32, eps: f32) -> f32 {
    let x = x.max(eps);
    (x.exp() - 1.0).ln()
}

/// Softplus activation: ln(1 + exp(x)).
pub fn softplus<B: Backend, const D: usize>(tensor: Tensor<B, D>) -> Tensor<B, D> {
    tensor.exp().log1p()
}

fn pt<B: Backend>(vals: &[f32], device: &B::Device) -> Param<Tensor<B, 1>> {
    Param::initialized(ParamId::new(), Tensor::from_floats(vals, device))
}

#[derive(Module, Debug)]
pub struct CircleShape<B: Backend> {
    pub cx_raw: Param<Tensor<B, 1>>,
    pub cy_raw: Param<Tensor<B, 1>>,
    pub radius_raw: Param<Tensor<B, 1>>,
    pub intensity_raw: Param<Tensor<B, 1>>,
}

impl<B: Backend> CircleShape<B> {
    pub fn new(cx: f32, cy: f32, radius: f32, intensity: f32, device: &B::Device) -> Self {
        let e = 1e-6;
        let intensity = intensity.clamp(0.01, 0.99);
        Self {
            cx_raw: pt(&[logit_param(cx, e)], device),
            cy_raw: pt(&[logit_param(cy, e)], device),
            radius_raw: pt(&[inv_softplus(radius, e)], device),
            intensity_raw: pt(&[logit_param(intensity, e)], device),
        }
    }

    pub fn cx(&self) -> Tensor<B, 1> { sigmoid(self.cx_raw.val()) }
    pub fn cy(&self) -> Tensor<B, 1> { sigmoid(self.cy_raw.val()) }
    pub fn radius(&self) -> Tensor<B, 1> { softplus(self.radius_raw.val()) }
    pub fn intensity(&self) -> Tensor<B, 1> { sigmoid(self.intensity_raw.val()) }
}

#[derive(Module, Debug)]
pub struct EllipseShape<B: Backend> {
    pub cx_raw: Param<Tensor<B, 1>>,
    pub cy_raw: Param<Tensor<B, 1>>,
    pub rx_raw: Param<Tensor<B, 1>>,
    pub ry_raw: Param<Tensor<B, 1>>,
    pub rotation_raw: Param<Tensor<B, 1>>,
    pub intensity_raw: Param<Tensor<B, 1>>,
}

impl<B: Backend> EllipseShape<B> {
    pub fn new(cx: f32, cy: f32, rx: f32, ry: f32, rotation: f32, intensity: f32, device: &B::Device) -> Self {
        let e = 1e-6;
        let intensity = intensity.clamp(0.01, 0.99);
        Self {
            cx_raw: pt(&[logit_param(cx, e)], device),
            cy_raw: pt(&[logit_param(cy, e)], device),
            rx_raw: pt(&[inv_softplus(rx, e)], device),
            ry_raw: pt(&[inv_softplus(ry, e)], device),
            rotation_raw: pt(&[rotation], device),
            intensity_raw: pt(&[logit_param(intensity, e)], device),
        }
    }

    pub fn cx(&self) -> Tensor<B, 1> { sigmoid(self.cx_raw.val()) }
    pub fn cy(&self) -> Tensor<B, 1> { sigmoid(self.cy_raw.val()) }
    pub fn rx(&self) -> Tensor<B, 1> { softplus(self.rx_raw.val()) }
    pub fn ry(&self) -> Tensor<B, 1> { softplus(self.ry_raw.val()) }
    pub fn rotation(&self) -> Tensor<B, 1> { self.rotation_raw.val() }
    pub fn intensity(&self) -> Tensor<B, 1> { sigmoid(self.intensity_raw.val()) }
}

#[derive(Module, Debug)]
pub struct TriangleShape<B: Backend> {
    pub v0_raw: Param<Tensor<B, 1>>,
    pub v1_raw: Param<Tensor<B, 1>>,
    pub v2_raw: Param<Tensor<B, 1>>,
    pub intensity_raw: Param<Tensor<B, 1>>,
}

impl<B: Backend> TriangleShape<B> {
    pub fn new(v0: [f32; 2], v1: [f32; 2], v2: [f32; 2], intensity: f32, device: &B::Device) -> Self {
        let e = 1e-6;
        let intensity = intensity.clamp(0.01, 0.99);
        Self {
            v0_raw: pt(&[logit_param(v0[0], e), logit_param(v0[1], e)], device),
            v1_raw: pt(&[logit_param(v1[0], e), logit_param(v1[1], e)], device),
            v2_raw: pt(&[logit_param(v2[0], e), logit_param(v2[1], e)], device),
            intensity_raw: pt(&[logit_param(intensity, e)], device),
        }
    }

    pub fn v0(&self) -> Tensor<B, 1> { sigmoid(self.v0_raw.val()) }
    pub fn v1(&self) -> Tensor<B, 1> { sigmoid(self.v1_raw.val()) }
    pub fn v2(&self) -> Tensor<B, 1> { sigmoid(self.v2_raw.val()) }
    pub fn intensity(&self) -> Tensor<B, 1> { sigmoid(self.intensity_raw.val()) }
}

#[derive(Module, Debug)]
pub enum ShapeKind<B: Backend> {
    Circle(CircleShape<B>),
    Ellipse(EllipseShape<B>),
    Triangle(TriangleShape<B>),
}

impl<B: Backend> ShapeKind<B> {
    pub fn intensity(&self) -> Tensor<B, 1> {
        match self {
            ShapeKind::Circle(c) => c.intensity(),
            ShapeKind::Ellipse(e) => e.intensity(),
            ShapeKind::Triangle(t) => t.intensity(),
        }
    }
}

#[derive(Module, Debug)]
pub struct PelicanModel<B: Backend> {
    pub shapes: Vec<ShapeKind<B>>,
}

/// Create the initial 9-shape pelican geometry.
pub fn create_initial_pelican<B: Backend>(device: &B::Device) -> PelicanModel<B> {
    PelicanModel {
        shapes: vec![
            ShapeKind::Ellipse(EllipseShape::new(0.42, 0.55, 0.22, 0.28, -0.3, 0.35, device)),
            ShapeKind::Ellipse(EllipseShape::new(0.52, 0.35, 0.06, 0.15, -0.2, 0.40, device)),
            ShapeKind::Circle(CircleShape::new(0.58, 0.18, 0.08, 0.35, device)),
            ShapeKind::Triangle(TriangleShape::new([0.62, 0.15], [0.62, 0.22], [0.88, 0.20], 0.25, device)),
            ShapeKind::Triangle(TriangleShape::new([0.62, 0.22], [0.88, 0.20], [0.65, 0.28], 0.30, device)),
            ShapeKind::Ellipse(EllipseShape::new(0.38, 0.50, 0.18, 0.15, -0.4, 0.30, device)),
            ShapeKind::Triangle(TriangleShape::new([0.18, 0.52], [0.25, 0.48], [0.12, 0.60], 0.20, device)),
            ShapeKind::Circle(CircleShape::new(0.60, 0.16, 0.015, 0.05, device)),
            ShapeKind::Ellipse(EllipseShape::new(0.45, 0.88, 0.06, 0.04, 0.0, 0.15, device)),
        ],
    }
}
