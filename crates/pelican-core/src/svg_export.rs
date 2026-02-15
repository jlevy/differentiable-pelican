use burn::prelude::*;

use crate::geometry::{PelicanModel, ShapeKind};

/// Convert the model's shapes to an SVG string.
pub fn shapes_to_svg<B: Backend>(model: &PelicanModel<B>, width: u32, height: u32) -> String {
    let w = width as f32;
    let h = height as f32;

    let mut elements = String::new();

    // White background
    elements.push_str(&format!(
        "  <rect width=\"{}\" height=\"{}\" fill=\"white\"/>\n",
        width, height
    ));

    for shape in &model.shapes {
        let elem = match shape {
            ShapeKind::Circle(c) => {
                let cx = scalar_f32(&c.cx()) * w;
                let cy = scalar_f32(&c.cy()) * h;
                let r = scalar_f32(&c.radius()) * w;
                let fill = intensity_to_fill(&c.intensity());
                format!(
                    "  <circle cx=\"{:.1}\" cy=\"{:.1}\" r=\"{:.1}\" fill=\"{}\"/>",
                    cx, cy, r, fill
                )
            }
            ShapeKind::Ellipse(e) => {
                let cx = scalar_f32(&e.cx()) * w;
                let cy = scalar_f32(&e.cy()) * h;
                let rx = scalar_f32(&e.rx()) * w;
                let ry = scalar_f32(&e.ry()) * h;
                let rot_deg = scalar_f32(&e.rotation()) * 180.0 / std::f32::consts::PI;
                let fill = intensity_to_fill(&e.intensity());
                format!(
                    "  <ellipse cx=\"{:.1}\" cy=\"{:.1}\" rx=\"{:.1}\" ry=\"{:.1}\" fill=\"{}\" transform=\"rotate({:.1} {:.1} {:.1})\"/>",
                    cx, cy, rx, ry, fill, rot_deg, cx, cy
                )
            }
            ShapeKind::Triangle(t) => {
                let v0 = get_vertex_f32(&t.v0(), w, h);
                let v1 = get_vertex_f32(&t.v1(), w, h);
                let v2 = get_vertex_f32(&t.v2(), w, h);
                let fill = intensity_to_fill(&t.intensity());
                format!(
                    "  <polygon points=\"{:.1},{:.1} {:.1},{:.1} {:.1},{:.1}\" fill=\"{}\"/>",
                    v0.0, v0.1, v1.0, v1.1, v2.0, v2.1, fill
                )
            }
        };
        elements.push_str(&elem);
        elements.push('\n');
    }

    format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {} {}\" width=\"{}\" height=\"{}\">\n{}</svg>\n",
        width, height, width, height, elements
    )
}

fn scalar_f32<B: Backend>(tensor: &Tensor<B, 1>) -> f32 {
    tensor.clone().into_data().to_vec::<f32>().unwrap()[0]
}

fn get_vertex_f32<B: Backend>(v: &Tensor<B, 1>, w: f32, h: f32) -> (f32, f32) {
    let data: Vec<f32> = v.clone().into_data().to_vec().unwrap();
    (data[0] * w, data[1] * h)
}

fn intensity_to_fill<B: Backend>(intensity: &Tensor<B, 1>) -> String {
    let val = (scalar_f32(intensity) * 255.0).clamp(0.0, 255.0) as u8;
    format!("rgb({},{},{})", val, val, val)
}
