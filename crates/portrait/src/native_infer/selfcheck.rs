//! A known-answer device check, independent of installed models. Both inputs go through
//! the production dispatcher, with the same compiled session. This is not a model-suite score.
use super::{
    backend::{BackendKind, BackendRequest, NamedInputs, NamedOutputs, ShapeOverrides},
    dispatch,
};
use std::time::Instant;

/// Programmatic ONNX fixture: X[2,3] * W[3,2] + bias[2] -> Relu -> out[2,2].
/// IR v8, default domain opset 13. The integers and weights below are our own.
pub(crate) fn model() -> Vec<u8> {
    fn vint(mut n: usize, out: &mut Vec<u8>) {
        while n >= 128 {
            out.push((n as u8 & 127) | 128);
            n >>= 7;
        }
        out.push(n as u8);
    }
    fn field(out: &mut Vec<u8>, n: u8, bytes: &[u8]) {
        out.push(n * 8 + 2);
        vint(bytes.len(), out);
        out.extend_from_slice(bytes);
    }
    fn val(name: &str, dims: &[u8]) -> Vec<u8> {
        let mut shape = Vec::new();
        for &d in dims {
            field(&mut shape, 1, &[8, d]);
        }
        let mut tensor = vec![8, 1];
        field(&mut tensor, 2, &shape);
        let mut ty = Vec::new();
        field(&mut ty, 1, &tensor);
        let mut v = Vec::new();
        field(&mut v, 1, name.as_bytes());
        field(&mut v, 2, &ty);
        v
    }
    fn weight(name: &str, dims: &[u8], data: &[f32]) -> Vec<u8> {
        let mut w = Vec::new();
        for &d in dims {
            w.extend_from_slice(&[8, d]);
        }
        w.extend_from_slice(&[16, 1]);
        field(&mut w, 8, name.as_bytes());
        let raw = data
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect::<Vec<_>>();
        field(&mut w, 9, &raw);
        w
    }
    let mut g = Vec::new();
    field(&mut g, 2, b"aurora_public_device_check");
    field(&mut g, 11, &val("x", &[2, 3]));
    field(&mut g, 12, &val("out", &[2, 2]));
    field(&mut g, 5, &weight("w", &[3, 2], &[1., 0., 0., 1., -1., 2.]));
    field(&mut g, 5, &weight("bias", &[2], &[0.5, -1.]));
    for (op, inputs, output) in [
        ("MatMul", &["x", "w"][..], "prod"),
        ("Add", &["prod", "bias"][..], "biased"),
        ("Relu", &["biased"][..], "out"),
    ] {
        let mut n = Vec::new();
        for i in inputs {
            field(&mut n, 1, i.as_bytes());
        }
        field(&mut n, 2, output.as_bytes());
        field(&mut n, 3, op.as_bytes());
        field(&mut n, 4, op.as_bytes());
        field(&mut g, 1, &n);
    }
    let mut model = vec![8, 8];
    field(&mut model, 7, &g);
    field(&mut model, 8, &[16, 13]);
    model
}

pub struct CheckedDevice {
    pub actual: BackendKind,
    pub adapter: Option<String>,
    pub first_us: u64,
    pub warm_us: u64,
    pub prior: Vec<(BackendKind, String)>,
}

fn check(out: NamedOutputs, wanted: [f32; 4]) -> Result<(), String> {
    let (real, dims) = out.get("out").ok_or("missing output named 'out'")?;
    if out.len() != 1 || dims != &[2, 2] || real.as_slice() != wanted.as_slice() {
        return Err("matrix / broadcast / activation results differ from known answers".into());
    }
    Ok(())
}

pub fn run(request: BackendRequest) -> Result<CheckedDevice, String> {
    let a = [1., 2., 3., 4., 5., 6.];
    let b = [-2., 0., -3., 4., 1., -1.];
    let input = |v| NamedInputs::from([("x".into(), (v, vec![2, 3]))]);
    let bytes = model();
    let t = Instant::now();
    let (route, ()) =
        dispatch::open_and_run(request, &bytes, &ShapeOverrides::new(), &input(&a), |out| {
            check(out, [0., 7., 0., 16.])
        })
        .map_err(|e| e.to_string())?;
    let first_us = t.elapsed().as_micros() as u64;
    let t = Instant::now();
    check(route.session.run_f32(&input(&b))?, [1.5, 0., 5.5, 0.])?;
    Ok(CheckedDevice {
        actual: route.session.kind(),
        adapter: route.session.device_detail(),
        first_us,
        warm_us: t.elapsed().as_micros() as u64,
        prior: route.previous_attempts,
    })
}
