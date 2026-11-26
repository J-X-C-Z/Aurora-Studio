//! Platform-neutral native session contract. A requested backend is never evidence that it executed.
use std::{collections::HashMap, fmt, str::FromStr};

pub type ShapeOverrides = HashMap<String, Vec<usize>>;
pub type NamedInputs<'a> = HashMap<String, (&'a [f32], Vec<usize>)>;
pub type NamedOutputs = HashMap<String, (Vec<f32>, Vec<usize>)>;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum BackendKind {
    Cuda,
    Metal,
    Dx12,
    OpenCl,
    Cpu,
}
impl BackendKind {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Cuda => "cuda",
            Self::Metal => "metal",
            Self::Dx12 => "dx12",
            Self::OpenCl => "opencl",
            Self::Cpu => "cpu",
        }
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum BackendRequest {
    Auto,
    Only(BackendKind),
}
impl FromStr for BackendRequest {
    type Err = String;
    fn from_str(text: &str) -> Result<Self, String> {
        let k = match text.trim().to_ascii_lowercase().as_str() {
            "auto" => return Ok(Self::Auto),
            "cuda" => BackendKind::Cuda,
            "metal" => BackendKind::Metal,
            "dx12" => BackendKind::Dx12,
            "opencl" => BackendKind::OpenCl,
            "cpu" => BackendKind::Cpu,
            other => {
                return Err(format!(
                    "invalid AURORA_GPU_BACKEND={other:?}; expected auto, cuda, metal, dx12, opencl or cpu"
                ));
            }
        };
        Ok(Self::Only(k))
    }
}
impl fmt::Display for BackendKind {fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt::Result{f.write_str(self.label())}}
impl BackendRequest {
    pub fn parse(text:&str)->Result<Self,String>{text.parse()}
    pub fn permits(self,kind:BackendKind)->bool{matches!(self,Self::Auto) || self==Self::Only(kind)}

    pub fn from_env() -> Result<Self, String> {
        match std::env::var("AURORA_GPU_BACKEND") {
            Ok(s) => s.parse(),
            Err(std::env::VarError::NotPresent) => Ok(Self::Auto),
            Err(_) => Err("AURORA_GPU_BACKEND is not valid UTF-8".into()),
        }
    }
    pub const fn label(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Only(k) => k.label(),
        }
    }
}

/// A compiled model, not a silicon inventory. Construction must reject a graph with unsupported
/// operations before first dispatch. Named model inputs and outputs retain their dimensions.
pub trait NativeInferBackend: Send + Sync {
    fn kind(&self) -> BackendKind;
    /// Adapter identity supplied by the opened session, never guessed from a platform label.
    fn device_detail(&self) -> Option<String> { None }
    fn arena_peak_bytes(&self) -> usize;
    fn run_f32(&self, inputs: &NamedInputs<'_>) -> Result<NamedOutputs, String>;
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Host {
    Windows,
    MacOs,
    Other,
}
impl Host {
    /// Ordered candidates, not a promise that their drivers or model kernels exist.
    pub const fn auto_order(self) -> &'static [BackendKind] {
        use BackendKind::*;
        match self {
            Self::Windows => &[Cuda, Dx12, OpenCl, Cpu],
            Self::MacOs => &[Metal, Cpu],
            Self::Other => &[Cpu],
        }
    }
    pub const fn current() -> Self {
        if cfg!(windows) {
            Self::Windows
        } else if cfg!(target_os = "macos") {
            Self::MacOs
        } else {
            Self::Other
        }
    }
}
#[derive(Debug)]
pub struct RouteError(pub Vec<(BackendKind, String)>);
impl fmt::Display for RouteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, (k, e)) in self.0.iter().enumerate() {
            if i != 0 {
                f.write_str("; ")?;
            }
            write!(f, "{}: {e}", k.label())?;
        }
        Ok(())
    }
}
impl std::error::Error for RouteError {}

/// The probe must open/compile this model on the actual backend. A strict request invokes the
/// factory exactly once and propagates its error. Auto records why earlier backends failed.
pub fn select<T>(
    request: BackendRequest,
    host: Host,
    mut open: impl FnMut(BackendKind) -> Result<T, String>,
) -> Result<(T, BackendKind, Vec<(BackendKind, String)>), RouteError> {
    let only;
    let order: &[BackendKind] = match request {
        BackendRequest::Only(k) => {
            only = [k];
            &only
        }
        BackendRequest::Auto => host.auto_order(),
    };
    let mut failures = Vec::new();
    for &k in order {
        match open(k) {
            Ok(v) => return Ok((v, k, failures)),
            Err(reason) => failures.push((k, reason)),
        }
    }
    Err(RouteError(failures))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn explicit_fails_instead_of_measuring_cpu() {
        let err = select::<()>(
            BackendRequest::Only(BackendKind::Dx12),
            Host::Windows,
            |k| {
                assert_eq!(k, BackendKind::Dx12);
                Err("unsupported graph".into())
            },
        )
        .unwrap_err();
        assert_eq!(err.0.len(), 1);
    }
    #[test]
    fn hybrid_falls_through_a_real_failed_probe() {
        let (_, actual, failures) = select(BackendRequest::Auto, Host::Windows, |k| {
            if k == BackendKind::Dx12 {
                Ok(())
            } else {
                Err("no device".into())
            }
        })
        .unwrap();
        assert_eq!(actual, BackendKind::Dx12);
        assert_eq!(failures[0].0, BackendKind::Cuda);
    }
    #[test]
    fn mac_does_not_probe_windows_drivers() {
        let mut seen = Vec::new();
        let e = select::<()>(BackendRequest::Auto, Host::MacOs, |k| {
            seen.push(k);
            Err("not compiled".into())
        })
        .unwrap_err();
        assert_eq!(seen, [BackendKind::Metal, BackendKind::Cpu]);
        assert_eq!(e.0.len(), 2);
    }
    #[test]
    fn all_six_values_roundtrip() {
        for s in ["auto", "cuda", "metal", "dx12", "opencl", "cpu"] {
            assert_eq!(s.parse::<BackendRequest>().unwrap().label(), s);
        }
        assert!("gpu".parse::<BackendRequest>().is_err());
        assert!("".parse::<BackendRequest>().is_err());
    }
}
