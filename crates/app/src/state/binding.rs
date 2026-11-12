//! Bindings from a control to one value inside an [`AdjustmentSet`], optionally
//! scoped to the selected local layer.

use std::rc::Rc;

use lumen_engine::{AdjustmentSet, LocalAdjustments};

/// Where tool edits land: the background, or one local adjustment layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scope {
    Global,
    Layer(u64),
}

/// Reads and writes one `f32` of an adjustment set.
#[derive(Clone)]
pub struct Binding {
    pub get: Rc<dyn Fn(&AdjustmentSet) -> f32>,
    pub set: Rc<dyn Fn(&mut AdjustmentSet, f32)>,
}

impl Binding {
    pub fn new(get: impl Fn(&AdjustmentSet) -> f32 + 'static, set: impl Fn(&mut AdjustmentSet, f32) + 'static) -> Self {
        Self { get: Rc::new(get), set: Rc::new(set) }
    }
}

/// Binds a contract parameter stored in `AdjustmentSet::extra`; a value equal
/// to the default is removed so untouched images stay default.
pub fn extra_binding(key: impl Into<String>, default: f32) -> Binding {
    let key: Rc<str> = Rc::from(key.into());
    let k2 = key.clone();
    Binding::new(
        move |a: &AdjustmentSet| a.extra.get(&*key).copied().unwrap_or(default),
        move |a: &mut AdjustmentSet, v: f32| {
            if (v - default).abs() < 1e-6 {
                a.extra.remove(&*k2);
            } else {
                a.extra.insert(k2.to_string(), v);
            }
        },
    )
}

/// Binds a background-only field: `global!(geometry.rotation)`.
#[macro_export]
macro_rules! global {
    ($($path:tt)+) => {
        $crate::state::Binding::new(
            move |a: &lumen_engine::AdjustmentSet| a.$($path)+,
            move |a: &mut lumen_engine::AdjustmentSet, v: f32| a.$($path)+ = v,
        )
    };
}

/// Binds a field that local layers also carry: `scoped!(scope, exposure.contrast)`.
#[macro_export]
macro_rules! scoped {
    ($scope:expr, $($path:tt)+) => {{
        let scope: $crate::state::Scope = $scope;
        $crate::state::Binding::new(
            move |a: &lumen_engine::AdjustmentSet| match scope {
                $crate::state::Scope::Global => a.$($path)+,
                $crate::state::Scope::Layer(id) => a
                    .layers
                    .iter()
                    .find(|l| l.id == id)
                    .map(|l| l.adjustments.$($path)+)
                    .unwrap_or_default(),
            },
            move |a: &mut lumen_engine::AdjustmentSet, v: f32| match scope {
                $crate::state::Scope::Global => a.$($path)+ = v,
                $crate::state::Scope::Layer(id) => {
                    if let Some(l) = a.layers.iter_mut().find(|l| l.id == id) {
                        l.adjustments.$($path)+ = v;
                    }
                }
            },
        )
    }};
}

/// Binds a field of the selected layer itself: `layer_field!(id, opacity)`.
#[macro_export]
macro_rules! layer_field {
    ($id:expr, $($path:tt)+) => {{
        let id: u64 = $id;
        $crate::state::Binding::new(
            move |a: &lumen_engine::AdjustmentSet| a.layers.iter().find(|l| l.id == id).map(|l| l.$($path)+).unwrap_or_default(),
            move |a: &mut lumen_engine::AdjustmentSet, v: f32| {
                if let Some(l) = a.layers.iter_mut().find(|l| l.id == id) {
                    l.$($path)+ = v;
                }
            },
        )
    }};
}

/// Runs `f` on the scoped copy of a structure that both the background and layers carry.
pub fn with_scoped<T>(
    a: &mut AdjustmentSet,
    scope: Scope,
    global: impl FnOnce(&mut AdjustmentSet) -> &mut T,
    local: impl FnOnce(&mut LocalAdjustments) -> &mut T,
    f: impl FnOnce(&mut T),
) {
    match scope {
        Scope::Global => f(global(a)),
        Scope::Layer(id) => {
            if let Some(l) = a.layers.iter_mut().find(|l| l.id == id) {
                f(local(&mut l.adjustments));
            }
        }
    }
}

/// Reads the scoped copy of a structure.
pub fn read_scoped<'a, T>(
    a: &'a AdjustmentSet,
    scope: Scope,
    global: impl FnOnce(&'a AdjustmentSet) -> &'a T,
    local: impl FnOnce(&'a LocalAdjustments) -> &'a T,
) -> Option<&'a T> {
    match scope {
        Scope::Global => Some(global(a)),
        Scope::Layer(id) => a.layers.iter().find(|l| l.id == id).map(|l| local(&l.adjustments)),
    }
}
