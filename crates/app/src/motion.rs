//! Aurora's motion vocabulary. Every animated value goes through a keyed spring so
//! it is frame-rate independent, reverses smoothly when retargeted mid-flight, and
//! honors the system's reduced-motion preference.
//!
//! Each feel is a response time and a damping ratio. Controls that report a value
//! (slider knobs, zoom) are critically damped so they never overshoot the number
//! they show; things that move through space (panels, tiles, the loupe) carry a
//! touch of underdamping so they settle the way a physical object would.
//! Preferences → Appearance scales every response time, or turns motion off.

use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

use gpui_kit::base::{Spring, spring};
use gpui_kit::{App, ElementId, Window};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Feel {
    ms: f32,
    damping: f32,
}

impl Feel {
    const fn new(ms: f32, damping: f32) -> Self {
        Self { ms, damping }
    }

    fn spring(self) -> Spring {
        let ms = (self.ms * speed()).max(1.0);
        Spring::new(Duration::from_secs_f32(ms / 1000.0)).with_damping(self.damping).with_epsilon(0.0005)
    }
}

/// Hover and press feedback.
pub const QUICK: Feel = Feel::new(140.0, 1.0);
/// Knobs gliding to a new value, indicators following a selection.
pub const GLIDE: Feel = Feel::new(220.0, 1.0);
/// Panels opening and closing.
pub const PANEL: Feel = Feel::new(280.0, 0.9);
/// Viewer zoom and pan.
pub const VIEW: Feel = Feel::new(240.0, 1.0);
/// Objects moving to a new place: tiles reflowing, rows reordering, the loupe.
pub const FLOW: Feel = Feel::new(320.0, 0.82);

/// Response-time multiplier; 0 turns animation off.
static SPEED: AtomicU32 = AtomicU32::new(f32::to_bits(1.0));

pub fn speed() -> f32 {
    f32::from_bits(SPEED.load(Ordering::Relaxed))
}

pub fn set_speed(factor: f32) {
    SPEED.store(factor.max(0.0).to_bits(), Ordering::Relaxed);
}

pub fn key(base: impl std::fmt::Display, channel: &str) -> ElementId {
    ElementId::Name(format!("{base}::{channel}").into())
}

/// A springy scalar keyed by `id`.
pub fn value(id: ElementId, target: f32, feel: Feel, window: &mut Window, cx: &mut App) -> f32 {
    if speed() == 0.0 {
        return target;
    }
    spring(id, target, feel.spring(), window, cx)
}

/// A springy scalar that snaps while `travel` is false (for example while dragging).
pub fn follow(id: ElementId, target: f32, feel: Feel, travel: bool, window: &mut Window, cx: &mut App) -> f32 {
    if speed() == 0.0 {
        return target;
    }
    spring(id, target, feel.spring().with_travel(travel), window, cx)
}

/// 0 → 1 springing toward `on`.
pub fn toggle(id: ElementId, on: bool, feel: Feel, window: &mut Window, cx: &mut App) -> f32 {
    value(id, if on { 1.0 } else { 0.0 }, feel, window, cx)
}

/// Fades and lifts an element in when it first appears under `id`. Changing the
/// id (for example by including the selected tab) replays the entrance.
pub fn enter<E>(id: impl Into<ElementId>, el: E, lift: f32) -> impl gpui_kit::IntoElement
where
    E: gpui_kit::IntoElement + gpui_kit::Styled + 'static,
{
    use gpui_kit::{Animation, AnimationExt as _, ease_out_quint, px};
    let ms = (220.0 * speed()).max(1.0) as u64;
    el.with_animation(
        id,
        Animation::new(Duration::from_millis(ms)).with_easing(ease_out_quint()),
        move |el, t| el.opacity(t).mt(px(lift * (1.0 - t))),
    )
}

/// Like [`enter`], sliding in from the side by `shift` pixels (negative: from the left).
pub fn slide_in<E>(id: impl Into<ElementId>, el: E, shift: f32) -> impl gpui_kit::IntoElement
where
    E: gpui_kit::IntoElement + gpui_kit::Styled + 'static,
{
    use gpui_kit::{Animation, AnimationExt as _, ease_out_quint, px};
    let ms = (260.0 * speed()).max(1.0) as u64;
    el.with_animation(
        id,
        Animation::new(Duration::from_millis(ms)).with_easing(ease_out_quint()),
        move |el, t| el.opacity(t).ml(px(shift * (1.0 - t))),
    )
}

/// Staggered entrance for the `index`-th item of a list: each item starts a
/// little after the one before, capped so long lists do not crawl in.
pub fn stagger<E>(id: impl Into<ElementId>, index: usize, el: E, lift: f32) -> impl gpui_kit::IntoElement
where
    E: gpui_kit::IntoElement + gpui_kit::Styled + 'static,
{
    use gpui_kit::{Animation, AnimationExt as _, px};
    let delay = (index.min(12) as f32 * 0.035) * speed();
    let run = 0.24 * speed();
    let total = (delay + run).max(0.001);
    let start = delay / total;
    el.with_animation(id, Animation::new(Duration::from_secs_f32(total)), move |el, t| {
        let local = ((t - start) / (1.0 - start)).clamp(0.0, 1.0);
        // ease-out cubic
        let e = 1.0 - (1.0 - local).powi(3);
        el.opacity(e).mt(px(lift * (1.0 - e)))
    })
}

struct Morph {
    from: Vec<f32>,
    to: Vec<f32>,
    start: std::time::Instant,
}

impl Morph {
    fn at(&self, now: std::time::Instant, secs: f32) -> (Vec<f32>, bool) {
        let t = if secs <= 0.0 { 1.0 } else { (now.duration_since(self.start).as_secs_f32() / secs).min(1.0) };
        if t >= 1.0 || self.from.len() != self.to.len() {
            return (self.to.clone(), true);
        }
        // ease-out cubic: data settles quickly and never overshoots, so a
        // histogram never shows counts the image does not have.
        let e = 1.0 - (1.0 - t).powi(3);
        (self.from.iter().zip(&self.to).map(|(a, b)| a + (b - a) * e).collect(), false)
    }
}

/// Tweens a whole series (histogram bins, a curve's samples) toward `target`.
/// A new target starts from what is on screen, so a stream of preview frames
/// reads as one continuous change instead of a flicker.
pub fn morph(id: impl Into<ElementId>, target: Vec<f32>, window: &mut Window, cx: &mut App) -> Vec<f32> {
    morph_follow(id, target, true, window, cx)
}

/// Like [`morph`], but snaps while `travel` is false (the user is dragging the
/// thing being drawn, which must track the pointer exactly).
pub fn morph_follow(id: impl Into<ElementId>, target: Vec<f32>, travel: bool, window: &mut Window, cx: &mut App) -> Vec<f32> {
    let secs = if travel { 0.26 * speed() } else { 0.0 };
    let now = std::time::Instant::now();
    let state = window.use_keyed_state(id, cx, |_, _| Morph { from: target.clone(), to: target.clone(), start: now });
    let (shown, done) = state.update(cx, |m, _| {
        if m.to != target {
            let (current, _) = m.at(now, secs);
            *m = Morph { from: current, to: target, start: now };
        }
        m.at(now, secs)
    });
    if !done {
        window.request_animation_frame();
    }
    shown
}
