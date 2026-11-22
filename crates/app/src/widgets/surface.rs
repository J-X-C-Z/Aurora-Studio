//! An interactive area that reports pointer positions normalized to its bounds,
//! for widgets that paint their own content (curves, wheels, levels, the viewer).

use std::rc::Rc;

use gpui_kit::base::ElementExt as _;
use gpui_kit::{
    AnyElement, App, AppContext as _, Bounds, Context, Div, Entity, InteractiveElement, IntoElement, Modifiers,
    MouseButton, MouseDownEvent, ParentElement, Pixels, Point, Render, RenderOnce, SharedString, Stateful,
    StatefulInteractiveElement, StyleRefinement, Styled, Window, div,
};

use crate::motion;

#[derive(Clone, Copy, Debug)]
pub struct Pointer {
    /// Position normalized to the bounds; may fall outside 0..1 while dragging.
    pub pos: (f32, f32),
    pub point: Point<Pixels>,
    pub bounds: Bounds<Pixels>,
    pub modifiers: Modifiers,
    pub clicks: usize,
}

impl Pointer {
    fn new(point: Point<Pixels>, bounds: Bounds<Pixels>, modifiers: Modifiers, clicks: usize) -> Self {
        let w = f32::from(bounds.size.width).max(1.0);
        let h = f32::from(bounds.size.height).max(1.0);
        Self {
            pos: (f32::from(point.x - bounds.origin.x) / w, f32::from(point.y - bounds.origin.y) / h),
            point,
            bounds,
            modifiers,
            clicks,
        }
    }
}

type Handler = Rc<dyn Fn(Pointer, &mut Window, &mut App)>;
type UpHandler = Rc<dyn Fn(&mut Window, &mut App)>;

#[derive(Default)]
pub struct SurfaceState {
    pub bounds: Bounds<Pixels>,
    pub dragging: bool,
    pub hover: Option<(f32, f32)>,
}

#[derive(Clone)]
struct SurfaceDrag(SharedString);

struct Ghost;

impl Render for Ghost {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div()
    }
}

#[derive(IntoElement)]
pub struct Surface {
    id: SharedString,
    base: Div,
    children: Vec<AnyElement>,
    on_down: Option<Handler>,
    on_move: Option<Handler>,
    on_hover_move: Option<Handler>,
    on_up: Option<UpHandler>,
    button: MouseButton,
}

impl Surface {
    pub fn new(id: impl Into<SharedString>) -> Self {
        Self {
            id: id.into(),
            base: div(),
            children: Vec::new(),
            on_down: None,
            on_move: None,
            on_hover_move: None,
            on_up: None,
            button: MouseButton::Left,
        }
    }

    /// Retained state for this surface, so painters can read the hover position.
    pub fn state(id: &SharedString, window: &mut Window, cx: &mut App) -> Entity<SurfaceState> {
        window.use_keyed_state(motion::key(id, "surface"), cx, |_, _| SurfaceState::default())
    }

    pub fn on_down(mut self, f: impl Fn(Pointer, &mut Window, &mut App) + 'static) -> Self {
        self.on_down = Some(Rc::new(f));
        self
    }

    pub fn on_drag(mut self, f: impl Fn(Pointer, &mut Window, &mut App) + 'static) -> Self {
        self.on_move = Some(Rc::new(f));
        self
    }

    pub fn on_hover_move(mut self, f: impl Fn(Pointer, &mut Window, &mut App) + 'static) -> Self {
        self.on_hover_move = Some(Rc::new(f));
        self
    }

    pub fn on_up(mut self, f: impl Fn(&mut Window, &mut App) + 'static) -> Self {
        self.on_up = Some(Rc::new(f));
        self
    }
}

impl Styled for Surface {
    fn style(&mut self) -> &mut StyleRefinement {
        self.base.style()
    }
}

impl InteractiveElement for Surface {
    fn interactivity(&mut self) -> &mut gpui_kit::Interactivity {
        self.base.interactivity()
    }
}

impl ParentElement for Surface {
    fn extend(&mut self, elements: impl IntoIterator<Item = AnyElement>) {
        self.children.extend(elements);
    }
}

impl RenderOnce for Surface {
    fn render(self, window: &mut Window, cx: &mut App) -> impl IntoElement {
        let st = Self::state(&self.id, window, cx);
        let id = self.id.clone();
        let button = self.button;
        let el: Stateful<Div> = self.base.id(id.clone());
        el.relative()
            .children(self.children)
            .on_prepaint({
                let st = st.clone();
                move |bounds, _, cx| {
                    if st.read(cx).bounds != bounds {
                        st.update(cx, |s, _| s.bounds = bounds);
                    }
                }
            })
            .on_mouse_down(button, {
                let st = st.clone();
                let on_down = self.on_down.clone();
                move |e: &MouseDownEvent, window, cx| {
                    cx.stop_propagation();
                    let bounds = st.read(cx).bounds;
                    st.update(cx, |s, _| s.dragging = true);
                    if let Some(f) = &on_down {
                        f(Pointer::new(e.position, bounds, e.modifiers, e.click_count), window, cx);
                    }
                }
            })
            .on_mouse_move({
                let st = st.clone();
                let on_hover = self.on_hover_move.clone();
                move |e, window, cx| {
                    let bounds = st.read(cx).bounds;
                    let p = Pointer::new(e.position, bounds, e.modifiers, 0);
                    st.update(cx, |s, cx| {
                        s.hover = Some(p.pos);
                        cx.notify();
                    });
                    if let Some(f) = &on_hover {
                        f(p, window, cx);
                    }
                }
            })
            .on_hover({
                let st = st.clone();
                move |h, _, cx| {
                    if !*h {
                        st.update(cx, |s, cx| {
                            s.hover = None;
                            cx.notify();
                        });
                    }
                }
            })
            .on_drag(SurfaceDrag(id.clone()), |_, _, _, cx| cx.new(|_| Ghost))
            .on_drag_move::<SurfaceDrag>({
                let st = st.clone();
                let on_move = self.on_move.clone();
                move |e, window, cx| {
                    if e.drag(cx).0 != id || !st.read(cx).dragging {
                        return;
                    }
                    let bounds = st.read(cx).bounds;
                    if let Some(f) = &on_move {
                        f(Pointer::new(e.event.position, bounds, e.event.modifiers, 0), window, cx);
                    }
                }
            })
            .on_mouse_up(button, {
                let st = st.clone();
                let on_up = self.on_up.clone();
                move |_, window, cx| {
                    if st.read(cx).dragging {
                        st.update(cx, |s, _| s.dragging = false);
                        if let Some(f) = &on_up {
                            f(window, cx);
                        }
                    }
                }
            })
            .on_mouse_up_out(button, {
                let st = st.clone();
                let on_up = self.on_up.clone();
                move |_, window, cx| {
                    if st.read(cx).dragging {
                        st.update(cx, |s, _| s.dragging = false);
                        if let Some(f) = &on_up {
                            f(window, cx);
                        }
                    }
                }
            })
    }
}
