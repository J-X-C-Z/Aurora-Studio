//! Text fields for stateless tool panels. The input state lives in keyed element
//! state, so a panel function can show a field without owning an entity; change
//! the key (for example by including the image id) to start from a new value.

use std::rc::Rc;

use gpui_kit::component::{
    Sizable as _,
    input::{Input, InputEvent, InputState},
};
use gpui_kit::{AnyElement, App, AppContext as _, ElementId, Entity, IntoElement, Subscription, Window};

pub struct Field {
    pub input: Entity<InputState>,
    _sub: Subscription,
}

type OnText = Rc<dyn Fn(String, bool, &mut App)>;

/// Returns the retained input for `key`. `on_text` receives the text on every
/// change, with `true` when Enter was pressed.
pub fn field_state(
    key: impl Into<ElementId>,
    initial: impl Into<String>,
    placeholder: &'static str,
    on_text: impl Fn(String, bool, &mut App) + 'static,
    window: &mut Window,
    cx: &mut App,
) -> Entity<InputState> {
    let initial: String = initial.into();
    let on_text: OnText = Rc::new(on_text);
    let holder = window.use_keyed_state(key, cx, move |window, cx| {
        let input = cx.new(|cx| InputState::new(window, cx).placeholder(placeholder).default_value(initial));
        let sub = cx.subscribe(&input, move |_, input, e: &InputEvent, cx| {
            let text = input.read(cx).value().to_string();
            match e {
                InputEvent::Change => on_text(text, false, cx),
                InputEvent::PressEnter { .. } => on_text(text, true, cx),
                _ => {}
            }
        });
        Field { input, _sub: sub }
    });
    holder.read(cx).input.clone()
}

/// A small single-line field.
pub fn text_field(
    key: impl Into<ElementId>,
    initial: impl Into<String>,
    placeholder: &'static str,
    on_text: impl Fn(String, bool, &mut App) + 'static,
    window: &mut Window,
    cx: &mut App,
) -> AnyElement {
    let input = field_state(key, initial, placeholder, on_text, window, cx);
    Input::new(&input).xsmall().into_any_element()
}
