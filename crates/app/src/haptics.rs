//! Trackpad haptic feedback.
//!
//! Only a macOS Force Touch trackpad can play these, and only while a finger
//! rests on it; the user turns them off in System Settings › Trackpad. On other
//! platforms, or with a mouse, [`play`] does nothing.

/// The three feedback patterns AppKit offers, by what they mean here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Haptic {
    /// Something lined up: a slider reaching its default, a snap.
    Alignment,
    /// A step between levels, such as the fit and 100% zoom stops.
    LevelChange,
    /// Anything else, such as panning into the image edge.
    Generic,
}

/// Plays `pattern` once, unless the same pattern played moments ago: a
/// gesture that wobbles across a mark should not buzz continuously.
pub(crate) fn play(pattern: Haptic) {
    use std::cell::Cell;
    use std::time::{Duration, Instant};
    thread_local! {
        static LAST: Cell<Option<(Haptic, Instant)>> = const { Cell::new(None) };
    }
    let now = Instant::now();
    let repeat = LAST.with(|l| l.get().is_some_and(|(p, at)| p == pattern && now.duration_since(at) < Duration::from_millis(80)));
    if repeat {
        return;
    }
    LAST.with(|l| l.set(Some((pattern, now))));
    imp::play(pattern);
}

#[cfg(target_os = "macos")]
mod imp {
    //! `[[NSHapticFeedbackManager defaultPerformer] performFeedbackPattern:performanceTime:]`
    //! through the Objective-C runtime, so no binding crate is needed. GPUI
    //! calls event handlers on the main thread, which AppKit requires here.

    use std::ffi::{c_char, c_void};

    use super::Haptic;

    #[link(name = "AppKit", kind = "framework")]
    unsafe extern "C" {}

    #[link(name = "objc")]
    unsafe extern "C" {
        fn objc_getClass(name: *const c_char) -> *mut c_void;
        fn sel_registerName(name: *const c_char) -> *mut c_void;
        fn objc_msgSend();
    }

    /// `NSHapticFeedbackPerformanceTimeDrawCompleted`: play with the next
    /// frame, so the bump lines up with what the view shows.
    const DRAW_COMPLETED: usize = 2;

    pub(super) fn play(pattern: Haptic) {
        // NSHapticFeedbackPattern{Generic, Alignment, LevelChange}.
        let pattern: isize = match pattern {
            Haptic::Generic => 0,
            Haptic::Alignment => 1,
            Haptic::LevelChange => 2,
        };
        // SAFETY: both selectors exist on macOS 10.11+, and objc_msgSend is
        // called through function types matching their signatures.
        unsafe {
            let class = objc_getClass(c"NSHapticFeedbackManager".as_ptr());
            if class.is_null() {
                return;
            }
            let send: unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void =
                std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
            let performer = send(class, sel_registerName(c"defaultPerformer".as_ptr()));
            if performer.is_null() {
                return;
            }
            let perform: unsafe extern "C" fn(*mut c_void, *mut c_void, isize, usize) =
                std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
            perform(performer, sel_registerName(c"performFeedbackPattern:performanceTime:".as_ptr()), pattern, DRAW_COMPLETED);
        }
    }
}

#[cfg(not(target_os = "macos"))]
mod imp {
    pub(super) fn play(_: super::Haptic) {}
}
