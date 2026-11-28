//! Reading trackpad scrolls: one gesture from fingers down to fingers up,
//! locked to the axis it started along, and the two-finger swipe that pages
//! through images.
//!
//! On macOS GPUI reports a trackpad scroll as `Started`, `Moved`…, `Ended`,
//! then the system's momentum as more `Moved` events with no `Started` before
//! them. A mouse wheel arrives as line deltas. Platforms that report no phases
//! for touchpads see every event as momentum here, so phase-driven gestures
//! (the image swipe) simply never fire there; plain panning is unaffected.

use gpui_kit::TouchPhase;

/// Travel, in pixels, before a gesture commits to an axis.
const LOCK_PX: f32 = 8.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Axis {
    X,
    Y,
}

/// What part of a gesture a scroll event was.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Part {
    Begin,
    Move,
    /// Fingers lifted.
    End,
    /// The system cancelled the gesture; undo anything it started.
    Cancel,
    /// Momentum after the fingers lifted, or a scroll with no phase.
    Coast,
}

/// One trackpad gesture.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Gesture {
    live: bool,
    sum: (f32, f32),
    axis: Option<Axis>,
}

impl Gesture {
    pub(crate) fn feed(&mut self, phase: TouchPhase, d: (f32, f32)) -> Part {
        match phase {
            TouchPhase::Started => {
                // macOS sends MayBegin and Began, both `Started`; the first
                // carries no travel, so restarting on the second is harmless.
                *self = Gesture { live: true, ..Gesture::default() };
                self.add(d);
                Part::Begin
            }
            TouchPhase::Moved if self.live => {
                self.add(d);
                Part::Move
            }
            TouchPhase::Moved => Part::Coast,
            TouchPhase::Ended if self.live => {
                self.live = false;
                Part::End
            }
            TouchPhase::Cancelled if self.live => {
                self.live = false;
                Part::Cancel
            }
            TouchPhase::Ended | TouchPhase::Cancelled => Part::Coast,
        }
    }

    fn add(&mut self, d: (f32, f32)) {
        self.sum.0 += d.0;
        self.sum.1 += d.1;
        if self.axis.is_none() && self.sum.0.abs().max(self.sum.1.abs()) >= LOCK_PX {
            self.axis = Some(if self.sum.0.abs() > self.sum.1.abs() { Axis::X } else { Axis::Y });
        }
    }

    pub(crate) fn live(&self) -> bool {
        self.live
    }

    pub(crate) fn axis(&self) -> Option<Axis> {
        self.axis
    }

    pub(crate) fn sum(&self) -> (f32, f32) {
        self.sum
    }
}

/// Whether moving from `prev` to `next` reached or passed `mark` from either side.
pub(crate) fn crossed(prev: f32, next: f32, mark: f32) -> bool {
    (prev < mark && next >= mark) || (prev > mark && next <= mark)
}

/// What a [`Swipe`] asks the viewer to do.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SwipeAction {
    None,
    /// The swipe passed the point where letting go changes the image.
    Armed,
    /// It came back short of that point.
    Disarmed,
    /// Change the image by this many places.
    Go(isize),
}

/// Two fingers sideways across a fitted image: past a quarter of the view
/// (or a quick flick) and released, the next or previous image comes up.
/// Momentum never pages, so one swipe moves one image.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Swipe {
    gesture: Gesture,
    armed: bool,
    /// Fastest single step along x, for telling a flick from a slow drag.
    peak: f32,
}

impl Swipe {
    /// Travel needed to change images in a view `width` pixels wide.
    pub(crate) fn threshold(width: f32) -> f32 {
        (width * 0.25).clamp(80.0, 240.0)
    }

    pub(crate) fn feed(&mut self, phase: TouchPhase, d: (f32, f32), width: f32) -> SwipeAction {
        match self.gesture.feed(phase, d) {
            Part::Begin => {
                self.armed = false;
                self.peak = 0.0;
                SwipeAction::None
            }
            Part::Move => {
                if self.gesture.axis() != Some(Axis::X) {
                    return SwipeAction::None;
                }
                self.peak = self.peak.max(d.0.abs());
                let armed = self.gesture.sum().0.abs() >= Self::threshold(width);
                if armed == self.armed {
                    return SwipeAction::None;
                }
                self.armed = armed;
                if armed { SwipeAction::Armed } else { SwipeAction::Disarmed }
            }
            Part::End => {
                let armed = std::mem::take(&mut self.armed);
                let x = self.gesture.sum().0;
                let flick = x.abs() >= 48.0 && self.peak >= 30.0;
                if self.gesture.axis() == Some(Axis::X) && (armed || flick) {
                    // Content follows the fingers: fingers left shows what is to the right.
                    SwipeAction::Go(if x < 0.0 { 1 } else { -1 })
                } else {
                    SwipeAction::None
                }
            }
            Part::Cancel => {
                self.armed = false;
                SwipeAction::None
            }
            Part::Coast => SwipeAction::None,
        }
    }

    /// Whether the last move left the swipe past its threshold.
    #[cfg(test)]
    pub(crate) fn armed(&self) -> bool {
        self.armed
    }

    /// How far the image should trail the fingers right now, in pixels.
    pub(crate) fn offset(&self) -> f32 {
        if self.gesture.live() && self.gesture.axis() == Some(Axis::X) { self.gesture.sum().0 } else { 0.0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use TouchPhase::*;

    fn run(swipe: &mut Swipe, events: &[(TouchPhase, f32, f32)]) -> Vec<SwipeAction> {
        events.iter().map(|&(p, x, y)| swipe.feed(p, (x, y), 800.0)).filter(|a| *a != SwipeAction::None).collect()
    }

    #[test]
    fn a_gesture_locks_to_the_axis_it_starts_along() {
        let mut g = Gesture::default();
        g.feed(Started, (0.0, 0.0));
        g.feed(Moved, (3.0, 1.0));
        assert_eq!(g.axis(), None, "not committed before {LOCK_PX}px");
        g.feed(Moved, (6.0, 2.0));
        assert_eq!(g.axis(), Some(Axis::X));
        g.feed(Moved, (0.0, 40.0));
        assert_eq!(g.axis(), Some(Axis::X), "the lock holds once taken");
    }

    #[test]
    fn momentum_and_phaseless_scrolls_are_coasting() {
        let mut g = Gesture::default();
        assert_eq!(g.feed(Moved, (5.0, 0.0)), Part::Coast);
        g.feed(Started, (0.0, 0.0));
        assert_eq!(g.feed(Moved, (5.0, 0.0)), Part::Move);
        assert_eq!(g.feed(Ended, (0.0, 0.0)), Part::End);
        assert_eq!(g.feed(Moved, (9.0, 0.0)), Part::Coast);
        assert_eq!(g.feed(Ended, (0.0, 0.0)), Part::Coast, "a stray end after momentum is not a second end");
    }

    #[test]
    fn a_long_swipe_left_goes_to_the_next_image_once() {
        let mut s = Swipe::default();
        let mut events = vec![(Started, 0.0, 0.0)];
        events.extend(std::iter::repeat_n((Moved, -12.0, 0.0), 20));
        events.push((Ended, 0.0, 0.0));
        events.extend(std::iter::repeat_n((Moved, -20.0, 0.0), 10));
        assert_eq!(run(&mut s, &events), vec![SwipeAction::Armed, SwipeAction::Go(1)]);
    }

    #[test]
    fn a_swipe_right_goes_back() {
        let mut s = Swipe::default();
        let mut events = vec![(Started, 0.0, 0.0)];
        events.extend(std::iter::repeat_n((Moved, 15.0, 0.0), 16));
        events.push((Ended, 0.0, 0.0));
        assert_eq!(run(&mut s, &events), vec![SwipeAction::Armed, SwipeAction::Go(-1)]);
    }

    #[test]
    fn coming_back_short_of_the_threshold_stays_put() {
        let mut s = Swipe::default();
        let mut events = vec![(Started, 0.0, 0.0)];
        events.extend(std::iter::repeat_n((Moved, -8.0, 0.0), 30));
        events.extend(std::iter::repeat_n((Moved, 4.0, 0.0), 30));
        events.push((Ended, 0.0, 0.0));
        assert_eq!(run(&mut s, &events), vec![SwipeAction::Armed, SwipeAction::Disarmed]);
        assert_eq!(s.offset(), 0.0, "no offset once the fingers lift");
    }

    #[test]
    fn a_quick_flick_pages_without_reaching_the_threshold() {
        let mut s = Swipe::default();
        let events = [(Started, 0.0, 0.0), (Moved, -10.0, 0.0), (Moved, -35.0, 0.0), (Moved, -15.0, 0.0), (Ended, 0.0, 0.0)];
        assert_eq!(run(&mut s, &events), vec![SwipeAction::Go(1)]);
    }

    #[test]
    fn vertical_and_cancelled_gestures_never_page() {
        let mut s = Swipe::default();
        let mut events = vec![(Started, 0.0, 0.0)];
        events.extend(std::iter::repeat_n((Moved, 2.0, 30.0), 20));
        events.push((Ended, 0.0, 0.0));
        assert!(run(&mut s, &events).is_empty());

        let mut events = vec![(Started, 0.0, 0.0)];
        events.extend(std::iter::repeat_n((Moved, -20.0, 0.0), 20));
        events.push((Cancelled, 0.0, 0.0));
        assert_eq!(run(&mut s, &events), vec![SwipeAction::Armed]);
        assert!(!s.armed());
    }

    #[test]
    fn crossing_a_mark_counts_from_either_side() {
        assert!(crossed(0.9, 1.1, 1.0));
        assert!(crossed(1.1, 0.9, 1.0));
        assert!(crossed(0.9, 1.0, 1.0), "landing on the mark counts");
        assert!(!crossed(1.0, 1.2, 1.0), "leaving the mark does not");
        assert!(!crossed(0.5, 0.8, 1.0));
    }

    #[test]
    fn the_swipe_threshold_scales_with_the_view_within_limits() {
        assert_eq!(Swipe::threshold(200.0), 80.0);
        assert_eq!(Swipe::threshold(800.0), 200.0);
        assert_eq!(Swipe::threshold(3000.0), 240.0);
    }
}
