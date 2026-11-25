//! Byte-budgeted LRU shared by the ONNX session registry and the AI mask
//! cache.
//!
//! Both caches used to drop *everything* once an entry-count threshold was
//! crossed, including entries a caller was still using. This container charges
//! each entry its estimated resident bytes instead, evicts in least-recently-
//! used order, and never releases an entry that another clone still owns: such
//! an entry is marked pending and reclaimed on a later pass, once the last
//! outside reference is gone.
//!
//! Pure `std`: no ONNX Runtime and no GPU, so every unit test here runs in the
//! default CI job.

use std::collections::HashMap;

/// `AURORA_*` budget in MiB. `0` disables retention: nothing is kept except
/// the single entry just inserted, because the cache never drops the value a
/// caller is in the middle of using. A missing or unparseable value falls back
/// to `default_mib`.
pub(crate) fn budget_from_env(name: &str, default_mib: u64) -> u64 {
    budget_from_value(name, std::env::var(name).ok().as_deref(), default_mib)
}

fn budget_from_value(name: &str, raw: Option<&str>, default_mib: u64) -> u64 {
    let default_bytes = default_mib.saturating_mul(1024 * 1024);
    let Some(raw) = raw.map(str::trim).filter(|raw| !raw.is_empty()) else {
        return default_bytes;
    };
    match raw.parse::<u64>() {
        Ok(mib) => mib.saturating_mul(1024 * 1024),
        Err(_) => {
            log::warn!("{name}={raw:?} is not a MiB byte count; using {default_mib} MiB");
            default_bytes
        }
    }
}

struct Slot<V> {
    value: V,
    bytes: u64,
    last_used: u64,
    /// Wanted out of the cache but still referenced elsewhere.
    pending: bool,
}

/// What one [`ByteLru::evict`] pass did.
pub(crate) struct Eviction<V> {
    /// Entries removed from the map. The caller drops them *after* releasing
    /// its lock, so a release that calls into a driver never runs under it.
    pub(crate) removed: Vec<(String, V)>,
    /// Entries that had to go but are still referenced elsewhere; they stay
    /// accounted against the budget until a later pass can release them.
    pub(crate) pending: usize,
}

impl<V> Default for Eviction<V> {
    fn default() -> Self {
        Self {
            removed: Vec::new(),
            pending: 0,
        }
    }
}

pub(crate) struct ByteLru<V> {
    slots: HashMap<String, Slot<V>>,
    bytes: u64,
    budget: u64,
    max_entries: usize,
    tick: u64,
}

impl<V> ByteLru<V> {
    pub(crate) fn new(budget: u64, max_entries: usize) -> Self {
        Self {
            slots: HashMap::new(),
            bytes: 0,
            budget,
            max_entries: max_entries.max(1),
            tick: 0,
        }
    }

    /// Entry count. Only the tests need it today.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.slots.len()
    }

    /// Bytes currently charged, including entries waiting for a pending
    /// eviction. They still occupy memory, so the budget must keep counting
    /// them until they are really released.
    #[cfg(test)]
    pub(crate) fn bytes(&self) -> u64 {
        self.bytes
    }

    /// Records a use and returns the value. The caller clones it, and that
    /// clone keeps the entry protected from eviction while it lives.
    pub(crate) fn get(&mut self, key: &str) -> Option<&V> {
        self.tick = self.tick.wrapping_add(1);
        let tick = self.tick;
        let slot = self.slots.get_mut(key)?;
        slot.last_used = tick;
        Some(&slot.value)
    }

    /// Inserts or replaces an entry. Eviction is a separate pass so the caller
    /// can drop removed values outside its own lock. Replacing an existing key
    /// returns the previous value, which the caller drops after unlocking.
    pub(crate) fn insert(&mut self, key: String, value: V, bytes: u64) -> Option<V> {
        self.tick = self.tick.wrapping_add(1);
        let old = self.slots.get(&key).map(|slot| slot.bytes);
        if let Some(bytes) = old {
            self.bytes = self.bytes.saturating_sub(bytes);
        }
        let previous = self.slots.remove(&key).map(|slot| slot.value);
        let tick = self.tick;
        self.slots.insert(
            key,
            Slot {
                value,
                bytes,
                last_used: tick,
                pending: false,
            },
        );
        self.bytes = self.bytes.saturating_add(bytes);
        previous
    }

    fn over_budget(&self) -> bool {
        self.bytes > self.budget || self.slots.len() > self.max_entries
    }

    /// Drops least-recently-used entries until the budget fits. An entry that
    /// `in_use` reports as referenced elsewhere is skipped; when the excess
    /// cannot be removed now, the LRU entries needed to cover it are marked
    /// pending and reclaimed by a later pass as soon as their clones drop.
    ///
    /// A lone entry is never evicted: a model larger than the whole budget
    /// must still be usable, even if that means the cache holds exactly it.
    pub(crate) fn evict(&mut self, mut in_use: impl FnMut(&str, &V) -> bool) -> Eviction<V> {
        let mut eviction = Eviction::default();
        while self.slots.len() > 1 && self.over_budget() {
            let Some(key) = self.victim(&mut in_use) else {
                break;
            };
            if let Some(slot) = self.slots.remove(&key) {
                self.bytes = self.bytes.saturating_sub(slot.bytes);
                eviction.removed.push((key, slot.value));
            }
        }
        if self.slots.len() > 1 && self.over_budget() {
            let mut wanted_bytes = self.bytes.saturating_sub(self.budget);
            let mut wanted_entries = self.slots.len().saturating_sub(self.max_entries);
            let mut order: Vec<(u64, String)> = self
                .slots
                .iter()
                .map(|(key, slot)| (slot.last_used, key.clone()))
                .collect();
            order.sort_unstable_by_key(|(last_used, _)| *last_used);
            for (_, key) in order {
                if wanted_bytes == 0 && wanted_entries == 0 {
                    break;
                }
                let Some(slot) = self.slots.get_mut(&key) else {
                    continue;
                };
                slot.pending = true;
                wanted_bytes = wanted_bytes.saturating_sub(slot.bytes);
                wanted_entries = wanted_entries.saturating_sub(1);
                eviction.pending += 1;
            }
        }
        eviction
    }

    /// Least-recently-used evictable key, preferring entries already marked
    /// pending.
    fn victim(&self, in_use: &mut impl FnMut(&str, &V) -> bool) -> Option<String> {
        let mut best: Option<(bool, u64, &String)> = None;
        for (key, slot) in &self.slots {
            if in_use(key, &slot.value) {
                continue;
            }
            let better = match &best {
                None => true,
                Some((pending, last_used, _)) => {
                    (slot.pending, std::cmp::Reverse(slot.last_used))
                        > (*pending, std::cmp::Reverse(*last_used))
                }
            };
            if better {
                best = Some((slot.pending, slot.last_used, key));
            }
        }
        best.map(|(_, _, key)| key.clone())
    }

    /// Number of entries currently marked pending, for tests and callers that
    /// want to know whether memory is waiting on live references.
    #[cfg(test)]
    fn pending_count(&self) -> usize {
        self.slots.values().filter(|slot| slot.pending).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    const MIB: u64 = 1024 * 1024;

    fn free(_key: &str, _value: &Arc<u32>) -> bool {
        false
    }

    fn shared(_key: &str, value: &Arc<u32>) -> bool {
        Arc::strong_count(value) > 1
    }

    #[test]
    fn budget_parsing_accepts_mib_and_falls_back_on_garbage() {
        assert_eq!(budget_from_value("TEST", None, 512), 512 * MIB);
        assert_eq!(budget_from_value("TEST", Some("8"), 512), 8 * MIB);
        assert_eq!(budget_from_value("TEST", Some(" 0 "), 512), 0);
        assert_eq!(budget_from_value("TEST", Some(""), 512), 512 * MIB);
        assert_eq!(budget_from_value("TEST", Some("1.5"), 512), 512 * MIB);
        assert_eq!(
            budget_from_value("TEST", Some("99999999999999999999"), 512),
            512 * MIB,
            "an out-of-range value falls back, it does not wrap"
        );
    }

    #[test]
    fn evicts_the_least_recently_used_entry_over_the_byte_budget() {
        let mut lru = ByteLru::new(100, 16);
        let _ = lru.insert("a".to_string(), Arc::new(1), 60);
        let _ = lru.insert("b".to_string(), Arc::new(2), 60);
        assert_eq!(lru.bytes(), 120);

        let eviction = lru.evict(free);
        assert_eq!(eviction.removed.len(), 1);
        assert_eq!(eviction.removed[0].0, "a");
        assert_eq!(eviction.pending, 0);
        assert_eq!(lru.len(), 1);
        assert_eq!(lru.bytes(), 60);
    }

    #[test]
    fn a_lookup_makes_an_entry_too_recent_to_evict() {
        let mut lru = ByteLru::new(100, 16);
        let _ = lru.insert("a".to_string(), Arc::new(1), 60);
        let _ = lru.insert("b".to_string(), Arc::new(2), 60);

        assert!(lru.get("a").is_some());
        let eviction = lru.evict(free);
        assert_eq!(eviction.removed.len(), 1);
        assert_eq!(eviction.removed[0].0, "b", "the untouched entry is the victim");
    }

    #[test]
    fn an_entry_still_owned_outside_the_cache_is_deferred_not_dropped() {
        let mut lru = ByteLru::new(100, 16);
        let a = Arc::new(1);
        let b = Arc::new(2);
        let _ = lru.insert("a".to_string(), a.clone(), 60);
        let _ = lru.insert("b".to_string(), b.clone(), 60);

        let eviction = lru.evict(shared);
        assert!(eviction.removed.is_empty(), "nothing may be released while clones live");
        assert_eq!(eviction.pending, 1, "the LRU entry is marked for eviction");
        assert_eq!(lru.pending_count(), 1);
        assert_eq!(lru.len(), 2);
        assert_eq!(lru.bytes(), 120, "a pending entry still occupies memory");

        // The caller drops its clone; the next pass reclaims the entry and the
        // budget with it.
        drop(a);
        let eviction = lru.evict(shared);
        assert_eq!(eviction.removed.len(), 1);
        assert_eq!(eviction.removed[0].0, "a");
        assert_eq!(lru.len(), 1);
        assert_eq!(lru.bytes(), 60);
    }

    #[test]
    fn pending_entries_shrink_as_each_owner_releases_them() {
        let mut lru = ByteLru::new(50, 16);
        let a = Arc::new(1);
        let b = Arc::new(2);
        let _ = lru.insert("a".to_string(), a.clone(), 60);
        let _ = lru.insert("b".to_string(), b.clone(), 60);

        let eviction = lru.evict(shared);
        assert_eq!(eviction.pending, 2, "both entries are needed to fit 50 bytes");
        assert_eq!(lru.bytes(), 120);

        drop(b);
        let eviction = lru.evict(shared);
        assert_eq!(eviction.removed.len(), 1, "the pending entry is freed first");
        assert_eq!(eviction.removed[0].0, "b");
        assert_eq!(lru.bytes(), 60);
    }

    #[test]
    fn one_oversized_entry_is_kept_and_counted() {
        let mut lru = ByteLru::new(10, 16);
        let _ = lru.insert("only".to_string(), Arc::new(1), 60);
        let eviction = lru.evict(free);
        assert!(eviction.removed.is_empty());
        assert_eq!(eviction.pending, 0);
        assert_eq!(lru.len(), 1);
        assert_eq!(lru.bytes(), 60);
    }

    #[test]
    fn the_entry_cap_still_evicts_under_a_byte_budget() {
        let mut lru = ByteLru::new(1 << 40, 2);
        let _ = lru.insert("a".to_string(), Arc::new(1), 1);
        let _ = lru.insert("b".to_string(), Arc::new(2), 1);
        let _ = lru.insert("c".to_string(), Arc::new(3), 1);
        let eviction = lru.evict(free);
        assert_eq!(eviction.removed.len(), 1);
        assert_eq!(eviction.removed[0].0, "a");
        assert_eq!(lru.len(), 2);
    }

    #[test]
    fn replacing_a_key_updates_its_charge() {
        let mut lru = ByteLru::new(1 << 20, 16);
        let _ = lru.insert("a".to_string(), Arc::new(1), 60);
        let old = lru.insert("a".to_string(), Arc::new(2), 10);
        assert!(old.is_some(), "the replaced value is returned to the caller");
        assert_eq!(lru.len(), 1);
        assert_eq!(lru.bytes(), 10);
    }
}
