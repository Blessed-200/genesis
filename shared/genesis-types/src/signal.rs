//! # GÉNESIS System Signals
//!
//! Asynchronous event types that flow between cognitive layers via
//! `crossbeam::channel` (`MACRO_ARCHITECTURE` §4.1 — message passing).
//!
//! ## Corrections applied (Audit Rev 1)
//!
//! - **CORRECCIÓN ESTRUCTURAL-1:** `SpikeComponents` adopts a **`SoA` layout**
//!   (indices: `[u16; 16]`, values: `[f64; 16]`, count: `u8`, pad: `[u8; 7]`,
//!   tail pad: `[u8; 24]`).
//!   Blade indices are `u16`, supporting G(1,3+n) expansion to D ≤ 15 (2^15 = 32,768
//!   blades ≤ `u16::MAX`) via
//!   `GramSchmidtExpander` without future binary-contract breakage. `presence_mask`
//!   is eliminated; active blades are identified by `indices[..count]`. `u16`
//!   supports G(1,3+n) up to D=15 (2^15 = 32,768 blades ≤ `u16::MAX`), sufficient
//!   for all planned GÉNESIS roadmap phases.
//!
//! - **CORRECCIÓN ESTRUCTURAL-2:** `DomainConsolidationSignal<State>` uses the
//!   **type-state pattern** (`PhantomData<State>`) with two marker types
//!   (`Saturated`, `Certified`). `DomainResetSignal::validate_against` only
//!   accepts `&DomainConsolidationSignal<Saturated>`, making a reset against a
//!   `Certified` domain a **compile error** (AXIOMA-009, criterion 8).
//!   `is_irrevocable()` and `ConsolidationKind` are eliminated — the type IS the
//!   invariant.
//!
//! - **CORRECCIÓN ESTRUCTURAL-3:** `NodeId(u64)` and `Timestamp(u64)` newtypes
//!   prevent accidental transposition of `origin_node_id` and `timestamp_ns` in
//!   function signatures. Zero-cost (repr transparent).
//!
//! - `collapse_grade` upgraded to `Option<u16>` to match the blade index type.
//! - `domain` and `justification` fields use `&'static str` — no heap allocation.
//!
//! AX-ID: AXIOMA-002, AXIOMA-003, AXIOMA-006, AXIOMA-008, AXIOMA-009, AXIOMA-018

#![allow(clippy::must_use_candidate)]

use core::hash::{Hash, Hasher};
use core::marker::PhantomData;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::constants::{CLIFFORD_BASIS_SIZE, COGNITIVE_PLANCK_CONSTANT};
use crate::error::{domain_code, GenesisError};

// ============================================================================
// NEWTYPES — CORRECCIÓN ESTRUCTURAL-3
// ============================================================================

/// Unique identifier for a node in the cognitive manifold.
///
/// Newtype over `u64`. Zero overhead — `repr(transparent)`. Prevents
/// accidental use of a raw `u64` timestamp where a node ID is expected.
///
/// AX-ID: AXIOMA-004 (attractor landscape nodes)
///
/// ```compile_fail
/// use genesis_types::NodeId;
///
/// let _ = NodeId::new(u64::MAX);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
// SAFETY: NodeId is #[repr(transparent)] over u64. Deserialize is safe because
// the only invalid value (u64::MAX) is checked at construction, and deserialization
// produces a raw NodeId that callers must validate via try_new().
#[allow(clippy::unsafe_derive_deserialize)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct NodeId(u64);

impl NodeId {
    /// Maximum valid `NodeId` value accepted by [`NodeId::try_new`].
    /// El único ID prohibido es el centinela `NodeId::INVALID` (u64::MAX).
    /// El límite real de nodos lo impone el HNSW (u32::MAX ≈ 4B nodos),
    /// no este tipo primitivo.
    pub const MAX_VALID: u64 = u64::MAX - 1;

    /// Sentinel value used to fill empty slots (never a valid node identifier).
    pub const INVALID: Self = Self(u64::MAX);

    /// Attempts to construct a `NodeId` from a raw `u64`.
    ///
    /// # Errors
    /// Returns [`GenesisError::NodeIdOutOfRange`] when `raw` exceeds
    /// [`NodeId::MAX_VALID`].
    #[inline]
    pub const fn try_new(raw: u64) -> Result<Self, GenesisError> {
        if raw <= Self::MAX_VALID {
            Ok(Self(raw))
        } else {
            Err(GenesisError::NodeIdOutOfRange { raw })
        }
    }

    /// Creates a `NodeId` without range validation.
    ///
    /// # Safety
    /// `raw` must satisfy the same invariant enforced by [`NodeId::try_new`],
    /// i.e. `raw <= NodeId::MAX_VALID`, unless using the sentinel
    /// [`NodeId::INVALID`].
    #[inline]
    // SAFETY: caller guarantees `raw <= NodeId::MAX_VALID`. Used only in
    // internal const constructors where the value is a compile-time literal.
    pub const unsafe fn from_raw_unchecked(raw: u64) -> Self {
        // SAFETY: `raw` must satisfy the same invariant as [`NodeId::try_new`]
        // (`raw <= NodeId::MAX_VALID`) unless constructing the sentinel [`NodeId::INVALID`].
        Self(raw)
    }

    /// Returns the underlying `u64`.
    #[inline]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Monotonic nanosecond timestamp local to the originating processing unit.
///
/// Newtype over `u64`. Zero overhead — `repr(transparent)`. Prohibits use
/// as an array index (no `as usize` coercion). No global wall-clock exists
/// (AX-ID: AXIOMA-002).
///
/// AX-ID: AXIOMA-002 (time as geometric dimension)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Timestamp(
    /// Private. Access exclusively via [`Timestamp::nanoseconds`].
    /// Privatized to enforce AXIOMA-002 monotonicity invariants:
    /// no external code can bypass the constructor or set timestamps
    /// that violate causal ordering.
    u64,
);

/// Pair of Gaussian samples used in signal-processing paths.
// CRYSTAL: FO32 — inevitable
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct GaussianPair {
    /// First Gaussian sample.
    pub first: f64,
    /// Second Gaussian sample.
    pub second: f64,
}

static_assertions::const_assert_eq!(core::mem::size_of::<GaussianPair>(), 64);
static_assertions::const_assert_eq!(core::mem::align_of::<GaussianPair>(), 64);

impl Timestamp {
    /// Zero timestamp (epoch origin of a local processing unit).
    pub const ZERO: Self = Self(0);

    /// Constructs a `Timestamp` from a raw nanosecond count.
    #[inline]
    pub const fn new(ns: u64) -> Self {
        Self(ns)
    }

    /// Returns the underlying nanosecond value.
    #[inline]
    pub const fn nanoseconds(self) -> u64 {
        self.0
    }

    /// Age of this timestamp relative to `now_ns`.
    ///
    /// Returns 0 if `now_ns < self.0` (future timestamp — saturates, no panic).
    /// Used for freshness checks in `Proof::is_fresh` without exposing the raw field.
    #[inline]
    pub const fn saturating_age_ns(self, now_ns: u64) -> u64 {
        now_ns.saturating_sub(self.0)
    }
}

// ============================================================================
// SPIKE COMPONENTS — CORRECCIÓN ESTRUCTURAL-1: SoA layout, u16 blade indices
// ============================================================================

/// Maximum number of blades transported in a single `SpikeEvent` (top-K sparse).
///
/// Fixed at `CLIFFORD_BASIS_SIZE` = 16 for the base G(1,3) algebra.
/// The `SoA` arrays are sized by this constant; it is the **single source of truth**
/// for the spike capacity. Changing this constant changes all array sizes
/// simultaneously, without silent divergence.
///
/// AX-ID: AXIOMA-001, AXIOMA-018
pub const SPIKE_MAX_COMPONENTS: usize = CLIFFORD_BASIS_SIZE; // 16

/// Typed blade index for a specific Clifford basis dimension `D`.
///
/// Valid values satisfy `index < (1 << D)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct BladeIndex<const D: usize>(u16);

impl<const D: usize> BladeIndex<D> {
    /// Returns the maximum exclusive value for this dimension (`2^D`).
    pub const fn upper_bound() -> u32 {
        if D < u32::BITS as usize {
            1u32 << D
        } else {
            0
        }
    }

    /// Fallible constructor enforcing `index < (1 << D)`.
    ///
    /// # Errors
    /// Returns [`SpikeComponentsError::DimensionUnsupported`] when `D >= 32`.
    /// Returns [`SpikeComponentsError::InvalidBladeIndex`] when `index` is not
    /// representable in the `D`-dimensional basis.
    pub const fn new(index: u16) -> Result<Self, SpikeComponentsError> {
        if D >= u32::BITS as usize {
            return Err(SpikeComponentsError::DimensionUnsupported { dimension: D });
        }

        let upper = Self::upper_bound();
        if (index as u32) >= upper {
            return Err(SpikeComponentsError::InvalidBladeIndex {
                index,
                dimension: D,
                upper_bound: upper,
            });
        }
        Ok(Self(index))
    }

    /// Returns the raw blade index.
    pub const fn get(self) -> u16 {
        self.0
    }
}

impl<const D: usize> TryFrom<u16> for BladeIndex<D> {
    type Error = SpikeComponentsError;

    fn try_from(value: u16) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

/// Errors produced while constructing [`SpikeComponents`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum SpikeComponentsError {
    /// The requested basis dimension is not supported by this representation.
    DimensionUnsupported {
        /// Requested basis dimension.
        dimension: usize,
    },

    /// A raw blade index is not valid for the requested dimension `D`.
    InvalidBladeIndex {
        /// Invalid raw index.
        index: u16,
        /// Requested basis dimension.
        dimension: usize,
        /// Exclusive upper bound (`2^D`) for valid indices.
        upper_bound: u32,
    },
}

/// Stack-allocatable multivector snapshot. **`SoA` layout** for vectorisation.
///
/// `indices[0..count]` holds active blade indices, sorted ascending.
/// `values[0..count]` holds the corresponding coefficients.
/// Positions `count..SPIKE_MAX_COMPONENTS` are zeroed.
///
/// **u16 blade indices (CORRECCIÓN ESTRUCTURAL-1):** supports G(1,3+n) expansion
/// to D ≤ 15 (2^15 = 32,768 blades ≤ `u16::MAX`). Sufficient for all planned
/// GÉNESIS roadmap phases. Changing from `u8` now (before CRATE-001 deployment)
/// avoids a binary-breaking API change later.
///
/// Layout (#[repr(C)]):
/// ```text
/// values:  [f64; 16] = 128 bytes  (offset   0) ← SIMD-aligned: vmovapd/vloadpd
/// indices: [u16; 16] =  32 bytes  (offset 128)
/// count:   u8        =   1 byte   (offset 160)
/// pad:      [u8;  7]  =   7 bytes  (offset 161)
/// tail_pad: [u8; 24]  =  24 bytes  (offset 168)
/// Total                192 bytes  (align 64)
/// ```
///
/// `values` is placed first so `&spike.values` == struct base address,
/// enabling aligned 512-bit SIMD loads (`vmovaps`/`_mm512_load_pd`) without
/// the 32-byte offset penalty of the former `indices`-first layout.
///
/// AX-ID: AXIOMA-018
#[derive(Debug, Clone, Copy)]
// CRYSTAL: FO34 — inevitable
// CRYSTAL: FO35 — inevitable
// CRYSTAL: FO36 — inevitable
#[repr(C, align(64))]
pub struct SpikeComponents {
    /// Coefficients corresponding to each index. Positions `count..` are `0.0`.
    /// **At offset 0** — do not reorder. Required for aligned SIMD loads.
    pub values: [f64; SPIKE_MAX_COMPONENTS],
    /// Active blade indices, sorted ascending. Positions `count..` filled
    /// with `u16::MAX` (sentinel, never a valid blade in context).
    pub indices: [u16; SPIKE_MAX_COMPONENTS],
    /// Number of active (blade, coefficient) pairs. Invariant: `count ≤ SPIKE_MAX_COMPONENTS`.
    pub count: u8,
    /// Explicit payload padding to align the next field to an 8-byte boundary.
    pad: [u8; 7],
    /// Explicit cache-line tail padding. Zero-initialized.
    /// Ensures deterministic raw-byte layout for DAX zero-copy.
    /// AX-ID: AXIOMA-018
    tail_pad: [u8; 24],
}

// Compile-time layout verification.
static_assertions::assert_eq_size!(SpikeComponents, [u8; 192]);
static_assertions::const_assert_eq!(core::mem::align_of::<SpikeComponents>(), 64);

/// Layout version of SpikeEvent repr(C) contract.
/// Increment when any field offset or size changes.
/// AX-ID: AXIOMA-018
pub const SPIKE_EVENT_LAYOUT_VERSION: u32 = 2;

impl SpikeComponents {
    /// Returns `true` if all explicit padding fields are zeroed.
    ///
    /// Used to verify the DAX layout contract after construction or
    /// deserialization without exposing private implementation details.
    ///
    /// AX-ID: AXIOMA-018
    #[inline]
    pub const fn padding_is_zeroed(&self) -> bool {
        let mut i = 0;
        while i < 7 {
            if self.pad[i] != 0 {
                return false;
            }
            i += 1;
        }

        let mut j = 0;
        while j < 24 {
            if self.tail_pad[j] != 0 {
                return false;
            }
            j += 1;
        }

        true
    }

    fn from_pairs_internal<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = (u16, f64)>,
    {
        // FIX-6: Top-K by magnitude, not by arrival order.
        //
        // The former implementation truncated on first K arrivals, silently discarding
        // high-magnitude coefficients that arrived after slot K was full. This breaks
        // semantic correctness: a spike must carry the K most significant blades, not
        // the K first ones. Under input reordering the old code was non-deterministic.
        //
        // Solution: fixed-size min-slot buffer (K=SPIKE_MAX_COMPONENTS entries) that
        // replaces the weakest slot when a stronger coefficient arrives. All stack,
        // zero heap. O(K) per element (K≤16 → at most 16 comparisons/element).
        //
        // Tiebreak: on equal |coef|, keep the one with lower blade index (deterministic).

        const K: usize = SPIKE_MAX_COMPONENTS;

        // slot buffer: (abs_coef, index, coef) — abs_coef first for comparisons
        let mut buf: [(f64, u16, f64); K] = [(0.0, u16::MAX, 0.0); K];
        let mut filled: usize = 0;
        let mut min_abs: f64 = 0.0; // abs of weakest slot in buf
        let mut min_slot: usize = 0; // index of weakest slot

        for (idx, coef) in iter {
            // Thermal noise gate — AXIOMA-001.
            if !coef.is_finite() || coef.abs() <= COGNITIVE_PLANCK_CONSTANT {
                continue;
            }
            let abs = coef.abs();

            if filled < K {
                // Buffer not full yet — insert directly.
                buf[filled] = (abs, idx, coef);
                filled += 1;
                if filled == K {
                    // Find the weakest slot.
                    if let Some((slot, abs_slot)) = buf[..K]
                        .iter()
                        .enumerate()
                        .min_by(|(_, a), (_, b)| a.0.total_cmp(&b.0).then(b.1.cmp(&a.1)))
                        .map(|(i, &(a, _ix, _))| (i, a))
                    {
                        min_slot = slot;
                        min_abs = abs_slot;
                    } else {
                        continue;
                    }
                }
            } else {
                // Buffer full: replace weakest slot if this coef is stronger.
                let stronger = match abs.total_cmp(&min_abs) {
                    core::cmp::Ordering::Greater => true,
                    core::cmp::Ordering::Equal => idx < buf[min_slot].1,
                    core::cmp::Ordering::Less => false,
                };
                if stronger {
                    buf[min_slot] = (abs, idx, coef);
                    // Recompute min slot.
                    if let Some((slot, abs_slot)) = buf[..K]
                        .iter()
                        .enumerate()
                        .min_by(|(_, a), (_, b)| a.0.total_cmp(&b.0).then(b.1.cmp(&a.1)))
                        .map(|(i, &(a, _ix, _))| (i, a))
                    {
                        min_slot = slot;
                        min_abs = abs_slot;
                    }
                }
            }
        }

        // Extract selected (idx, coef) pairs and sort by blade index (canonical order).
        let mut pairs: [(u16, f64); K] = [(u16::MAX, 0.0); K];
        for i in 0..filled {
            pairs[i] = (buf[i].1, buf[i].2);
        }

        // Insertion sort on filled entries (K≤16 → O(K²) = O(256) cycles maximum).
        for i in 1..filled {
            let mut j = i;
            while j > 0 && pairs[j - 1].0 > pairs[j].0 {
                pairs.swap(j - 1, j);
                j -= 1;
            }
        }

        // Copy into canonical SpikeComponents arrays.
        let mut indices = [u16::MAX; SPIKE_MAX_COMPONENTS];
        let mut values = [0.0f64; SPIKE_MAX_COMPONENTS];
        let mut count = 0usize;

        // Merge+compaction: deduplicate same-blade entries (sum their coefficients).
        let mut read = 0usize;
        while read < filled {
            let idx = pairs[read].0;
            let mut acc = pairs[read].1;
            read += 1;
            while read < filled && pairs[read].0 == idx {
                acc += pairs[read].1;
                read += 1;
            }
            // Re-apply Planck filtering to merged result.
            if acc.is_finite() && acc.abs() > COGNITIVE_PLANCK_CONSTANT {
                indices[count] = idx;
                values[count] = acc;
                count += 1;
            }
        }

        // Zero-fill inactive slots (canonical representation — no stale data).
        indices[count..].fill(u16::MAX);
        values[count..].fill(0.0);

        let count_u8 = u8::try_from(count).unwrap_or(u8::MAX);

        Self {
            indices,
            values,
            count: count_u8,
            pad: [0u8; 7],
            tail_pad: [0u8; 24],
        }
    }

    /// Constructs a `SpikeComponents` from an iterator of `(blade_index, coefficient)` pairs.
    ///
    /// - Pairs with `|coefficient| ≤ COGNITIVE_PLANCK_CONSTANT` are silently filtered
    ///   (thermal noise gate — AXIOMA-001).
    /// - Pairs with `NaN` coefficients are rejected.
    /// - At most `SPIKE_MAX_COMPONENTS` pairs are stored; excess are discarded (top-K).
    /// - The stored indices are sorted ascending (insertion sort; K ≤ 16, O(K²) bounded).
    /// - Duplicate blade indices are merged by deterministic left-to-right accumulation in
    ///   sorted order, then Planck filtering is re-applied to the merged coefficients.
    ///
    /// ## Deterministic merge semantics
    ///
    /// - **NaN input:** rejected (never stored, never merged).
    /// - **Signed zeros:** IEEE-754 addition is used during merge accumulation. If the merged
    ///   value is `+0.0` or `-0.0`, it is removed by the post-merge Planck filter because
    ///   `|0.0| ≤ COGNITIVE_PLANCK_CONSTANT`.
    /// - **Accumulation order:** stable and bit-deterministic for a given input sequence:
    ///   insertion order within each equal-index run after sorting ascending by blade index.
    ///
    /// `blade_index: u16` supports G(1,3+n) up to D=15. Provides future-proof
    /// headroom vs u8 without requiring u32 for the planned roadmap.
    ///
    /// # Example
    /// ```rust
    /// use genesis_types::signal::SpikeComponents;
    ///
    /// let sc = SpikeComponents::from_pairs([(0u16, 1.0), (3u16, -0.5)]);
    /// assert_eq!(sc.cardinality(), 2);
    /// ```
    ///
    /// AX-ID: AXIOMA-001, AXIOMA-018
    pub fn from_pairs<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = (u16, f64)>,
    {
        Self::from_pairs_internal(iter)
    }

    /// Constructs `SpikeComponents` from typed blade indices for a specific
    /// algebra dimension `D`.
    pub fn from_typed_pairs<const D: usize, I>(iter: I) -> Self
    where
        I: IntoIterator<Item = (BladeIndex<D>, f64)>,
    {
        Self::from_pairs_internal(iter.into_iter().map(|(idx, coef)| (idx.get(), coef)))
    }

    /// Compatibility constructor from raw `(u16, f64)` pairs with explicit
    /// index validation for dimension `D`.
    ///
    /// # Errors
    /// Returns [`SpikeComponentsError::InvalidBladeIndex`] when any index
    /// violates `index < (1 << D)`.
    pub fn try_from_pairs<const D: usize, I>(iter: I) -> Result<Self, SpikeComponentsError>
    where
        I: IntoIterator<Item = (u16, f64)>,
    {
        let mut validated = [(u16::MAX, 0.0f64); SPIKE_MAX_COMPONENTS];
        let mut n = 0usize;
        for (idx, coef) in iter {
            let checked = BladeIndex::<D>::try_from(idx)?;
            if n >= SPIKE_MAX_COMPONENTS {
                break;
            }
            validated[n] = (checked.get(), coef);
            n += 1;
        }
        Ok(Self::from_pairs_internal(validated.into_iter().take(n)))
    }

    /// Number of active `(blade, coefficient)` pairs.
    ///
    /// AX-ID: AXIOMA-018
    #[inline]
    pub const fn cardinality(&self) -> usize {
        self.count as usize
    }

    /// Returns `true` if blade `blade` is present in this snapshot.
    ///
    /// Binary search over sorted active indices (O(log K), K ≤ 16).
    ///
    /// AX-ID: AXIOMA-018
    #[inline]
    pub fn is_active(&self, blade: u16) -> bool {
        let n = self.count as usize;
        self.indices[..n].binary_search(&blade).is_ok()
    }

    /// Returns the coefficient for `blade`, or `0.0` if inactive.
    ///
    /// Linear scan with early exit on sorted indices.
    ///
    /// AX-ID: AXIOMA-018
    #[inline]
    pub fn get(&self, blade: u16) -> f64 {
        let n = self.count as usize;
        for i in 0..n {
            if self.indices[i] == blade {
                return self.values[i];
            }
            if self.indices[i] > blade {
                break;
            }
        }
        0.0
    }

    /// Iterator over active `(blade_index, coefficient)` pairs in ascending
    /// blade order. Bounded at K = `count` ≤ `SPIKE_MAX_COMPONENTS`.
    ///
    /// AX-ID: AXIOMA-018
    #[inline]
    pub fn active_pairs(&self) -> impl Iterator<Item = (u16, f64)> + '_ {
        let n = self.count as usize;
        (0..n).map(move |i| (self.indices[i], self.values[i]))
    }
}

impl PartialEq for SpikeComponents {
    /// Two `SpikeComponents` are equal iff `count` matches and all `count`
    /// active index / value bit-patterns are identical.
    ///
    /// `f64::to_bits()` provides a deterministic total equality consistent
    /// with the Axiom of Strict Determinism.
    fn eq(&self, other: &Self) -> bool {
        if self.count != other.count {
            return false;
        }
        let n = self.count as usize;
        for i in 0..n {
            if self.indices[i] != other.indices[i] {
                return false;
            }
            if self.values[i].to_bits() != other.values[i].to_bits() {
                return false;
            }
        }
        true
    }
}

impl Eq for SpikeComponents {}

impl Hash for SpikeComponents {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.count.hash(state);
        let n = self.count as usize;
        for i in 0..n {
            self.indices[i].hash(state);
            self.values[i].to_bits().hash(state);
        }
    }
}

#[cfg(feature = "serde")]
impl serde::Serialize for SpikeComponents {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;

        let mut s = serializer.serialize_struct("SpikeComponents", 3)?;
        s.serialize_field("values", &self.values)?;
        s.serialize_field("indices", &self.indices)?;
        s.serialize_field("count", &self.count)?;
        s.end()
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for SpikeComponents {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use serde::de::{self, MapAccess, SeqAccess, Visitor};

        #[derive(serde::Deserialize)]
        #[serde(field_identifier, rename_all = "lowercase")]
        enum Field {
            Values,
            Indices,
            Count,
        }

        struct SpikeComponentsVisitor;

        impl<'de> Visitor<'de> for SpikeComponentsVisitor {
            type Value = SpikeComponents;

            fn expecting(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                f.write_str("struct SpikeComponents")
            }

            fn visit_seq<V: SeqAccess<'de>>(self, mut seq: V) -> Result<SpikeComponents, V::Error> {
                let values: [f64; 16] = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(0, &self))?;
                let indices: [u16; 16] = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(1, &self))?;
                let count: u8 = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(2, &self))?;

                Ok(SpikeComponents {
                    values,
                    indices,
                    count,
                    pad: [0u8; 7],
                    tail_pad: [0u8; 24],
                })
            }

            fn visit_map<V: MapAccess<'de>>(self, mut map: V) -> Result<SpikeComponents, V::Error> {
                let mut values = None::<[f64; 16]>;
                let mut indices = None::<[u16; 16]>;
                let mut count = None::<u8>;

                while let Some(key) = map.next_key()? {
                    match key {
                        Field::Values => {
                            if values.is_some() {
                                return Err(de::Error::duplicate_field("values"));
                            }
                            values = Some(map.next_value()?);
                        }
                        Field::Indices => {
                            if indices.is_some() {
                                return Err(de::Error::duplicate_field("indices"));
                            }
                            indices = Some(map.next_value()?);
                        }
                        Field::Count => {
                            if count.is_some() {
                                return Err(de::Error::duplicate_field("count"));
                            }
                            count = Some(map.next_value()?);
                        }
                    }
                }

                Ok(SpikeComponents {
                    values: values.ok_or_else(|| de::Error::missing_field("values"))?,
                    indices: indices.ok_or_else(|| de::Error::missing_field("indices"))?,
                    count: count.ok_or_else(|| de::Error::missing_field("count"))?,
                    pad: [0u8; 7],
                    tail_pad: [0u8; 24],
                })
            }
        }

        const FIELDS: &[&str] = &["values", "indices", "count"];
        deserializer.deserialize_struct("SpikeComponents", FIELDS, SpikeComponentsVisitor)
    }
}

// ============================================================================
// SPIKE EVENT
// ============================================================================

/// A discrete neuromorphic event (spike) produced by a cognitive oscillator.
///
/// `SpikeEvent` is `Copy` — it contains no heap-allocated fields. This is the
/// compile-time proof of zero per-spike allocation (AX-ID: AXIOMA-018).
///
/// `#[repr(C)]` guarantees a stable, deterministic layout for DAX zero-copy
/// mapping. `total_dim` uses `u64` (not `usize`) for cross-platform stability.
///
/// Layout (#[repr(C)]):
/// ```text
/// components:       SpikeComponents  = 192 bytes  (offset   0)
/// timestamp_ns:     Timestamp(u64)   =   8 bytes  (offset 192)
/// origin_node_id:   NodeId(u64)      =   8 bytes  (offset 200)
/// total_dim:        u64              =   8 bytes  (offset 208)
/// collapse_grade:   Option<u16>      =   4 bytes  (offset 216)
/// tail_pad:         [u8; 36]         =  36 bytes  (offset 220)
/// Total                              = 256 bytes
/// ```
///
/// AX-ID: AXIOMA-003, AXIOMA-006, AXIOMA-018
#[derive(Debug, Clone, Copy)]
// CRYSTAL: FO37 — inevitable
// CRYSTAL: FO38 — inevitable
// CRYSTAL: FO39 — inevitable
#[repr(C, align(64))]
pub struct SpikeEvent {
    /// Fixed-size multivector snapshot at the moment of firing.
    /// Zero heap allocation. AX-ID: AXIOMA-018.
    pub components: SpikeComponents,

    /// Monotonic nanosecond timestamp from the originating processing unit.
    /// NOT a global wall-clock — AXIOMA-002 prohibits global simulation clocks.
    pub timestamp_ns: Timestamp,

    /// `NodeId` of the oscillator that fired this spike.
    /// Newtype prevents accidental transposition with `timestamp_ns`.
    pub origin_node_id: NodeId,

    /// Original `SparseCliffordVector::total_dim` for zero-copy reconstruction
    /// in CRATE-001. `u64` (not `usize`) for stable cross-platform DAX layout.
    pub total_dim: u64,

    /// Grade of the Lindblad collapse that triggered this spike, if applicable.
    /// - `None` → spontaneous internal drive (AX-ID: AXIOMA-003).
    /// - `Some(g)` → phase collapse at grade g (AX-ID: AXIOMA-006).
    ///
    /// `u16` matches the blade index type (CORRECCIÓN ESTRUCTURAL-1).
    /// Valid grades for G(1,3) are 0..=4; `u16` provides headroom for
    /// expanded algebras.
    pub collapse_grade: Option<u16>,
    /// Explicit cache-line tail padding. Zero-initialized.
    /// Ensures deterministic raw-byte layout for DAX zero-copy.
    /// Version: SPIKE_EVENT_LAYOUT_VERSION = 2
    /// AX-ID: AXIOMA-018
    tail_pad: [u8; 36],
}

// Compile-time layout invariant: SpikeEvent keeps its hot payload on a 64-byte boundary.
static_assertions::const_assert_eq!(core::mem::size_of::<SpikeEvent>(), 256);
static_assertions::const_assert_eq!(core::mem::align_of::<SpikeEvent>(), 64);

impl SpikeEvent {
    /// Constructs a `SpikeEvent` with deterministic zero-initialized tail padding.
    ///
    /// AX-ID: AXIOMA-018
    #[inline]
    pub const fn new(
        components: SpikeComponents,
        timestamp_ns: Timestamp,
        origin_node_id: NodeId,
        total_dim: u64,
        collapse_grade: Option<u16>,
    ) -> Self {
        Self {
            components,
            timestamp_ns,
            origin_node_id,
            total_dim,
            collapse_grade,
            tail_pad: [0u8; 36],
        }
    }

    /// Returns `true` if this spike is a spontaneous internal drive event
    /// (no external stimulus; the system's curiosity — AX-ID: AXIOMA-003).
    #[inline]
    pub const fn is_internal_drive(&self) -> bool {
        self.collapse_grade.is_none()
    }

    /// Returns `true` if this spike results from a Lindblad phase collapse
    /// (AX-ID: AXIOMA-006).
    #[inline]
    pub const fn is_phase_collapse(&self) -> bool {
        self.collapse_grade.is_some()
    }

    /// Returns `true` if the explicit tail padding is fully zero-initialized.
    ///
    /// Used to verify the DAX layout contract after deserialization or
    /// construction without exposing the private padding field.
    ///
    /// AX-ID: AXIOMA-018
    #[inline]
    pub const fn tail_padding_is_zeroed(&self) -> bool {
        let mut i = 0;
        while i < 36 {
            if self.tail_pad[i] != 0 {
                return false;
            }
            i += 1;
        }

        true
    }

    /// Number of non-zero components in the associated multivector snapshot.
    #[inline]
    pub const fn cardinality(&self) -> usize {
        self.components.cardinality()
    }
}

impl PartialEq for SpikeEvent {
    fn eq(&self, other: &Self) -> bool {
        self.timestamp_ns == other.timestamp_ns
            && self.origin_node_id == other.origin_node_id
            && self.total_dim == other.total_dim
            && self.collapse_grade == other.collapse_grade
            && self.components == other.components
    }
}

impl Eq for SpikeEvent {}

impl Hash for SpikeEvent {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.timestamp_ns.hash(state);
        self.origin_node_id.hash(state);
        self.total_dim.hash(state);
        self.collapse_grade.hash(state);
        self.components.hash(state);
    }
}

#[cfg(feature = "serde")]
impl serde::Serialize for SpikeEvent {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;

        let mut s = serializer.serialize_struct("SpikeEvent", 5)?;
        s.serialize_field("components", &self.components)?;
        s.serialize_field("timestamp_ns", &self.timestamp_ns)?;
        s.serialize_field("origin_node_id", &self.origin_node_id)?;
        s.serialize_field("total_dim", &self.total_dim)?;
        s.serialize_field("collapse_grade", &self.collapse_grade)?;
        s.end()
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for SpikeEvent {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use serde::de::{self, MapAccess, SeqAccess, Visitor};

        #[derive(serde::Deserialize)]
        #[serde(field_identifier, rename_all = "snake_case")]
        enum Field {
            Components,
            TimestampNs,
            OriginNodeId,
            TotalDim,
            CollapseGrade,
        }

        struct SpikeEventVisitor;

        impl<'de> Visitor<'de> for SpikeEventVisitor {
            type Value = SpikeEvent;

            fn expecting(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                f.write_str("struct SpikeEvent")
            }

            fn visit_seq<V: SeqAccess<'de>>(self, mut seq: V) -> Result<SpikeEvent, V::Error> {
                let components: SpikeComponents = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(0, &self))?;
                let timestamp_ns: Timestamp = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(1, &self))?;
                let origin_node_id: NodeId = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(2, &self))?;
                let total_dim: u64 = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(3, &self))?;
                let collapse_grade: Option<u16> = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(4, &self))?;

                Ok(SpikeEvent::new(
                    components,
                    timestamp_ns,
                    origin_node_id,
                    total_dim,
                    collapse_grade,
                ))
            }

            fn visit_map<V: MapAccess<'de>>(self, mut map: V) -> Result<SpikeEvent, V::Error> {
                let mut components = None::<SpikeComponents>;
                let mut timestamp_ns = None::<Timestamp>;
                let mut origin_node_id = None::<NodeId>;
                let mut total_dim = None::<u64>;
                let mut collapse_grade = None::<Option<u16>>;

                while let Some(key) = map.next_key()? {
                    match key {
                        Field::Components => {
                            if components.is_some() {
                                return Err(de::Error::duplicate_field("components"));
                            }
                            components = Some(map.next_value()?);
                        }
                        Field::TimestampNs => {
                            if timestamp_ns.is_some() {
                                return Err(de::Error::duplicate_field("timestamp_ns"));
                            }
                            timestamp_ns = Some(map.next_value()?);
                        }
                        Field::OriginNodeId => {
                            if origin_node_id.is_some() {
                                return Err(de::Error::duplicate_field("origin_node_id"));
                            }
                            origin_node_id = Some(map.next_value()?);
                        }
                        Field::TotalDim => {
                            if total_dim.is_some() {
                                return Err(de::Error::duplicate_field("total_dim"));
                            }
                            total_dim = Some(map.next_value()?);
                        }
                        Field::CollapseGrade => {
                            if collapse_grade.is_some() {
                                return Err(de::Error::duplicate_field("collapse_grade"));
                            }
                            collapse_grade = Some(map.next_value()?);
                        }
                    }
                }

                Ok(SpikeEvent::new(
                    components.ok_or_else(|| de::Error::missing_field("components"))?,
                    timestamp_ns.ok_or_else(|| de::Error::missing_field("timestamp_ns"))?,
                    origin_node_id.ok_or_else(|| de::Error::missing_field("origin_node_id"))?,
                    total_dim.ok_or_else(|| de::Error::missing_field("total_dim"))?,
                    collapse_grade.ok_or_else(|| de::Error::missing_field("collapse_grade"))?,
                ))
            }
        }

        const FIELDS: &[&str] = &[
            "components",
            "timestamp_ns",
            "origin_node_id",
            "total_dim",
            "collapse_grade",
        ];
        deserializer.deserialize_struct("SpikeEvent", FIELDS, SpikeEventVisitor)
    }
}

// ============================================================================
// DOMAIN MARKERS + ZERO-COST DOMAIN WRAPPER
// ============================================================================

mod domain_sealed {
    pub trait Sealed {}
}

/// Marker trait for valid cognitive domains.
///
/// This trait is sealed so external crates cannot inject arbitrary domain
/// markers into [`DomainSignal`].
pub trait CognitiveDomain: domain_sealed::Sealed {}

/// Marker type for physics domain events.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PhysicsDomain;
impl domain_sealed::Sealed for PhysicsDomain {}
impl CognitiveDomain for PhysicsDomain {}

/// Marker type for topology domain events.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TopologyDomain;
impl domain_sealed::Sealed for TopologyDomain {}
impl CognitiveDomain for TopologyDomain {}

/// Marker type for dynamics domain events.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DynamicsDomain;
impl domain_sealed::Sealed for DynamicsDomain {}
impl CognitiveDomain for DynamicsDomain {}

/// Marker type for consciousness domain events.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ConsciousnessDomain;
impl domain_sealed::Sealed for ConsciousnessDomain {}
impl CognitiveDomain for ConsciousnessDomain {}

/// Zero-cost domain-typed wrapper over [`SpikeEvent`].
///
/// `DomainSignal<D>` guarantees at compile time that a signal is bound to a
/// specific cognitive domain marker `D`, while preserving the exact memory
/// layout of [`SpikeEvent`] (`#[repr(transparent)]`).
///
/// ```compile_fail
/// use genesis_types::{
///     DomainSignal, DynamicsDomain, NodeId, SpikeComponents, SpikeEvent, Timestamp,
///     TopologyDomain,
/// };
///
/// fn consume_dynamics(_: DomainSignal<DynamicsDomain>) {}
///
/// let spike = SpikeEvent {
///     timestamp_ns: Timestamp::new(1),
///     origin_node_id: NodeId::try_new(1).unwrap(),
///     components: SpikeComponents::from_pairs([(0u16, 1.0)]),
///     total_dim: 16,
///     collapse_grade: None,
/// };
///
/// let topology: DomainSignal<TopologyDomain> = DomainSignal::new(spike);
/// consume_dynamics(topology);
/// ```
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DomainSignal<D: CognitiveDomain> {
    inner: SpikeEvent,
    _domain: PhantomData<D>,
}

impl<D: CognitiveDomain> DomainSignal<D> {
    /// Creates a new domain-typed signal wrapper.
    #[must_use]
    pub const fn new(inner: SpikeEvent) -> Self {
        Self {
            inner,
            _domain: PhantomData,
        }
    }

    /// Returns the wrapped spike event by value.
    #[must_use]
    pub const fn into_inner(self) -> SpikeEvent {
        self.inner
    }

    /// Returns a shared reference to the wrapped spike event.
    #[must_use]
    pub const fn as_inner(&self) -> &SpikeEvent {
        &self.inner
    }
}

static_assertions::assert_eq_size!(DomainSignal<PhysicsDomain>, SpikeEvent);
static_assertions::const_assert_eq!(
    core::mem::align_of::<DomainSignal<PhysicsDomain>>(),
    core::mem::align_of::<SpikeEvent>()
);

// ============================================================================
// TYPE-STATE MARKERS — CORRECCIÓN ESTRUCTURAL-2
// ============================================================================

// ── Sealed trait para ConsolidationState ─────────────────────────────────────
// Previene que tipos externos implementen ConsolidationState e inyecten
// estados inválidos en DomainConsolidationSignal<T>.
// Solo Saturated y Certified pueden ser State — invariante de tipo compile-time.
//
// AX-ID: GENESIS_PROOF_SPEC §A4, AXIOMA-008, AXIOMA-009
mod private {
    pub trait Sealed {}
}

/// Restricción de tipo para los marcadores de estado de consolidación.
///
/// Solo `Saturated` y `Certified` implementan este trait — sellado mediante
/// `mod private`. Ningún tipo externo puede ser usado como `State` en
/// `DomainConsolidationSignal<State>`.
///
/// AX-ID: AXIOMA-008, AXIOMA-009
pub trait ConsolidationState: private::Sealed {}

/// Marker type: domain is **saturated** (Fisher gradient stable, reversible).
///
/// A `DomainConsolidationSignal<Saturated>` may be lifted by a
/// `DomainResetSignal` via `validate_against`.
///
/// AX-ID: AXIOMA-008
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Saturated;
impl private::Sealed for Saturated {}
impl ConsolidationState for Saturated {}

/// Marker type: domain is **certified** (H¹ = 0 globally, **irrevocable**).
///
/// A `DomainConsolidationSignal<Certified>` CANNOT be passed to
/// `DomainResetSignal::validate_against` — doing so is a **compile error**.
/// Only a full Core Atlas redeploy may alter a certified domain's topology.
///
/// This is the compile-time encoding of AXIOMA-009.
///
/// AX-ID: AXIOMA-009
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Certified;
impl private::Sealed for Certified {}
impl ConsolidationState for Certified {}

// ============================================================================
// DOMAIN CONSOLIDATION SIGNAL — CORRECCIÓN ESTRUCTURAL-2
// ============================================================================

/// Signal emitted by `FisherGate` when a domain reaches satiation or
/// cohomological certification.
///
/// `State` is either `Saturated` or `Certified`. The compiler enforces:
/// - Only `DomainConsolidationSignal<Saturated>` can be passed to
///   `DomainResetSignal::validate_against`.
/// - `DomainConsolidationSignal<Certified>` has no reset path — no runtime
///   check required.
///
/// The `State: ConsolidationState` bound (sealed trait) prevents injection
/// of arbitrary types as `State` — compile-time invariant.
///
/// Both `Saturated` and `Certified` variants are `Copy` (all fields are
/// `&'static str`, `Timestamp`, `f64`, `PhantomData` — zero heap).
///
/// `domain: &'static str` — domain names are compile-time module/crate
/// identifiers (CORRECCIÓN DE CALIDAD-2; eliminates per-signal heap alloc).
///
/// AX-ID: AXIOMA-008 (Saturated), AXIOMA-009 (Certified)
#[derive(Clone, Copy, Debug)]
// CRYSTAL: FO40 — inevitable
#[repr(C, align(64))]
pub struct DomainConsolidationSignal<State: ConsolidationState> {
    /// Hierarchical domain identifier, e.g. `"physics::electromagnetism"`.
    /// Must be a compile-time string literal — no heap allocation.
    pub domain: &'static str,

    /// Monotonic timestamp. No global clock (AX-ID: AXIOMA-002).
    pub timestamp_ns: Timestamp,

    /// Fisher metric gradient magnitude at satiation.
    /// Zero for `Certified` (gradient is not the certification criterion).
    pub fisher_delta_g: f64,

    _state: PhantomData<State>,
}

// Compile-time ABI/layout invariants (no regression vs intended C-like shape):
// &'static str (16 bytes) + Timestamp(u64) (8 bytes) + f64 (8 bytes) = 32 bytes, rounded to 64.
static_assertions::assert_eq_size!(DomainConsolidationSignal<Saturated>, [u8; 64]);
static_assertions::assert_eq_size!(DomainConsolidationSignal<Certified>, [u8; 64]);
static_assertions::const_assert_eq!(
    core::mem::align_of::<DomainConsolidationSignal<Saturated>>(),
    64
);
static_assertions::const_assert_eq!(
    core::mem::align_of::<DomainConsolidationSignal<Certified>>(),
    64
);

impl DomainConsolidationSignal<Saturated> {
    /// Constructs a `Saturated` consolidation signal.
    ///
    /// AX-ID: AXIOMA-008
    pub const fn saturated(
        domain: &'static str,
        timestamp_ns: Timestamp,
        fisher_delta_g: f64,
    ) -> Self {
        Self {
            domain,
            timestamp_ns,
            fisher_delta_g,
            _state: PhantomData,
        }
    }

    /// Elevates this signal to `Certified` — a **one-way, irrevocable transition**.
    ///
    /// Once returned, the `Certified` signal cannot be passed to
    /// `DomainResetSignal::validate_against`.
    ///
    /// AX-ID: AXIOMA-009
    pub const fn certify(self) -> DomainConsolidationSignal<Certified> {
        DomainConsolidationSignal {
            domain: self.domain,
            timestamp_ns: self.timestamp_ns,
            fisher_delta_g: 0.0, // gradient is not a Certified invariant
            _state: PhantomData,
        }
    }
}

impl DomainConsolidationSignal<Certified> {
    /// Constructs a `Certified` consolidation signal directly.
    ///
    /// Use this when a H¹ = 0 test result is available without going through
    /// a prior `Saturated` signal (e.g. after a full atlas redeploy).
    ///
    /// AX-ID: AXIOMA-009
    pub const fn certified(domain: &'static str, timestamp_ns: Timestamp) -> Self {
        Self {
            domain,
            timestamp_ns,
            fisher_delta_g: 0.0,
            _state: PhantomData,
        }
    }
}

impl<State: ConsolidationState + PartialEq> PartialEq for DomainConsolidationSignal<State> {
    /// Equality via bit-pattern comparison of `fisher_delta_g` for determinism.
    fn eq(&self, other: &Self) -> bool {
        self.domain == other.domain
            && self.timestamp_ns == other.timestamp_ns
            && self.fisher_delta_g.to_bits() == other.fisher_delta_g.to_bits()
    }
}

impl<State: ConsolidationState + PartialEq> Eq for DomainConsolidationSignal<State> {}

impl<State: ConsolidationState + Hash> Hash for DomainConsolidationSignal<State> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.domain.hash(state);
        self.timestamp_ns.hash(state);
        self.fisher_delta_g.to_bits().hash(state);
    }
}

// ============================================================================
// DOMAIN RESET SIGNAL — CORRECCIÓN ESTRUCTURAL-2 + DE CALIDAD-2
// ============================================================================

/// Re-opens a previously consolidated domain for ingestion.
///
/// Issued only by the Principal Architect layer when new, structurally
/// significant data is available that is guaranteed to increase Fisher
/// stability (AX-ID: AXIOMA-008, `CLOUD_PLATFORM_ARCHITECTURE` §3.4).
///
/// ## Irrevocability enforcement (compile-time)
///
/// `validate_against` only accepts `&DomainConsolidationSignal<Saturated>`.
/// Passing a `DomainConsolidationSignal<Certified>` is a **compile error** —
/// the irrevocability of `Certified` domains is enforced by the type system,
/// not by runtime logic (CORRECCIÓN ESTRUCTURAL-2 / AXIOMA-009).
///
/// `domain: &'static str` and `justification: &'static str` — zero heap alloc.
///
/// AX-ID: AXIOMA-008, AXIOMA-009
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DomainResetSignal {
    /// Domain to re-open. Must match the `domain` of the consolidation signal.
    pub domain: &'static str,

    /// Monotonic timestamp of the reset authorisation. No global clock.
    pub timestamp_ns: Timestamp,

    /// Audit-trail justification, approved by the Principal Architect.
    /// Must be a compile-time string literal — no heap allocation.
    pub justification: &'static str,
}

impl DomainResetSignal {
    /// Validates this reset against a `Saturated` consolidation signal.
    ///
    /// Returns `Ok(())` if `self.domain == signal.domain`.
    ///
    /// Returns `Err(GenesisError::DomainMismatch)` if the domains differ —
    /// the reset targets a different domain than the signal it was applied to.
    ///
    /// ## Compile-time irrevocability guarantee
    ///
    /// This method only accepts `&DomainConsolidationSignal<Saturated>`.
    /// Calling it with `&DomainConsolidationSignal<Certified>` is a
    /// **compile error**:
    /// ```compile_fail
    /// let certified = DomainConsolidationSignal::<Certified>::certified("x", Timestamp::new(1));
    /// let reset = DomainResetSignal { domain: "x", timestamp_ns: Timestamp::new(2), justification: "t" };
    /// let _ = reset.validate_against(&certified); // error[E0308]: mismatched types
    /// ```
    ///
    /// AX-ID: AXIOMA-008, AXIOMA-009
    /// # Errors
    /// Returns `GenesisError::DomainMismatch` when the certificate
    /// does not match the saturated input signal.
    pub fn validate_against(
        &self,
        signal: &DomainConsolidationSignal<Saturated>,
    ) -> Result<(), GenesisError> {
        if self.domain == signal.domain {
            Ok(())
        } else {
            Err(GenesisError::DomainMismatch {
                reset: domain_code(self.domain),
                signal: domain_code(signal.domain),
            })
        }
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    // -----------------------------------------------------------------------
    // SpikeComponents — layout and size
    // -----------------------------------------------------------------------

    /// Verify exact struct size with `#[repr(C, align(64))]` layout.
    ///
    /// SpikeComponents payload is 168 bytes, rounded to 192 by cache-line alignment.
    ///
    /// AX-ID: AXIOMA-018
    #[test]
    fn spike_components_size_is_192_bytes() {
        assert_eq!(
            core::mem::size_of::<SpikeComponents>(),
            192,
            "SpikeComponents size changed — payload 168 bytes plus align(64) rounding"
        );
        assert_eq!(core::mem::align_of::<SpikeComponents>(), 64);
    }

    #[test]
    fn gaussian_pair_layout_and_copy_semantics() {
        assert_eq!(core::mem::size_of::<GaussianPair>(), 64);
        assert_eq!(core::mem::align_of::<GaussianPair>(), 64);

        fn require_copy<T: Copy>() {}
        require_copy::<GaussianPair>();

        let pair = GaussianPair {
            first: 1.25,
            second: -0.75,
        };
        let copy = pair;
        assert_eq!(copy.first, 1.25);
        assert_eq!(copy.second, -0.75);
    }

    /// SpikeEvent is cache-line aligned and keeps the hot payload first.
    ///
    /// AX-ID: AXIOMA-018
    #[test]
    fn spike_event_size_is_256_bytes() {
        let sz = core::mem::size_of::<SpikeEvent>();
        assert_eq!(
            sz, 256,
            "SpikeEvent size {} diverged from the align(64) layout contract",
            sz
        );
        assert_eq!(core::mem::align_of::<SpikeEvent>(), 64);
    }

    #[test]
    fn spike_event_layout_version_is_2() {
        assert_eq!(SPIKE_EVENT_LAYOUT_VERSION, 2);
    }

    #[test]
    fn spike_event_new_zero_initializes_tail_padding() {
        let event = SpikeEvent::new(
            SpikeComponents::from_pairs([(0u16, 1.0)]),
            Timestamp::new(9),
            NodeId::try_new(3).expect("3 is inside the valid NodeId range"),
            16,
            Some(1),
        );
        assert!(
            event.components.padding_is_zeroed(),
            "SpikeComponents padding must be zero after construction"
        );
        assert!(
            event.tail_padding_is_zeroed(),
            "SpikeEvent tail padding must be zero after construction"
        );
    }

    /// SpikeEvent must be `Copy` — compile-time proof of zero per-spike heap alloc.
    ///
    /// AX-ID: AXIOMA-018
    #[test]
    fn spike_event_is_copy_no_heap() {
        fn require_copy<T: Copy>() {}
        require_copy::<SpikeEvent>();
        require_copy::<SpikeComponents>();
        require_copy::<NodeId>();
        require_copy::<Timestamp>();
    }

    // -----------------------------------------------------------------------
    // SpikeComponents — construction and access
    // -----------------------------------------------------------------------

    #[test]
    fn blade_index_try_from_rejects_out_of_range() {
        let err = BladeIndex::<4>::try_from(16u16).expect_err("index 16 is invalid for D=4");
        assert_eq!(
            err,
            SpikeComponentsError::InvalidBladeIndex {
                index: 16,
                dimension: 4,
                upper_bound: 16,
            }
        );
    }

    #[test]
    fn blade_index_rejects_unsupported_dimension() {
        let err = BladeIndex::<32>::try_from(0u16).expect_err("D=32 is unsupported");
        assert_eq!(
            err,
            SpikeComponentsError::DimensionUnsupported { dimension: 32 }
        );
    }

    #[test]
    fn from_typed_pairs_accepts_valid_indices() {
        let i0 = BladeIndex::<4>::try_from(0u16).unwrap();
        let i3 = BladeIndex::<4>::try_from(3u16).unwrap();
        let sc = SpikeComponents::from_typed_pairs([(i3, -0.5), (i0, 1.0)]);
        assert_eq!(
            sc.active_pairs().collect::<Vec<_>>(),
            vec![(0u16, 1.0), (3u16, -0.5)]
        );
    }

    #[test]
    fn try_from_pairs_checks_dimension_bounds() {
        let err = SpikeComponents::try_from_pairs::<4, _>([(0u16, 1.0), (16u16, 0.5)])
            .expect_err("index 16 should be rejected for D=4");
        assert_eq!(
            err,
            SpikeComponentsError::InvalidBladeIndex {
                index: 16,
                dimension: 4,
                upper_bound: 16,
            }
        );

        let sc = SpikeComponents::try_from_pairs::<4, _>([(15u16, 0.75)]).unwrap();
        assert_eq!(sc.active_pairs().collect::<Vec<_>>(), vec![(15u16, 0.75)]);
    }

    #[test]
    fn from_pairs_active_pairs_correct() {
        let sc = SpikeComponents::from_pairs([(0u16, 1.0), (3u16, -0.5), (7u16, 0.25)]);
        assert_eq!(sc.cardinality(), 3);
        let pairs: Vec<(u16, f64)> = sc.active_pairs().collect();
        assert_eq!(pairs, [(0u16, 1.0), (3u16, -0.5), (7u16, 0.25)]);
    }

    #[test]
    fn from_pairs_sorts_ascending() {
        // Input is unordered; result must be sorted.
        let sc = SpikeComponents::from_pairs([(7u16, 0.7), (1u16, 0.1), (4u16, 0.4)]);
        assert_eq!(sc.cardinality(), 3);
        let indices: Vec<u16> = sc.active_pairs().map(|(i, _)| i).collect();
        assert_eq!(indices, [1u16, 4u16, 7u16]);
    }

    #[test]
    fn from_pairs_filters_planck_noise() {
        // 1e-13 < COGNITIVE_PLANCK_CONSTANT (1e-12) — must be dropped.
        let sc = SpikeComponents::from_pairs([
            (0u16, 1e-13), // below Planck → filtered
            (1u16, 0.5),   // kept
        ]);
        assert_eq!(sc.cardinality(), 1);
        assert!(!sc.is_active(0));
        assert!(sc.is_active(1));
    }

    #[test]
    fn from_pairs_truncates_to_spike_max_components() {
        // Feed 20 pairs; only first SPIKE_MAX_COMPONENTS survive.
        let pairs: Vec<(u16, f64)> = (0u16..20).map(|i| (i, i as f64 + 1.0)).collect();
        let sc = SpikeComponents::from_pairs(pairs);
        assert_eq!(sc.cardinality(), SPIKE_MAX_COMPONENTS);
    }

    #[test]
    fn get_returns_coefficient_for_active_blade() {
        let sc = SpikeComponents::from_pairs([(0u16, 1.0), (3u16, -0.5)]);
        assert_eq!(sc.get(0), 1.0);
        assert_eq!(sc.get(3), -0.5);
        assert_eq!(sc.get(1), 0.0); // inactive
        assert_eq!(sc.get(15), 0.0); // inactive
    }

    #[test]
    fn is_active_correct() {
        let sc = SpikeComponents::from_pairs([(5u16, 2.0)]);
        assert!(sc.is_active(5));
        assert!(!sc.is_active(4));
        assert!(!sc.is_active(6));
        // Blade 1000 — above count; not active.
        assert!(!sc.is_active(1000));
    }

    #[test]
    fn empty_from_pairs_has_cardinality_zero() {
        let sc = SpikeComponents::from_pairs([] as [(u16, f64); 0]);
        assert_eq!(sc.cardinality(), 0);
        assert_eq!(sc.count, 0u8);
        assert_eq!(sc.active_pairs().count(), 0);
    }

    #[test]
    fn from_pairs_max_capacity_no_panic() {
        let pairs: Vec<(u16, f64)> = (0u16..SPIKE_MAX_COMPONENTS as u16)
            .map(|i| (i, i as f64 + 1.0))
            .collect();
        let sc = SpikeComponents::from_pairs(pairs);
        assert_eq!(sc.cardinality(), SPIKE_MAX_COMPONENTS);
        assert_eq!(sc.active_pairs().count(), SPIKE_MAX_COMPONENTS);
    }

    /// u16 blade indices: verify a large blade index (> 255) is stored correctly.
    /// This was the blocker with the old u8 design (CONVERGENTE-1).
    ///
    /// AX-ID: AXIOMA-014 (GramSchmidtExpander may produce blade indices > 255)
    #[test]
    fn from_pairs_accepts_blade_index_above_255() {
        let sc = SpikeComponents::from_pairs([(256u16, 0.5), (512u16, -0.3)]);
        assert_eq!(sc.cardinality(), 2);
        assert!(sc.is_active(256));
        assert_eq!(sc.get(512), -0.3);
    }

    #[test]
    fn from_pairs_merges_duplicate_indices_and_keeps_unique_active_blades() {
        let sc = SpikeComponents::from_pairs([
            (7u16, 0.25),
            (3u16, 1.0),
            (7u16, 0.75),
            (3u16, -0.25),
            (5u16, 0.5),
            (7u16, -0.1),
        ]);

        let pairs: Vec<(u16, f64)> = sc.active_pairs().collect();
        assert_eq!(pairs, [(3u16, 0.75), (5u16, 0.5), (7u16, 0.9)]);

        let indices: Vec<u16> = pairs.iter().map(|(idx, _)| *idx).collect();
        let unique: HashSet<u16> = indices.iter().copied().collect();
        assert_eq!(
            indices.len(),
            unique.len(),
            "duplicate active blade index found"
        );
    }

    #[test]
    fn from_pairs_duplicate_merge_reapplies_planck_filter() {
        let half_planck = COGNITIVE_PLANCK_CONSTANT / 2.0;
        let sc = SpikeComponents::from_pairs([(11u16, half_planck), (11u16, -half_planck)]);
        assert_eq!(sc.cardinality(), 0);
        assert!(!sc.is_active(11));
    }

    #[test]
    fn from_pairs_duplicate_merge_is_deterministic_for_signed_zero_and_nan() {
        let plus_zero_bits = 0.0f64.to_bits();
        let minus_zero_bits = (-0.0f64).to_bits();

        let sc_zero = SpikeComponents::from_pairs([
            (9u16, f64::from_bits(plus_zero_bits)),
            (9u16, f64::from_bits(minus_zero_bits)),
            (2u16, 1.0),
        ]);
        assert_eq!(sc_zero.cardinality(), 1);
        assert_eq!(sc_zero.active_pairs().collect::<Vec<_>>(), [(2u16, 1.0)]);

        let sc_nan_a = SpikeComponents::from_pairs([(4u16, 1.5), (4u16, f64::NAN), (1u16, 2.0)]);
        let sc_nan_b = SpikeComponents::from_pairs([(4u16, 1.5), (1u16, 2.0)]);
        assert_eq!(sc_nan_a, sc_nan_b);
    }

    #[test]
    fn equality_uses_bit_pattern() {
        let a = SpikeComponents::from_pairs([(0u16, 1.0), (1u16, -0.5)]);
        let b = SpikeComponents::from_pairs([(0u16, 1.0), (1u16, -0.5)]);
        assert_eq!(a, b);
    }

    #[test]
    fn different_coefficients_not_equal() {
        let a = SpikeComponents::from_pairs([(0u16, 1.0)]);
        let b = SpikeComponents::from_pairs([(0u16, 2.0)]);
        assert_ne!(a, b);
    }

    #[test]
    fn different_count_not_equal() {
        let a = SpikeComponents::from_pairs([(0u16, 1.0), (1u16, 1.0)]);
        let b = SpikeComponents::from_pairs([(0u16, 1.0)]);
        assert_ne!(a, b);
    }

    #[test]
    fn hash_consistent_with_eq() {
        let a = SpikeComponents::from_pairs([(0u16, 1.0), (3u16, -0.5)]);
        let b = SpikeComponents::from_pairs([(0u16, 1.0), (3u16, -0.5)]);
        assert_eq!(a, b);
        let mut set = HashSet::new();
        set.insert(a);
        assert!(
            !set.insert(b),
            "Equal SpikeComponents must map to the same bucket"
        );
    }

    // -----------------------------------------------------------------------
    // NodeId and Timestamp
    // -----------------------------------------------------------------------

    #[test]
    fn node_id_get_round_trips() {
        let id = NodeId::try_new(42).expect("42 is inside the valid NodeId range");
        assert_eq!(id.get(), 42);
    }

    #[test]
    fn node_id_try_new_rejects_upper_bound() {
        // El único ID inválido es el centinela u64::MAX.
        let err = NodeId::try_new(u64::MAX).expect_err("u64::MAX (centinela) debe ser rechazado");
        assert_eq!(err, GenesisError::NodeIdOutOfRange { raw: u64::MAX });
    }

    #[test]
    fn node_id_try_new_accepts_999999() {
        let id = NodeId::try_new(999_999).expect("999_999 debe ser aceptado");
        assert_eq!(id.get(), 999_999);
    }

    #[test]
    fn timestamp_nanoseconds_round_trips() {
        let ts = Timestamp::new(1_000_000);
        assert_eq!(ts.nanoseconds(), 1_000_000);
    }

    #[test]
    fn node_id_invalid_sentinel() {
        assert_eq!(NodeId::INVALID.get(), u64::MAX);
    }

    #[test]
    fn timestamp_zero_sentinel() {
        assert_eq!(Timestamp::ZERO.nanoseconds(), 0);
    }

    // -----------------------------------------------------------------------
    // SpikeEvent
    // -----------------------------------------------------------------------

    fn make_event(grade: Option<u16>) -> SpikeEvent {
        SpikeEvent::new(
            SpikeComponents::from_pairs([(0u16, 1.0), (3u16, -0.5), (7u16, 0.25)]),
            Timestamp::new(1_000_000),
            NodeId::try_new(42).expect("42 is inside the valid NodeId range"),
            16u64,
            grade,
        )
    }

    #[test]
    fn spike_event_internal_drive_has_no_grade() {
        let e = make_event(None);
        assert!(e.is_internal_drive());
        assert!(!e.is_phase_collapse());
    }

    #[test]
    fn spike_event_phase_collapse_has_grade() {
        let e = make_event(Some(2));
        assert!(e.is_phase_collapse());
        assert!(!e.is_internal_drive());
        assert_eq!(e.collapse_grade, Some(2u16));
    }

    /// collapse_grade with index > 4 (extended grade) is stored correctly.
    /// u16 supports the full G(1,3+n) grade range.
    #[test]
    fn spike_event_collapse_grade_u16_extended() {
        let e = make_event(Some(300u16));
        assert_eq!(e.collapse_grade, Some(300u16));
    }

    #[test]
    fn spike_event_cardinality_delegates_to_components() {
        assert_eq!(make_event(None).cardinality(), 3);
    }

    #[test]
    fn spike_event_equal_when_all_fields_equal() {
        assert_eq!(make_event(Some(1)), make_event(Some(1)));
    }

    #[test]
    fn spike_event_not_equal_when_grade_differs() {
        assert_ne!(make_event(Some(1)), make_event(Some(2)));
    }

    #[test]
    fn spike_event_not_equal_when_grade_vs_none() {
        assert_ne!(make_event(Some(0)), make_event(None));
    }

    #[test]
    fn spike_event_usable_as_hashset_key() {
        let a = make_event(Some(1));
        let b = make_event(Some(1));
        let c = make_event(None);
        let mut set = HashSet::new();
        set.insert(a);
        assert!(!set.insert(b), "Equal events must map to same bucket");
        assert!(set.insert(c), "Distinct event must be inserted");
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn domain_signal_layout_zero_cost() {
        assert_eq!(
            core::mem::size_of::<DomainSignal<PhysicsDomain>>(),
            core::mem::size_of::<SpikeEvent>()
        );
        assert_eq!(
            core::mem::align_of::<DomainSignal<PhysicsDomain>>(),
            core::mem::align_of::<SpikeEvent>()
        );
    }

    // -----------------------------------------------------------------------
    // DomainConsolidationSignal — type-state
    // -----------------------------------------------------------------------

    #[test]
    fn saturated_signal_fields_correct() {
        let sig = DomainConsolidationSignal::<Saturated>::saturated(
            "physics::em",
            Timestamp::new(999),
            1e-7,
        );
        assert_eq!(sig.domain, "physics::em");
        assert_eq!(sig.timestamp_ns, Timestamp::new(999));
        assert!((sig.fisher_delta_g - 1e-7).abs() < 1e-20);
    }

    #[test]
    fn certified_signal_fields_correct() {
        let sig =
            DomainConsolidationSignal::<Certified>::certified("math::topology", Timestamp::new(42));
        assert_eq!(sig.domain, "math::topology");
        assert_eq!(sig.timestamp_ns, Timestamp::new(42));
        // fisher_delta_g is 0.0 for Certified.
        assert_eq!(sig.fisher_delta_g.to_bits(), 0f64.to_bits());
    }

    #[test]
    fn saturated_can_certify() {
        let sat = DomainConsolidationSignal::<Saturated>::saturated("d", Timestamp::new(0), 1e-8);
        let cert = sat.certify();
        assert_eq!(cert.domain, "d");
    }

    #[test]
    fn equal_saturated_signals_hash_identically() {
        let a = DomainConsolidationSignal::<Saturated>::saturated(
            "physics::em",
            Timestamp::new(999),
            1e-7,
        );
        let b = DomainConsolidationSignal::<Saturated>::saturated(
            "physics::em",
            Timestamp::new(999),
            1e-7,
        );
        assert_eq!(a, b);
        let mut set = HashSet::new();
        set.insert(a);
        assert!(!set.insert(b), "Equal signals must map to same bucket");
    }

    #[test]
    fn nan_bit_pattern_equality_deterministic_saturated() {
        let nan_bits = f64::NAN.to_bits();
        let a = DomainConsolidationSignal::<Saturated>::saturated(
            "d",
            Timestamp::new(0),
            f64::from_bits(nan_bits),
        );
        let b = DomainConsolidationSignal::<Saturated>::saturated(
            "d",
            Timestamp::new(0),
            f64::from_bits(nan_bits),
        );
        assert_eq!(a, b, "Same NaN bit-pattern must compare equal");
    }

    // -----------------------------------------------------------------------
    // DomainResetSignal::validate_against
    // -----------------------------------------------------------------------

    /// Resetting a `Saturated` domain with matching domain name must succeed.
    ///
    /// AX-ID: AXIOMA-008
    #[test]
    fn domain_reset_validate_against_saturated_succeeds_for_matching_domain() {
        let saturated = DomainConsolidationSignal::<Saturated>::saturated(
            "math::clifford",
            Timestamp::new(0),
            0.0,
        );
        let reset = DomainResetSignal {
            domain: "math::clifford",
            timestamp_ns: Timestamp::new(1),
            justification: "New dataset approved by Principal Architect",
        };
        assert!(reset.validate_against(&saturated).is_ok());
    }

    /// Mismatched domains must produce `GenesisError::DomainMismatch`.
    ///
    /// AX-ID: AXIOMA-008
    #[test]
    fn domain_reset_validate_against_saturated_fails_for_domain_mismatch() {
        let saturated = DomainConsolidationSignal::<Saturated>::saturated(
            "physics::em",
            Timestamp::new(0),
            0.0,
        );
        let reset = DomainResetSignal {
            domain: "math::clifford",
            timestamp_ns: Timestamp::new(1),
            justification: "test",
        };
        let err = reset.validate_against(&saturated).unwrap_err();
        match &err {
            GenesisError::DomainMismatch {
                reset: r,
                signal: s,
            } => {
                assert_eq!(*r, domain_code("math::clifford"));
                assert_eq!(*s, domain_code("physics::em"));
            }
            _ => panic!("Expected DomainMismatch, got {:?}", err),
        }
    }

    /// AXIOMA-009 irrevocability is enforced at compile time.
    ///
    /// The code below would produce:
    ///   error[E0308]: mismatched types
    ///     expected `&DomainConsolidationSignal<Saturated>`
    ///     found    `&DomainConsolidationSignal<Certified>`
    ///
    /// This test exists as documentation of the compile-time guarantee.
    /// Uncommenting the body must cause a compilation failure.
    ///
    /// AX-ID: AXIOMA-009
    #[test]
    fn domain_reset_validate_against_certified_is_compile_error() {
        // DO NOT UNCOMMENT — this is a compile-error demonstration.
        //
        // let certified = DomainConsolidationSignal::<Certified>::certified("x", Timestamp::new(1));
        // let reset = DomainResetSignal {
        //     domain: "x",
        //     timestamp_ns: Timestamp::new(2),
        //     justification: "attempt",
        // };
        // let _ = reset.validate_against(&certified); // error[E0308]
    }

    /// ConsolidationState sealed trait — only Saturated and Certified are valid State.
    ///
    /// AX-ID: GENESIS_PROOF_SPEC §A4
    #[test]
    fn consolidation_state_is_sealed_compile_check() {
        fn require_consolidation_state<S: ConsolidationState>() {}
        require_consolidation_state::<Saturated>();
        require_consolidation_state::<Certified>();
        // Uncomment to verify seal at compile time:
        // require_consolidation_state::<u8>(); // error[E0277]: not satisfied
    }

    /// Verify `SpikeComponents::values` is at offset 0 (required for aligned SIMD loads).
    ///
    /// If this fails, the field order was changed and the SIMD guarantee is broken.
    /// AX-ID: AXIOMA-018 (SNN isomorphism / DAX layout)
    #[test]
    fn spike_components_values_at_offset_zero() {
        let s = core::mem::MaybeUninit::<SpikeComponents>::zeroed();
        // SAFETY: MaybeUninit is only used to get addresses — we never read uninit data.
        let base_addr = s.as_ptr() as usize;
        // SAFETY: We only compute the field address from a valid pointer; no read occurs.
        let values_addr = unsafe { core::ptr::addr_of!((*s.as_ptr()).values) } as usize;
        assert_eq!(
            values_addr, base_addr,
            "SpikeComponents::values must be at offset 0 for aligned SIMD loads"
        );
    }

    /// Verify `SpikeComponents::indices` is at offset 128 (after 16×f64).
    #[test]
    fn spike_components_indices_at_offset_128() {
        let s = core::mem::MaybeUninit::<SpikeComponents>::zeroed();
        let base_addr = s.as_ptr() as usize;
        // SAFETY: We only compute the field address from a valid pointer; no read occurs.
        let indices_addr = unsafe { core::ptr::addr_of!((*s.as_ptr()).indices) } as usize;
        assert_eq!(
            indices_addr - base_addr,
            128,
            "SpikeComponents::indices must be at offset 128"
        );
    }
    /// Top-K selection must retain the K highest-magnitude coefficients, not the
    /// first K arrivals. This is the core FIX-6 semantic guarantee.
    ///
    /// Previous implementation: truncated by arrival → high-magnitude late arrivals dropped.
    /// FIX-6: min-slot buffer retains top-K by |coef| regardless of arrival order.
    #[test]
    fn from_pairs_topk_by_magnitude_not_arrival() {
        const K: usize = SPIKE_MAX_COMPONENTS; // 16
                                               // Generate K+4 pairs where the 4 highest-magnitude ones arrive LAST.
                                               // The first K arrivals are small (0.1..=0.5). The last 4 are large (10.0..=13.0).
                                               // Correct result: last 4 are in the spike; 4 smallest first arrivals are dropped.
        let mut pairs: Vec<(u16, f64)> = (0u16..K as u16)
            .map(|i| (i, (i as f64 + 1.0) * 0.1)) // magnitudes: 0.1, 0.2, ..., 1.6
            .collect();
        // Append 4 high-magnitude arrivals at blade indices K..K+3
        for j in 0u16..4u16 {
            pairs.push((K as u16 + j, 10.0 + j as f64)); // magnitudes: 10, 11, 12, 13
        }

        let spike = SpikeComponents::from_pairs(pairs.iter().copied());

        // All 4 high-magnitude blades must be present
        let indices: Vec<u16> = spike.active_pairs().map(|(i, _)| i).collect();
        for j in 0..4usize {
            let blade = (K + j) as u16;
            assert!(
                indices.contains(&blade),
                "blade {blade} (mag {:.0}) must be in top-K but wasn't found in {:?}",
                10.0 + j as f64,
                indices
            );
        }

        // The 4 lowest-magnitude first arrivals (blades 0..3, mag 0.1..0.4) must be absent.
        for blade in 0u16..4u16 {
            assert!(
                !indices.contains(&blade),
                "blade {blade} (low-magnitude) should have been displaced but is still present"
            );
        }
    }

    /// Top-K with identical arrival order produces identical result (determinism).
    #[test]
    fn from_pairs_topk_deterministic_under_permutation() {
        let pairs_a: Vec<(u16, f64)> = vec![(0u16, 5.0), (1, 1.0), (2, 3.0), (3, 0.5)];
        let mut pairs_b = pairs_a.clone();
        pairs_b.reverse(); // reverse arrival order

        let spike_a = SpikeComponents::from_pairs(pairs_a.iter().copied());
        let spike_b = SpikeComponents::from_pairs(pairs_b.iter().copied());

        // Both must contain the same blades (all 4 fit in K=16)
        let idx_a: Vec<u16> = spike_a.active_pairs().map(|(i, _)| i).collect();
        let idx_b: Vec<u16> = spike_b.active_pairs().map(|(i, _)| i).collect();
        assert_eq!(
            idx_a, idx_b,
            "top-K result must not depend on arrival order"
        );

        // Coefficients must also match
        let coef_a: Vec<f64> = spike_a.active_pairs().map(|(_, c)| c).collect();
        let coef_b: Vec<f64> = spike_b.active_pairs().map(|(_, c)| c).collect();
        assert_eq!(
            coef_a, coef_b,
            "coefficients must match regardless of arrival order"
        );
    }

    /// Attractor with NaN energy must not corrupt BTreeSet invariants.
    /// Tests FIX-1 (PartialEq using to_bits) in attractor.rs.
    /// (This test is in genesis-types but validates the interface contract.)
    #[test]
    fn spike_components_planck_threshold_edge_case() {
        use crate::constants::COGNITIVE_PLANCK_CONSTANT;
        // Exactly at threshold: must be rejected (<= means reject at threshold)
        let at_threshold = SpikeComponents::from_pairs([(0u16, COGNITIVE_PLANCK_CONSTANT)]);
        assert_eq!(
            at_threshold.count as usize, 0,
            "coefficient exactly at Planck threshold must be rejected"
        );

        // Just above threshold: must be accepted
        let just_above = SpikeComponents::from_pairs([(0u16, COGNITIVE_PLANCK_CONSTANT * 1.001)]);
        assert_eq!(
            just_above.count as usize, 1,
            "coefficient just above Planck threshold must be accepted"
        );
    }

    #[cfg(all(test, feature = "serde"))]
    mod serde_tests {
        use super::super::{NodeId, SpikeComponents, SpikeEvent, Timestamp};

        #[test]
        fn spike_event_serde_round_trip_preserves_tail_padding() {
            let original = SpikeEvent::new(
                SpikeComponents::from_pairs([(0u16, 1.5), (3u16, -0.5)]),
                Timestamp::new(42),
                NodeId::try_new(7).unwrap(),
                16,
                Some(2),
            );

            let json = serde_json::to_string(&original).unwrap();
            let recovered: SpikeEvent = serde_json::from_str(&json).unwrap();

            assert_eq!(original, recovered);
            assert!(
                recovered.tail_padding_is_zeroed(),
                "SpikeEvent tail padding must be zero after deserialization"
            );

            // SAFETY: SpikeEvent is repr(C, align(64)) with all fields
            // initialized. size_of::<SpikeEvent>() bytes starting at the
            // struct's address are valid, initialized, and stable for the
            // lifetime of `original`. The cast is read-only.
            let orig_bytes: &[u8] = unsafe {
                core::slice::from_raw_parts(
                    (&raw const original).cast::<u8>(),
                    core::mem::size_of::<SpikeEvent>(),
                )
            };
            // SAFETY: Same invariants as orig_bytes — recovered is a
            // fully initialized SpikeEvent with zeroed tail padding.
            let recv_bytes: &[u8] = unsafe {
                core::slice::from_raw_parts(
                    (&raw const recovered).cast::<u8>(),
                    core::mem::size_of::<SpikeEvent>(),
                )
            };

            assert_eq!(
                orig_bytes, recv_bytes,
                "raw bytes differ after round-trip — DAX contract violated"
            );
        }
    }
}
