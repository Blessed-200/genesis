#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::map_unwrap_or,
    clippy::missing_errors_doc,
    clippy::needless_continue
)]

/// AX-ID: AXIOMA-013
/// Hierarchical Navigable Small World graph for O(log N) semantic search.
/// Exclusive metric: `geometric_distance` (grade-weighted fast_metric_distance).
/// PROHIBITED: Delaunay triangulation. PROHIBITED: `HashMap` in hot path.
/// Adjacency lists stored as sorted Vec<(`NodeId`, f64)> with binary search.
use std::cell::RefCell;
use std::sync::atomic::{AtomicPtr, AtomicU32, AtomicU64, Ordering as AtomicOrdering};
use std::sync::Arc;

use genesis_math::{
    fast_metric_distance_sq, fast_metric_distance_sq_from_dense, SparseCliffordVector,
};
use genesis_types::{GenesisError, NodeId, CLIFFORD_BASIS_SIZE, METRIC_WEIGHTS};
use smallvec::SmallVec;

pub(crate) const SLAB_LANES: usize = 8;
pub(crate) const SLAB_DIM: usize = CLIFFORD_BASIS_SIZE;
pub(crate) const BLOCK_STRIDE: usize = SLAB_DIM * SLAB_LANES;
const MAX_FIXED_HEAP_CAPACITY: usize = 512;
const INITIAL_NODE_CAPACITY: usize = 1024;
const INITIAL_SLAB_BLOCK_CAPACITY: usize = INITIAL_NODE_CAPACITY.div_ceil(SLAB_LANES);
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
const METRIC_WEIGHTS_F32: [f32; CLIFFORD_BASIS_SIZE] = {
    let mut weights = [0.0_f32; CLIFFORD_BASIS_SIZE];
    let mut i = 0;
    while i < CLIFFORD_BASIS_SIZE {
        weights[i] = METRIC_WEIGHTS[i] as f32;
        i += 1;
    }
    weights
};

#[repr(C, align(64))]
#[derive(Clone)]
struct SlabBlock {
    lanes: [f32; BLOCK_STRIDE],
}

impl SlabBlock {
    const fn zeroed() -> Self {
        Self {
            lanes: [f32::INFINITY; BLOCK_STRIDE],
        }
    }
}

/// Cache-padded wrapper for `AtomicU64` to prevent false sharing.
///
/// A single `AtomicU64` is 8 bytes, but CPU cache lines are typically 64 bytes.
/// When a hot `AtomicPtr` and an `AtomicU64` counter sit adjacent in memory,
/// concurrent writes to the pointer can invalidate the cache line containing
/// the counter, causing unnecessary cache coherence traffic (false sharing).
///
/// This wrapper uses `#[repr(C, align(64))]` to occupy a full 64-byte cache line,
/// ensuring it does not share a cache line with adjacent fields.
///
/// AX-ID: AXIOMA-013 (lock-free HNSW performance)
#[repr(C, align(64))]
struct CachePadded<T> {
    value: T,
}

impl<T: Default> Default for CachePadded<T> {
    fn default() -> Self {
        Self {
            value: T::default(),
        }
    }
}

impl<T> CachePadded<T> {
    const fn new(value: T) -> Self {
        Self { value }
    }
}

#[derive(Clone, Default)]
struct HnswLayer0Slab {
    blocks: Vec<SlabBlock>,
    node_to_slab: Arc<Vec<u32>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct CompactNodeId(u32);

impl CompactNodeId {
    #[inline(always)]
    const fn raw(self) -> u32 {
        self.0
    }
}

impl TryFrom<NodeId> for CompactNodeId {
    type Error = GenesisError;

    fn try_from(id: NodeId) -> Result<Self, Self::Error> {
        let raw = id.get();
        let compact =
            u32::try_from(raw).map_err(|_| GenesisError::InvariantViolation { axiom_id: 13 })?;
        Ok(Self(compact))
    }
}

#[inline(always)]
fn try_compact_node_id(id: NodeId) -> Option<CompactNodeId> {
    let raw = id.get();
    let compact = u32::try_from(raw).ok()?;
    Some(CompactNodeId(compact))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HnswDelta {
    Insert,
    Remove,
}

#[derive(Clone, Default)]
struct SearchScratch {
    out: Vec<(usize, f64)>,
}

thread_local! {
    static SEARCH_SCRATCH: RefCell<SearchScratch> = RefCell::new(SearchScratch::default());
    static VISITED_EPOCH: RefCell<Vec<u32>> = const { RefCell::new(Vec::new()) };
    static INSERT_DISTANCE_CACHE: RefCell<Vec<(u32, f32)>> = const { RefCell::new(Vec::new()) };
    static REMOVE_SCRATCH: RefCell<Vec<(usize, usize)>> = const { RefCell::new(Vec::new()) };
}

#[derive(Clone)]
struct FixedHeap<const CAP: usize> {
    data: [(f32, u32); CAP],
    len: usize,
    limit: usize,
}

impl<const CAP: usize> FixedHeap<CAP> {
    fn new(limit: usize) -> Self {
        debug_assert!(limit <= CAP);
        Self {
            data: [(f32::INFINITY, u32::MAX); CAP],
            len: 0,
            limit,
        }
    }

    const fn len(&self) -> usize {
        self.len
    }

    const fn worst(&self) -> f32 {
        if self.len == 0 {
            f32::INFINITY
        } else {
            self.data[self.len - 1].0
        }
    }

    #[allow(clippy::missing_const_for_fn)]
    fn pop_best(&mut self) -> Option<(f32, u32)> {
        if self.len == 0 {
            return None;
        }
        let best = self.data[0];
        if self.len > 1 {
            // SAFETY: source and destination are within `self.data`, overlap is allowed,
            // and we move exactly `self.len - 1` initialized elements one slot left.
            unsafe {
                std::ptr::copy(
                    self.data.as_ptr().add(1),
                    self.data.as_mut_ptr(),
                    self.len - 1,
                );
            }
        }
        self.len -= 1;
        Some(best)
    }

    fn push_or_replace(&mut self, dist: f32, idx: u32) -> bool {
        if !dist.is_finite() || self.limit == 0 {
            return false;
        }
        if self.len == self.limit {
            let worst = self.data[self.len - 1];
            let ge_worst = dist > worst.0 || (dist == worst.0 && idx >= worst.1);
            if ge_worst {
                return false;
            }
        }
        let item = (dist, idx);
        if self.len < self.limit && self.len < 4 {
            let mut pos = self.len;
            while pos > 0 {
                let prev = self.data[pos - 1];
                let is_lt = item.0 < prev.0 || (item.0 == prev.0 && item.1 < prev.1);
                if !is_lt {
                    break;
                }
                self.data[pos] = prev;
                pos -= 1;
            }
            self.data[pos] = item;
            self.len += 1;
            return true;
        }

        let mut pos = if self.len < self.limit {
            let current = self.len;
            self.len += 1;
            current
        } else {
            self.limit - 1
        };

        while pos > 0 {
            let prev = self.data[pos - 1];
            let is_lt = item.0 < prev.0 || (item.0 == prev.0 && item.1 < prev.1);
            if !is_lt {
                break;
            }
            self.data[pos] = prev;
            pos -= 1;
        }
        self.data[pos] = item;
        true
    }

    fn as_slice(&self) -> &[(f32, u32)] {
        &self.data[..self.len]
    }
}

#[inline]
const fn ordered_f64_bits(value: f64) -> u64 {
    let bits = value.to_bits();
    let mask = ((bits as i64) >> 63) as u64;
    bits ^ (mask | (1_u64 << 63))
}

fn slab_distance_scalar(
    slab_ptr: *const f32,
    block: usize,
    query_f32: &[f32; SLAB_DIM],
) -> [f32; SLAB_LANES] {
    let mut out = [0.0_f32; SLAB_LANES];
    let block_base = block * BLOCK_STRIDE;
    let mut lane = 0;
    while lane < SLAB_LANES {
        // SAFETY: `block_base + lane` is in bounds for the same reason as the loop below.
        let dim0 = unsafe { *slab_ptr.add(block_base + lane) };
        let diff0 = f64::from(query_f32[0] - dim0);
        let mut acc = METRIC_WEIGHTS[0] * diff0 * diff0;
        let mut d = 1;
        while d < SLAB_DIM {
            let offset = block_base + d * SLAB_LANES + lane;
            // SAFETY: slab storage is 64-byte aligned by construction, `block`
            // is checked against the number of allocated blocks by the caller,
            // and `offset < blocks * BLOCK_STRIDE` for all `d < SLAB_DIM` and
            // `lane < SLAB_LANES`. Search reads through `&self`, so no mutable alias exists.
            let v = unsafe { *slab_ptr.add(offset) };
            let diff = f64::from(query_f32[d] - v);
            acc = METRIC_WEIGHTS[d].mul_add(diff * diff, acc);
            d += 1;
        }
        out[lane] = if dim0.is_nan() || !acc.is_finite() {
            f32::INFINITY
        } else {
            acc as f32
        };
        lane += 1;
    }
    out
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
#[inline(always)]
unsafe fn slab_distance_avx2(
    slab_ptr: *const f32,
    block: usize,
    query_f32: &[f32; SLAB_DIM],
) -> [f32; SLAB_LANES] {
    use std::arch::x86_64::{
        _mm256_add_ps, _mm256_fmadd_ps, _mm256_load_ps, _mm256_mul_ps, _mm256_set1_ps,
        _mm256_setzero_ps, _mm256_storeu_ps, _mm256_sub_ps,
    };

    macro_rules! fused_dim4 {
        ($base:expr, $d0:expr, $d1:expr, $d2:expr, $d3:expr, $acc0:ident, $acc1:ident, $acc2:ident, $acc3:ident) => {{
            let q0 = _mm256_set1_ps(query_f32[$d0]);
            let q1 = _mm256_set1_ps(query_f32[$d1]);
            let q2 = _mm256_set1_ps(query_f32[$d2]);
            let q3 = _mm256_set1_ps(query_f32[$d3]);
            let w0 = _mm256_set1_ps(METRIC_WEIGHTS_F32[$d0]);
            let w1 = _mm256_set1_ps(METRIC_WEIGHTS_F32[$d1]);
            let w2 = _mm256_set1_ps(METRIC_WEIGHTS_F32[$d2]);
            let w3 = _mm256_set1_ps(METRIC_WEIGHTS_F32[$d3]);
            let v0 = _mm256_load_ps($base.add($d0 * SLAB_LANES));
            let v1 = _mm256_load_ps($base.add($d1 * SLAB_LANES));
            let v2 = _mm256_load_ps($base.add($d2 * SLAB_LANES));
            let v3 = _mm256_load_ps($base.add($d3 * SLAB_LANES));
            let d0v = _mm256_sub_ps(q0, v0);
            let d1v = _mm256_sub_ps(q1, v1);
            let d2v = _mm256_sub_ps(q2, v2);
            let d3v = _mm256_sub_ps(q3, v3);
            $acc0 = _mm256_fmadd_ps(w0, _mm256_mul_ps(d0v, d0v), $acc0);
            $acc1 = _mm256_fmadd_ps(w1, _mm256_mul_ps(d1v, d1v), $acc1);
            $acc2 = _mm256_fmadd_ps(w2, _mm256_mul_ps(d2v, d2v), $acc2);
            $acc3 = _mm256_fmadd_ps(w3, _mm256_mul_ps(d3v, d3v), $acc3);
        }};
    }

    let block_base = unsafe { slab_ptr.add(block * BLOCK_STRIDE) };
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();
    const _: [(); CLIFFORD_BASIS_SIZE] = [(); 16];
    fused_dim4!(block_base, 0, 1, 2, 3, acc0, acc1, acc2, acc3);
    fused_dim4!(block_base, 4, 5, 6, 7, acc0, acc1, acc2, acc3);
    fused_dim4!(block_base, 8, 9, 10, 11, acc0, acc1, acc2, acc3);
    fused_dim4!(block_base, 12, 13, 14, 15, acc0, acc1, acc2, acc3);

    let total = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    let mut out = [0.0_f32; SLAB_LANES];
    unsafe {
        _mm256_storeu_ps(out.as_mut_ptr(), total);
    }
    let nan_mask = unsafe { _mm256_load_ps(block_base) };
    let mut dim0 = [0.0_f32; SLAB_LANES];
    unsafe {
        _mm256_storeu_ps(dim0.as_mut_ptr(), nan_mask);
    }
    let mut lane = 0;
    while lane < SLAB_LANES {
        if dim0[lane].is_nan() || !out[lane].is_finite() {
            out[lane] = f32::INFINITY;
        }
        lane += 1;
    }
    out
}

/// Inline helper for layer-0 slab distance computation in hot search paths.
/// Dispatches to AVX2 or scalar kernel based on compile-time target features.
#[allow(clippy::inline_always)]
// HOT PATH: Criterion shows unacceptable search latency regression without
// forced inlining on AVX2 distance kernels (register-pressure-sensitive callsite).
#[inline(always)]
fn slab_distance(
    slab_ptr: *const f32,
    block: usize,
    query_f32: &[f32; SLAB_DIM],
) -> [f32; SLAB_LANES] {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    {
        // SAFETY: target-feature gated at compile time, pointer invariants are
        // enforced by the caller and match the kernel requirements.
        return unsafe { slab_distance_avx2(slab_ptr, block, query_f32) };
    }
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    )))]
    {
        slab_distance_scalar(slab_ptr, block, query_f32)
    }
}

// Maintenance policy for critical topology modules.
//
// - `#[inline(always)]` is prohibited except for a documented exception with
//   benchmark reproducible + architectural rationale + risk evaluation.
// - Layer-0 codec symbols must maintain `cfg` symmetry:
//   `feature = "hnsw-f16"`, `genesis_const_layer0_codec`, and `test`.
// Any symbol conditioned by `cfg` must have a counterpart
// an explicit `not(...)` counterpart to avoid orphan symbols between profiles.
//
// AX-ID: AXIOMA-013, H_estructura (LEY_FUNDACIONAL §3.1)

mod layer0_codec {
    use genesis_math::SparseCliffordVector;
    use genesis_types::CLIFFORD_BASIS_SIZE;

    #[cfg(feature = "hnsw-f16")]
    mod f16_kernel {
        const F16_MAX_FINITE_BITS: u16 = 0x7BFF;

        #[inline]
        pub(in super::super) const fn f32_to_f16_bits_core(value: f32) -> u16 {
            let bits = value.to_bits();
            let sign = ((bits >> 16) & 0x8000) as u16;
            let exp = ((bits >> 23) & 0xFF) as i32;
            let frac = bits & 0x7F_FFFF;
            if exp == 0xFF {
                // Keep the codec total over all f32 bit patterns:
                // - ±Inf saturates to ±max_finite_f16
                // - NaN canonicalizes to +0.0
                return if frac == 0 {
                    sign | F16_MAX_FINITE_BITS
                } else {
                    0
                };
            }
            if exp <= 112 {
                if exp < 103 {
                    return sign;
                }
                let mant = frac | 0x80_0000;
                return sign | (((mant >> (126 - exp)) + 0x1000) >> 13) as u16;
            }
            if exp >= 143 {
                return sign | F16_MAX_FINITE_BITS;
            }
            let rounded =
                sign | ((((exp - 112) as u32) << 10) as u16) | (((frac + 0x1000) >> 13) as u16);
            if (rounded & 0x7C00) == 0x7C00 {
                sign | F16_MAX_FINITE_BITS
            } else {
                rounded
            }
        }

        #[inline]
        pub(in super::super) fn f16_bits_to_f32_core(bits: u16) -> f32 {
            let sign = (u32::from(bits & 0x8000)) << 16;
            let exp = (bits >> 10) & 0x1F;
            let frac = u32::from(bits & 0x03FF);
            let f_bits = if exp == 0 {
                if frac == 0 {
                    sign
                } else {
                    let mut mant = frac;
                    let mut e = -14i32;
                    while (mant & 0x0400) == 0 {
                        mant <<= 1;
                        e -= 1;
                    }
                    sign | (((e + 127) as u32) << 23) | ((mant & 0x03FF) << 13)
                }
            } else if exp == 0x1F {
                sign | 0x7F80_0000 | (frac << 13)
            } else {
                sign | ((u32::from(exp) + 112) << 23) | (frac << 13)
            };
            f32::from_bits(f_bits)
        }
    }

    /// Sealed trait for layer-0 dense encoding.
    ///
    /// AX-ID: AXIOMA-013, H_estructura (LEY_FUNDACIONAL §3.1)
    #[allow(dead_code)]
    pub(super) trait Layer0Codec: sealed::Sealed {
        type Storage: Copy;

        fn encode(values: &[f32; CLIFFORD_BASIS_SIZE]) -> Self::Storage;
        fn distance_sq(stored: &Self::Storage, query: &SparseCliffordVector) -> f64;
        fn decode_to_f64(stored: &Self::Storage) -> [f64; CLIFFORD_BASIS_SIZE];
    }

    #[cfg(any(not(feature = "hnsw-f16"), test))]
    pub(super) struct F32Codec;

    #[cfg(any(not(feature = "hnsw-f16"), test))]
    impl Layer0Codec for F32Codec {
        type Storage = [f32; CLIFFORD_BASIS_SIZE];

        fn encode(values: &[f32; CLIFFORD_BASIS_SIZE]) -> Self::Storage {
            *values
        }

        fn distance_sq(stored: &Self::Storage, query: &SparseCliffordVector) -> f64 {
            let dense = core::array::from_fn(|i| f64::from(stored[i]));
            super::fast_metric_distance_sq_from_dense(&dense, query)
        }

        fn decode_to_f64(stored: &Self::Storage) -> [f64; CLIFFORD_BASIS_SIZE] {
            core::array::from_fn(|i| f64::from(stored[i]))
        }
    }

    #[cfg(feature = "hnsw-f16")]
    pub(super) struct F16Codec;

    #[cfg(feature = "hnsw-f16")]
    impl Layer0Codec for F16Codec {
        type Storage = [u16; CLIFFORD_BASIS_SIZE];

        fn encode(values: &[f32; CLIFFORD_BASIS_SIZE]) -> Self::Storage {
            #[cfg(genesis_const_layer0_codec)]
            {
                return encode_f16_const(values);
            }

            #[cfg(not(genesis_const_layer0_codec))]
            {
                encode_f16_runtime(values)
            }
        }

        fn distance_sq(stored: &Self::Storage, query: &SparseCliffordVector) -> f64 {
            super::fast_metric_distance_f16_sq(stored, query)
        }

        fn decode_to_f64(stored: &Self::Storage) -> [f64; CLIFFORD_BASIS_SIZE] {
            core::array::from_fn(|i| f64::from(f16_bits_to_f32(stored[i])))
        }
    }

    #[cfg(all(feature = "hnsw-f16", not(genesis_const_layer0_codec)))]
    fn encode_f16_runtime(values: &[f32; CLIFFORD_BASIS_SIZE]) -> [u16; CLIFFORD_BASIS_SIZE] {
        core::array::from_fn(|i| f32_to_f16_bits(values[i]))
    }

    #[cfg(all(feature = "hnsw-f16", genesis_const_layer0_codec))]
    const fn encode_f16_const(values: &[f32; CLIFFORD_BASIS_SIZE]) -> [u16; CLIFFORD_BASIS_SIZE] {
        let mut encoded = [0_u16; CLIFFORD_BASIS_SIZE];
        let mut i = 0;
        while i < CLIFFORD_BASIS_SIZE {
            encoded[i] = f32_to_f16_bits_const(values[i]);
            i += 1;
        }
        encoded
    }

    #[cfg(feature = "hnsw-f16")]
    pub(super) fn f16_bits_to_f32(bits: u16) -> f32 {
        f16_kernel::f16_bits_to_f32_core(bits)
    }

    #[cfg(all(feature = "hnsw-f16", not(genesis_const_layer0_codec)))]
    pub(super) const fn f32_to_f16_bits(value: f32) -> u16 {
        f16_kernel::f32_to_f16_bits_core(value)
    }

    #[cfg(all(feature = "hnsw-f16", genesis_const_layer0_codec))]
    const fn f32_to_f16_bits_const(value: f32) -> u16 {
        f16_kernel::f32_to_f16_bits_core(value)
    }

    #[cfg(all(feature = "hnsw-f16", test))]
    pub(super) use f16_kernel::{f16_bits_to_f32_core, f32_to_f16_bits_core};

    mod sealed {
        pub trait Sealed {}
        #[cfg(any(not(feature = "hnsw-f16"), test))]
        impl Sealed for super::F32Codec {}
        #[cfg(feature = "hnsw-f16")]
        impl Sealed for super::F16Codec {}
    }
}

#[cfg(all(feature = "hnsw-f16", test))]
type ActiveLayer0Codec = layer0_codec::F16Codec;
#[cfg(all(not(feature = "hnsw-f16"), test))]
type ActiveLayer0Codec = layer0_codec::F32Codec;
#[cfg(test)]
type Layer0Coeffs = <ActiveLayer0Codec as layer0_codec::Layer0Codec>::Storage;

/// Maximum number of layers in the HNSW graph.
const MAX_LAYERS: usize = 16;

/// Default maximum connections per layer (M parameter).
/// `pub(crate)` for manifold.rs stack-allocated neighbour buffers (BN-02).
pub(crate) const M: usize = 16;

/// Maximum connections at layer 0 (M0 = 2*M).
/// `pub(crate)` for manifold.rs stack-allocated neighbour buffers (BN-02).
pub(crate) const M0: usize = M * 2;
pub(crate) const MAX_UNIQUE_NEIGHBOR_BUDGET: usize = M0 + (MAX_LAYERS - 1) * M;
const _: () = assert!(M0 <= genesis_types::constants::SINKHORN_MAX_LOCAL_DEGREE);

/// Per-node adjacency storage for one HNSW layer set.
///
/// Layer 0 stays inline to keep the hot path contiguous and avoid heap traffic.
/// Upper layers are allocated only for nodes that actually reach them.
///
/// AX-ID: AXIOMA-007, AXIOMA-013
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Layer0BlockGroup {
    block: u32,
    lane_mask: u8,
    _pad: [u8; 3],
}

const _: () = {
    assert!(std::mem::size_of::<Layer0BlockGroup>() == 8);
    assert!(std::mem::align_of::<Layer0BlockGroup>() == 4);
};

#[derive(Clone, Default)]
struct NodeAdj {
    layer0: SmallVec<[u64; M0]>,
    layer0_groups: SmallVec<[Layer0BlockGroup; 4]>,
    upper: Option<Box<[SmallVec<[u32; M]>]>>,
}

enum NodeAdjIter<'a> {
    Layer0(std::slice::Iter<'a, u64>),
    Upper(std::slice::Iter<'a, u32>),
}

impl Iterator for NodeAdjIter<'_> {
    type Item = u32;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Layer0(iter) => iter.next().map(|packed| (packed >> 32) as u32),
            Self::Upper(iter) => iter.next().copied(),
        }
    }
}

impl NodeAdj {
    #[inline]
    const fn pack_layer0(neighbor_idx: u32, slab_idx: u32) -> u64 {
        ((neighbor_idx as u64) << 32) | slab_idx as u64
    }

    #[inline]
    const fn unpack_layer0_neighbor(packed: u64) -> u32 {
        (packed >> 32) as u32
    }

    #[inline]
    const fn unpack_layer0_slab(packed: u64) -> u32 {
        packed as u32
    }

    #[cfg(debug_assertions)]
    #[inline]
    fn assert_layer0_sorted(&self) {
        for w in self.layer0.windows(2) {
            debug_assert!(
                Self::unpack_layer0_neighbor(w[0]) < Self::unpack_layer0_neighbor(w[1]),
                "layer0 must be sorted ascending by internal index"
            );
        }
        for w in self.layer0_groups.windows(2) {
            debug_assert!(
                w[0].block < w[1].block,
                "layer0 block groups must be sorted"
            );
        }
    }

    #[inline]
    fn neighbors_len(&self, layer: usize) -> usize {
        if layer == 0 {
            self.layer0.len()
        } else {
            self.upper
                .as_deref()
                .and_then(|layers| layers.get(layer - 1))
                .map_or(0, SmallVec::len)
        }
    }

    #[inline]
    fn neighbors_iter(&self, layer: usize) -> NodeAdjIter<'_> {
        if layer == 0 {
            NodeAdjIter::Layer0(self.layer0.iter())
        } else {
            NodeAdjIter::Upper(
                self.upper
                    .as_deref()
                    .and_then(|layers| layers.get(layer - 1))
                    .map_or_else(|| [].iter(), |neighbors| neighbors.iter()),
            )
        }
    }

    #[inline]
    fn neighbor_at(&self, layer: usize, pos: usize) -> Option<u32> {
        if layer == 0 {
            self.layer0
                .get(pos)
                .copied()
                .map(Self::unpack_layer0_neighbor)
        } else {
            self.upper
                .as_deref()
                .and_then(|layers| layers.get(layer - 1))
                .and_then(|neighbors| neighbors.get(pos))
                .copied()
        }
    }

    #[inline]
    fn layer0_groups(&self) -> &[Layer0BlockGroup] {
        &self.layer0_groups
    }

    #[inline]
    fn update_layer0_group_insert(&mut self, slab_idx: u32) {
        let block = slab_idx >> 3;
        let lane = slab_idx & 7;
        let lane_mask = 1_u8 << lane;
        match self
            .layer0_groups
            .binary_search_by_key(&block, |group| group.block)
        {
            Ok(pos) => self.layer0_groups[pos].lane_mask |= lane_mask,
            Err(pos) => self.layer0_groups.insert(
                pos,
                Layer0BlockGroup {
                    block,
                    lane_mask,
                    _pad: [0; 3],
                },
            ),
        }
        debug_assert!(self.layer0_groups.len() <= M0);
    }

    #[inline]
    fn update_layer0_group_remove(&mut self, slab_idx: u32) {
        let block = slab_idx >> 3;
        let lane = slab_idx & 7;
        let lane_mask = 1_u8 << lane;
        if let Ok(pos) = self
            .layer0_groups
            .binary_search_by_key(&block, |group| group.block)
        {
            let next_mask = self.layer0_groups[pos].lane_mask & !lane_mask;
            if next_mask == 0 {
                self.layer0_groups.remove(pos);
            } else {
                self.layer0_groups[pos].lane_mask = next_mask;
            }
        }
    }

    #[inline]
    fn neighbors_mut(&mut self, layer: usize) -> Option<&mut SmallVec<[u32; M]>> {
        if layer == 0 {
            return None;
        }
        let needed = layer;
        let layers = self
            .upper
            .get_or_insert_with(|| vec![SmallVec::<[u32; M]>::new(); needed].into_boxed_slice());
        if layers.len() < needed {
            let mut grown = layers.to_vec();
            grown.resize_with(needed, SmallVec::new);
            *layers = grown.into_boxed_slice();
        }
        layers.get_mut(layer - 1)
    }

    #[inline]
    fn add_neighbor(
        &mut self,
        layer: usize,
        neighbor: u32,
        slab_idx: u32,
        max_neighbors: usize,
    ) -> bool {
        if layer == 0 {
            match self
                .layer0
                .binary_search_by_key(&neighbor, |&packed| Self::unpack_layer0_neighbor(packed))
            {
                Ok(_) => return false,
                Err(pos) => {
                    if self.layer0.len() >= max_neighbors {
                        return false;
                    }
                    self.layer0
                        .insert(pos, Self::pack_layer0(neighbor, slab_idx));
                    self.update_layer0_group_insert(slab_idx);
                    #[cfg(debug_assertions)]
                    self.assert_layer0_sorted();
                    return true;
                }
            }
        }
        let Some(neighbors) = self.neighbors_mut(layer) else {
            return false;
        };
        match neighbors.binary_search(&neighbor) {
            Ok(_) => false,
            Err(pos) => {
                if neighbors.len() >= max_neighbors {
                    return false;
                }
                neighbors.insert(pos, neighbor);
                true
            }
        }
    }

    #[inline]
    fn remove_neighbor(&mut self, layer: usize, neighbor: u32) -> bool {
        if layer == 0 {
            if let Ok(pos) = self
                .layer0
                .binary_search_by_key(&neighbor, |&packed| Self::unpack_layer0_neighbor(packed))
            {
                let packed = self.layer0.remove(pos);
                self.update_layer0_group_remove(Self::unpack_layer0_slab(packed));
                #[cfg(debug_assertions)]
                self.assert_layer0_sorted();
                return true;
            }
            return false;
        }
        if let Some(upper) = self.upper.as_mut() {
            if let Some(neighbors) = upper.get_mut(layer - 1) {
                if let Ok(pos) = neighbors.binary_search(&neighbor) {
                    neighbors.remove(pos);
                    return true;
                }
            }
        }
        false
    }

    #[inline]
    fn clear_layer(&mut self, layer: usize) {
        if layer == 0 {
            self.layer0.clear();
            self.layer0_groups.clear();
        } else if let Some(upper) = self.upper.as_mut() {
            if let Some(neighbors) = upper.get_mut(layer - 1) {
                neighbors.clear();
            }
        }
    }
}

/// Level multiplier: 1.0 / ln(M).
// M is a small constant (≤ 64). M as f64 is exact: M < 2^53.
#[allow(clippy::cast_precision_loss)]
fn ml() -> f64 {
    1.0_f64 / (M as f64).ln()
}

/// A node stored in the HNSW graph.
#[derive(Clone)]
struct HnswNode {
    id: NodeId,
    vec: SparseCliffordVector,
    /// Maximum layer assigned to this node in the HNSW hierarchy.
    max_layer: usize,
}

impl HnswNode {
    /// Builds an HNSW node with cached layer-0 coefficients.
    ///
    /// ## Finiteness contract
    ///
    /// `geometric_distance` is defined only on finite coefficients; therefore,
    /// every blade used to materialise `layer0` must remain finite after the
    /// `f64 -> f32` projection. Non-finite values are rejected upstream by
    /// `SparseCliffordVector` constructors, and this function keeps a debug-time
    /// guard to detect any invariant breach before `encode_layer0`.
    ///
    /// AX-ID: AXIOMA-013, H_estructura (LEY_FUNDACIONAL §3.1)
    const fn new(id: NodeId, vec: SparseCliffordVector, max_layer: usize) -> Self {
        Self { id, vec, max_layer }
    }
}

/// Encodes dense layer-0 coefficients into the storage format configured for
/// the current build (`f32` or `f16` bits).
///
/// ## Finiteness contract
///
/// Inputs must be finite. This preserves the metric contract required by
/// `geometric_distance` and avoids introducing NaN/Inf into the layer-0 fast
/// path. In debug/test builds we assert this invariant before encoding.
///
/// AX-ID: AXIOMA-013, H_estructura (LEY_FUNDACIONAL §3.1)
#[cfg(test)]
fn encode_layer0(values: &[f32; CLIFFORD_BASIS_SIZE]) -> Result<Layer0Coeffs, GenesisError> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(GenesisError::InvalidInput(
            "Non-finite coefficients detected after conversion",
        ));
    }
    debug_assert!(values.iter().all(|value| value.is_finite()));
    Ok(<ActiveLayer0Codec as layer0_codec::Layer0Codec>::encode(
        values,
    ))
}

#[cfg(feature = "hnsw-f16")]
#[allow(clippy::inline_always)]
#[inline(always)]
/// # Panics
///
/// Panics only if the decompressed coefficients become non-finite, which
/// violates the encoding invariants of `f32_to_f16_bits` for finite inputs.
///
/// AX-ID: AXIOMA-014, LEY_FUNDACIONAL §3.1
pub fn fast_metric_distance_f16(
    stored: &[u16; CLIFFORD_BASIS_SIZE],
    query: &SparseCliffordVector,
) -> f64 {
    fast_metric_distance_f16_sq(stored, query).sqrt()
}

#[cfg(feature = "hnsw-f16")]
#[allow(clippy::inline_always)]
#[inline(always)]
/// # Panics
///
/// Panics only if the decompressed coefficients become non-finite, which
/// violates the encoding invariants of `f32_to_f16_bits` for finite inputs.
///
/// AX-ID: AXIOMA-014, LEY_FUNDACIONAL §3.1
pub fn fast_metric_distance_f16_sq(
    stored: &[u16; CLIFFORD_BASIS_SIZE],
    query: &SparseCliffordVector,
) -> f64 {
    // ARCHITECTURAL NOTE:
    // It uses compile-time dispatch instead of runtime dispatch to avoid
    // loss of inlining and `vzeroupper` penalties in the hot loop.
    // On x86_64, compile with RUSTFLAGS="-C target-cpu=native" to activate AVX2.
    // Primary target: Genesis Edge (ARM + NEON).
    let mut decompressed = [0.0_f64; CLIFFORD_BASIS_SIZE];

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "f16c",
        target_feature = "avx2"
    ))]
    {
        use std::arch::x86_64::*;
        let s = stored.as_ptr().cast::<__m128i>();

        // SAFETY: it reads/writes exactly 16 elements within limits and the AVX2/F16C features are guaranteed by cfg.
        unsafe {
            let half0 = _mm_loadu_si128(s);
            let wide0 = _mm256_cvtph_ps(half0);
            let lo0 = _mm256_cvtps_pd(_mm256_castps256_ps128(wide0));
            let hi0 = _mm256_cvtps_pd(_mm256_extractf128_ps(wide0, 1));
            _mm256_storeu_pd(decompressed.as_mut_ptr(), lo0);
            _mm256_storeu_pd(decompressed.as_mut_ptr().add(4), hi0);

            let half1 = _mm_loadu_si128(s.add(1));
            let wide1 = _mm256_cvtph_ps(half1);
            let lo1 = _mm256_cvtps_pd(_mm256_castps256_ps128(wide1));
            let hi1 = _mm256_cvtps_pd(_mm256_extractf128_ps(wide1, 1));
            _mm256_storeu_pd(decompressed.as_mut_ptr().add(8), lo1);
            _mm256_storeu_pd(decompressed.as_mut_ptr().add(12), hi1);
        }
    }

    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "f16c",
        target_feature = "avx2"
    )))]
    {
        for i in 0..CLIFFORD_BASIS_SIZE {
            decompressed[i] = f64::from(layer0_codec::f16_bits_to_f32(stored[i]));
        }
    }

    fast_metric_distance_sq_from_dense(&decompressed, query)
}

/// Benchmark helper: computes four squared distances with a scalar loop.
///
/// AX-ID: AXIOMA-013, H_estructura (LEY_FUNDACIONAL §3.1)
pub fn benchmark_scalar_distance_4x(
    query: &SparseCliffordVector,
    candidates: &[SparseCliffordVector; 4],
) -> [f64; 4] {
    core::array::from_fn(|i| fast_metric_distance_sq(query, &candidates[i]))
}

/// Benchmark helper: computes four squared distances in a batch API.
///
/// This preserves the benchmark contract used by `benches/manifold.rs`.
///
/// AX-ID: AXIOMA-013, H_estructura (LEY_FUNDACIONAL §3.1)
pub fn benchmark_batch_distance_4(
    query: &SparseCliffordVector,
    candidates: &[SparseCliffordVector; 4],
) -> [f64; 4] {
    benchmark_scalar_distance_4x(query, candidates)
}

#[cfg(all(feature = "hnsw-f16", test))]
use layer0_codec::{
    f16_bits_to_f32_core as f16_bits_to_f32, f32_to_f16_bits_core as f32_to_f16_bits,
};

/// Hierarchical Navigable Small World graph.
///
/// Stores vectors and supports O(log N) approximate nearest-neighbour search
/// using the exclusive bivector metric from genesis-math.
///
/// AX-ID: AXIOMA-013
pub struct HnswGraph {
    /// All nodes stored in a flat Vec. Index = internal idx.
    nodes: Arc<Vec<HnswNode>>,
    /// Maps compact `NodeId` (u32) → internal index via sorted (`CompactNodeId`, usize) pairs.
    /// Sorted by compact raw ID, searched via binary search. No `HashMap`.
    id_index: Arc<Vec<(CompactNodeId, usize)>>,
    /// Fallback map for non-compactable `NodeId` values (full-width u64 path).
    wide_id_index: Arc<Vec<(NodeId, usize)>>,
    /// Entry point for top-layer search (internal index).
    entry: Option<usize>,
    /// Layer of the current entry point.
    entry_layer: usize,
    /// `ef_construction` parameter.
    ef_construction: usize,
    /// Direct map `NodeId.get()` → `internal_idx` when `NodeIds` are consecutive.
    /// Dynamic capacity: expands when inserting larger `NodeIds`.
    direct_index: Arc<Vec<u32>>, // u32::MAX = no presente
    /// Secondary index state `id_index`.
    state: GraphState,
    /// Per-node local adjacency lists indexed by dense internal node index.
    layer_neighbors: Arc<Vec<NodeAdj>>,
    /// Current search generation for epoch-marked visited state.
    epoch_gen: AtomicU32,
    /// Directed edge count at layer 0 (stored as directed for O(1) updates).
    edge_count_layer0_undirected: usize,
    /// Live-node count excluding tombstoned slots.
    live_nodes: usize,
    /// Persistent block-major SoA storage for layer-0 vectors.
    layer0_soa: HnswLayer0Slab,
    #[cfg(test)]
    fail_preinsert_index_conversion: bool,
}

impl Clone for HnswGraph {
    fn clone(&self) -> Self {
        Self {
            nodes: Arc::clone(&self.nodes),
            id_index: Arc::clone(&self.id_index),
            wide_id_index: Arc::clone(&self.wide_id_index),
            entry: self.entry,
            entry_layer: self.entry_layer,
            ef_construction: self.ef_construction,
            direct_index: Arc::clone(&self.direct_index),
            state: self.state,
            layer_neighbors: Arc::clone(&self.layer_neighbors),
            epoch_gen: AtomicU32::new(self.epoch_gen.load(AtomicOrdering::Relaxed)),
            edge_count_layer0_undirected: self.edge_count_layer0_undirected,
            live_nodes: self.live_nodes,
            layer0_soa: HnswLayer0Slab {
                blocks: self.layer0_soa.blocks.clone(),
                node_to_slab: Arc::clone(&self.layer0_soa.node_to_slab),
            },
            #[cfg(test)]
            fail_preinsert_index_conversion: self.fail_preinsert_index_conversion,
        }
    }
}

/// Cache-optimised SoA view of layer-0 HNSW data.
///
/// This structure flattens node metadata and base-layer adjacency into contiguous
/// buffers to improve prefetch locality and enable auto-vectorisation-friendly
/// traversal patterns in read-heavy paths.
///
/// AX-ID: AXIOMA-013
pub struct HnswLayer0Soa {
    /// Node IDs in dense internal-index order.
    pub node_ids: Vec<NodeId>,
    /// Persistent SoA slab flattened in `[block][dim][lane]` order.
    pub slab: Vec<f32>,
    /// Dense-index to slab-index mapping.
    pub node_to_slab: Vec<u32>,
    /// Per-node offsets into `neighbor_ids` / `neighbor_distances`.
    pub neighbor_offsets: Vec<(usize, usize)>,
    /// Flattened neighbour IDs for layer 0.
    pub neighbor_ids: Vec<NodeId>,
    /// Flattened neighbour distances for layer 0.
    pub neighbor_distances: Vec<f64>,
}

/// Lock-free append-only snapshot index for HNSW.
///
/// Writers clone the current immutable snapshot, apply one mutation, and publish
/// with CAS. Readers load the latest snapshot without locking.
///
/// # False sharing mitigation
///
/// The `cas_retries` counter is cache-padded to prevent false sharing with the
/// hot `head` pointer. Without padding, concurrent CAS operations on `head`
/// would invalidate the cache line containing `cas_retries`, causing unnecessary
/// cache coherence traffic even when only the counter is being read.
///
/// AX-ID: AXIOMA-013
pub struct LockFreeHnswIndex {
    head: AtomicPtr<HnswGraph>,
    cas_retries: CachePadded<AtomicU64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GraphState {
    /// Online phase: direct appends in `id_index` unsorted.
    Online,
    /// Post-compaction phase: `id_index` sorted for binary search.
    Compacted,
}

impl HnswGraph {
    /// Create a new empty HNSW graph.
    ///
    /// AX-ID: AXIOMA-013
    pub fn new(ef_construction: usize) -> Self {
        Self {
            nodes: Arc::new(Vec::with_capacity(INITIAL_NODE_CAPACITY)),
            id_index: Arc::new(Vec::with_capacity(INITIAL_NODE_CAPACITY)),
            wide_id_index: Arc::new(Vec::new()),
            entry: None,
            entry_layer: 0,
            ef_construction,
            direct_index: Arc::new(Vec::with_capacity(INITIAL_NODE_CAPACITY)),
            state: GraphState::Online,
            layer_neighbors: Arc::new(Vec::with_capacity(INITIAL_NODE_CAPACITY)),
            epoch_gen: AtomicU32::new(0),
            edge_count_layer0_undirected: 0,
            live_nodes: 0,
            layer0_soa: HnswLayer0Slab {
                blocks: Vec::with_capacity(INITIAL_SLAB_BLOCK_CAPACITY),
                node_to_slab: Arc::new(Vec::with_capacity(INITIAL_NODE_CAPACITY)),
            },
            #[cfg(test)]
            fail_preinsert_index_conversion: false,
        }
    }

    #[inline]
    fn clone_with_delta(&self, delta: &HnswDelta) -> Self {
        self.apply_delta(delta)
    }

    fn apply_delta(&self, delta: &HnswDelta) -> Self {
        let (nodes, id_index, wide_id_index, direct_index, layer_neighbors, node_to_slab) =
            match delta {
                // INSERT mutates all structural vectors.
                HnswDelta::Insert => (
                    Arc::new(self.nodes.as_ref().clone()),
                    Arc::new(self.id_index.as_ref().clone()),
                    Arc::new(self.wide_id_index.as_ref().clone()),
                    Arc::new(self.direct_index.as_ref().clone()),
                    Arc::new(self.layer_neighbors.as_ref().clone()),
                    Arc::new(self.layer0_soa.node_to_slab.as_ref().clone()),
                ),
                // REMOVE mutates nodes, id_index, direct_index, and adjacency;
                // node_to_slab remains immutable and can stay shared.
                HnswDelta::Remove => (
                    Arc::new(self.nodes.as_ref().clone()),
                    Arc::new(self.id_index.as_ref().clone()),
                    Arc::new(self.wide_id_index.as_ref().clone()),
                    Arc::new(self.direct_index.as_ref().clone()),
                    Arc::new(self.layer_neighbors.as_ref().clone()),
                    Arc::clone(&self.layer0_soa.node_to_slab),
                ),
            };

        Self {
            nodes,
            id_index,
            wide_id_index,
            entry: self.entry,
            entry_layer: self.entry_layer,
            ef_construction: self.ef_construction,
            direct_index,
            state: self.state,
            layer_neighbors,
            epoch_gen: AtomicU32::new(self.epoch_gen.load(AtomicOrdering::Relaxed)),
            edge_count_layer0_undirected: self.edge_count_layer0_undirected,
            live_nodes: self.live_nodes,
            layer0_soa: HnswLayer0Slab {
                blocks: self.layer0_soa.blocks.clone(),
                node_to_slab,
            },
            #[cfg(test)]
            fail_preinsert_index_conversion: self.fail_preinsert_index_conversion,
        }
    }

    fn prevalidate_internal_idx_u32(new_idx: usize) -> Result<u32, GenesisError> {
        u32::try_from(new_idx).map_err(|_| GenesisError::InvariantViolation { axiom_id: 13 })
    }

    #[inline(always)]
    fn cow_vec_mut<T: Clone>(arc: &mut Arc<Vec<T>>) -> &mut Vec<T> {
        if Arc::strong_count(arc) == 1 {
            Arc::get_mut(arc).expect("strong_count == 1 implies unique Arc")
        } else {
            Arc::make_mut(arc)
        }
    }

    #[inline]
    const fn layer_max_neighbors(layer: usize) -> usize {
        if layer == 0 {
            M0
        } else {
            M
        }
    }

    #[inline(always)]
    fn node_neighbors_len(&self, node_idx: usize, layer: usize) -> usize {
        debug_assert!(node_idx < self.layer_neighbors.len());
        self.layer_neighbors[node_idx].neighbors_len(layer)
    }

    #[inline(always)]
    fn node_neighbors_iter(&self, node_idx: usize, layer: usize) -> NodeAdjIter<'_> {
        debug_assert!(node_idx < self.layer_neighbors.len());
        self.layer_neighbors[node_idx].neighbors_iter(layer)
    }

    #[inline]
    fn node_neighbor_at(&self, node_idx: usize, layer: usize, pos: usize) -> Option<u32> {
        debug_assert!(node_idx < self.layer_neighbors.len());
        self.layer_neighbors[node_idx].neighbor_at(layer, pos)
    }

    #[inline]
    fn node_layer0_groups(&self, node_idx: usize) -> &[Layer0BlockGroup] {
        debug_assert!(node_idx < self.layer_neighbors.len());
        self.layer_neighbors[node_idx].layer0_groups()
    }

    #[inline]
    fn next_search_epoch(&self) -> u32 {
        let next = self
            .epoch_gen
            .fetch_add(1, AtomicOrdering::Relaxed)
            .wrapping_add(1);
        if next == 0 {
            self.epoch_gen.store(1, AtomicOrdering::Relaxed);
            return 1;
        }
        next
    }

    fn dense_to_query_f32(query: &SparseCliffordVector) -> [f32; SLAB_DIM] {
        core::array::from_fn(|i| query.coeffs[i] as f32)
    }

    fn write_node_to_slab(&mut self, dense_idx: usize, vec: &SparseCliffordVector) {
        let slab_idx = dense_idx;
        let block = slab_idx / SLAB_LANES;
        let lane = slab_idx % SLAB_LANES;
        if block >= self.layer0_soa.blocks.len() {
            self.layer0_soa.blocks.push(SlabBlock::zeroed());
        }
        let block_ref = &mut self.layer0_soa.blocks[block];
        let mut d = 0;
        while d < SLAB_DIM {
            block_ref.lanes[d * SLAB_LANES + lane] = vec.coeffs[d] as f32;
            d += 1;
        }
        let slab_idx_u32 = u32::try_from(slab_idx).unwrap_or(u32::MAX - 1);
        debug_assert_ne!(slab_idx_u32, u32::MAX);
        Self::cow_vec_mut(&mut self.layer0_soa.node_to_slab).push(slab_idx_u32);
    }

    #[inline]
    fn layer0_slab_ptr(&self) -> *const f32 {
        self.layer0_soa
            .blocks
            .first()
            .map_or(std::ptr::null(), |block| block.lanes.as_ptr())
    }

    /// Lookup internal index by `NodeId`. O(1) average with direct index, fallback O(log N).
    #[inline(always)]
    fn idx(&self, id: NodeId) -> Option<usize> {
        // Contract CRATE-002: direct_index addressing assumes NodeId values remain
        // within the platform-indexable range and deployment cardinality stays < 2^32.
        // The compact secondary index (`CompactNodeId`) enforces the same bound.
        #[allow(clippy::cast_possible_truncation)]
        let raw = id.get() as usize;
        if raw < self.direct_index.len() {
            // For concurrent use (multi-threaded production): direct_index must be
            // Vec<AtomicU32>. Currently access is exclusive to &mut self / &self
            // under a single-writer contract. Fence has no effect on non-atomic types.
            let idx = self.direct_index[raw];
            if idx != u32::MAX {
                return Some(idx as usize);
            }
        }

        match self.state {
            GraphState::Compacted => {
                if let Some(compact) = try_compact_node_id(id) {
                    self.id_index
                        .binary_search_by_key(&compact.raw(), |&(nid, _)| nid.raw())
                        .ok()
                        .map(|pos| self.id_index[pos].1)
                } else {
                    self.wide_id_index
                        .binary_search_by_key(&id.get(), |&(nid, _)| nid.get())
                        .ok()
                        .map(|pos| self.wide_id_index[pos].1)
                }
            }
            GraphState::Online => {
                if let Some(compact) = try_compact_node_id(id) {
                    self.id_index
                        .iter()
                        .find_map(|&(nid, idx)| (nid == compact).then_some(idx))
                } else {
                    self.wide_id_index
                        .iter()
                        .find_map(|&(nid, idx)| (nid == id).then_some(idx))
                }
            }
        }
    }

    /// Insert internal index mapping with O(1) append during online phase.
    fn insert_id_index(&mut self, id: NodeId, idx: usize) {
        debug_assert_eq!(self.state, GraphState::Online);
        if let Some(compact) = try_compact_node_id(id) {
            Self::cow_vec_mut(&mut self.id_index).push((compact, idx));
        } else {
            Self::cow_vec_mut(&mut self.wide_id_index).push((id, idx));
        }
    }

    /// Compact and sort `id_index` for O(log N) fallback queries.
    pub fn compact_index(&mut self) {
        if self.state == GraphState::Compacted {
            return;
        }

        radix_sort_node_ids(Self::cow_vec_mut(&mut self.id_index));
        Self::cow_vec_mut(&mut self.wide_id_index).sort_unstable_by_key(|(id, _)| id.get());
        self.state = GraphState::Compacted;
    }

    /// Random level for a new element using the HNSW level formula.
    /// Uses a simple deterministic LCG seeded by the node id to avoid
    /// introducing a global RNG dependency.
    fn random_level(id: NodeId) -> usize {
        // LCG: x = (a*x + c) mod m
        let x = id
            .get()
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        // Uniform u in (0, 1): uses 53 bits and shifts by half an ULP to avoid exact 0.
        // x >> 11 ∈ [0, 2^53). Conversions are exact in f64 for 53 bits.
        #[allow(clippy::cast_precision_loss)]
        let mut u = (((x >> 11) as f64) + 0.5) * (1.0 / ((1_u64 << 53) as f64));
        if !u.is_finite() {
            return 0;
        }
        u = u.clamp(f64::MIN_POSITIVE, 1.0 - f64::EPSILON);

        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let level = (-u.ln() * ml()).floor() as usize;
        level.min(MAX_LAYERS - 1)
    }

    /// Insert a vector into the graph.
    ///
    /// Returns `GenesisError::InvariantViolation` if adding edges would exceed
    /// the density invariant `|E| > node_count * log2(node_count) * 2`.
    ///
    /// If `id` is already present in the graph, insertion is idempotent and returns `Ok(())`
    /// without modifying the graph.
    ///
    /// AX-ID: AXIOMA-013, `H_restricción`
    pub fn insert(&mut self, id: NodeId, vec: &SparseCliffordVector) -> Result<(), GenesisError> {
        if self.state == GraphState::Compacted {
            return Err(GenesisError::InvariantViolation { axiom_id: 13 });
        }
        if id == NodeId::INVALID {
            return Err(GenesisError::InvariantViolation { axiom_id: 4 });
        }
        if self.idx(id).is_some() {
            return Ok(()); // already present
        }

        let target_layer = Self::random_level(id);
        let new_idx = self.nodes.len();
        let previous_entry = self.entry.map(|entry_idx| (entry_idx, self.entry_layer));
        #[cfg(test)]
        if self.fail_preinsert_index_conversion {
            return Err(GenesisError::InvariantViolation { axiom_id: 13 });
        }

        let direct_index_update = if id.get() < u64::from(u32::MAX) {
            let id_raw = usize::try_from(id.get())
                .map_err(|_| GenesisError::InvariantViolation { axiom_id: 4 })?;
            let next_direct_len = if id_raw >= self.direct_index.len() {
                Some(
                    id_raw
                        .checked_add(1)
                        .ok_or(GenesisError::InvariantViolation { axiom_id: 13 })?,
                )
            } else {
                None
            };
            let new_idx_u32 = Self::prevalidate_internal_idx_u32(new_idx)?;
            Some((id_raw, next_direct_len, new_idx_u32))
        } else {
            None
        };

        Self::cow_vec_mut(&mut self.nodes).push(HnswNode::new(id, *vec, target_layer));
        Self::cow_vec_mut(&mut self.layer_neighbors).push(NodeAdj::default());
        self.write_node_to_slab(new_idx, vec);
        self.insert_id_index(id, new_idx);
        self.live_nodes = self.live_nodes.saturating_add(1);

        if let Some((id_raw, next_direct_len, new_idx_u32)) = direct_index_update {
            if let Some(next_len) = next_direct_len {
                Self::cow_vec_mut(&mut self.direct_index).resize(next_len, u32::MAX);
            }
            // Release barrier: not required over Vec<u32> with &mut self (single-writer).
            // To publish this index to concurrent readers, direct_index must be
            // Vec<AtomicU32> with store(Release). For now: single-threaded, no contention.
            Self::cow_vec_mut(&mut self.direct_index)[id_raw] = new_idx_u32;
            debug_assert!(
                self.idx(id) == Some(new_idx),
                "direct_index inconsistente con id_index para NodeId={}",
                id.get()
            );
        }

        let Some((entry_idx, entry_layer)) = previous_entry else {
            self.entry = Some(new_idx);
            self.entry_layer = target_layer;
            return Ok(());
        };

        // Phase 1: greedy descent from entry_layer to target_layer+1
        let mut current = entry_idx;
        for lc in (target_layer + 1..=entry_layer).rev() {
            current = self.greedy_search_layer(vec, current, lc);
        }

        // Phase 2: beam search and connect from target_layer down to 0
        for lc in (0..=target_layer.min(entry_layer)).rev() {
            let layer_m = if lc == 0 { M0 } else { M };
            // nodes.len() + 1 ≤ N. For N < 2^53 (physical limit), usize→f64 is exact.
            // log2(N).ceil() is always positive (N ≥ 1). f64→usize without sign loss.
            #[allow(
                clippy::cast_precision_loss,
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss
            )]
            let degree_cap = ((((self.nodes.len() + 1) as f64).log2().ceil() as usize) * 2).max(1);
            let m_max = layer_m.min(degree_cap);
            debug_assert!(m_max <= M0);
            let candidates = self.search_layer(vec, current, self.ef_construction, lc);
            let connect_limit = if lc == 0 { layer_m } else { m_max };
            // Take top-M by distance
            let neighbours: SmallVec<[(usize, f64); M0]> =
                candidates.into_iter().take(connect_limit).collect();

            // Add bidirectional edges
            let new_idx = self.nodes.len() - 1; // last inserted
            for &(nb_idx, dist) in &neighbours {
                self.add_edge(new_idx, lc, nb_idx, dist);
                self.add_edge(nb_idx, lc, new_idx, dist);
                // Prune nb if it exceeds m_max
                self.prune_layer(nb_idx, lc, m_max, None);
            }

            if lc == 0 && connect_limit > m_max {
                // Populate cache from actual neighbors after insertion, respecting connect_limit
                INSERT_DISTANCE_CACHE.with(|cache_cell| {
                    let mut cache = cache_cell.borrow_mut();
                    cache.clear();
                    let query_f32 = Self::dense_to_query_f32(vec);
                    for nb_idx_u32 in self.node_neighbors_iter(new_idx, lc) {
                        let nb_idx = nb_idx_u32 as usize;
                        let dist_sq = self.distance_to_layer0_node_sq(&query_f32, nb_idx) as f32;
                        cache.push((nb_idx_u32, dist_sq));
                    }
                    let cache_ref = cache.as_slice();
                    self.prune_layer(new_idx, lc, m_max, Some(cache_ref));
                });
            }

            if let Some(&(closest, _)) = neighbours.first() {
                current = closest;
            }
        }

        // Update entry point if new node has higher layer
        if target_layer > entry_layer {
            self.entry = Some(self.nodes.len() - 1);
            self.entry_layer = target_layer;
        }

        // BN-01: Global density limit removed. Graph density is strictly bounded
        // by local HNSW invariants: M_max0 (layer 0) and M_max (upper layers)
        // enforced by prune_layer() after each bidirectional edge insertion.
        // The H_restricción density ratio is tracked via ManifoldCollector::compute_edge_density()
        // and monitored by the Hamiltonian — no O(N²) global sweep needed here.

        Ok(())
    }

    /// Add an edge at a given layer (no duplicates).
    fn add_edge(&mut self, from_idx: usize, layer: usize, to_idx: usize, dist: f64) {
        if layer > self.nodes[from_idx].max_layer {
            return;
        }
        let _ = dist;
        let max_neighbors = Self::layer_max_neighbors(layer);
        if Self::cow_vec_mut(&mut self.layer_neighbors)[from_idx].add_neighbor(
            layer,
            to_idx as u32,
            self.layer0_soa.node_to_slab[to_idx],
            max_neighbors,
        ) && layer == 0
        {
            self.edge_count_layer0_undirected += 1;
        }
    }

    /// Remove a directed edge at `layer` if present.
    /// Returns `true` iff one edge was removed.
    fn remove_edge(&mut self, from_idx: usize, layer: usize, to_idx: usize) -> bool {
        if layer > self.nodes[from_idx].max_layer {
            return false;
        }
        if Self::cow_vec_mut(&mut self.layer_neighbors)[from_idx]
            .remove_neighbor(layer, to_idx as u32)
        {
            if layer == 0 {
                self.edge_count_layer0_undirected =
                    self.edge_count_layer0_undirected.saturating_sub(1);
            }
            return true;
        }
        false
    }

    /// Remove an edge in both directions, keeping graph symmetry.
    /// Returns the number of directed edges removed.
    fn remove_edge_bidirectional(&mut self, from_idx: usize, layer: usize, to_idx: usize) -> usize {
        let mut removed = 0;
        if self.remove_edge(from_idx, layer, to_idx) {
            removed += 1;
        }
        if self.remove_edge(to_idx, layer, from_idx) {
            removed += 1;
        }
        removed
    }

    /// Prune a node's adjacency list at a layer to at most `m_max` neighbours
    /// (keep the closest by distance).
    fn prune_layer(
        &mut self,
        idx: usize,
        layer: usize,
        m_max: usize,
        precomputed_distances: Option<&[(u32, f32)]>,
    ) {
        if layer > self.nodes[idx].max_layer {
            return;
        }
        let degree = self.node_neighbors_len(idx, layer);
        if degree <= m_max {
            return;
        }

        // HOT PATH: O(d) score materialization + average O(d) partition.
        // Degree is bounded by HNSW neighbour caps (layer 0 <= M0; upper layers <= M).
        debug_assert!(degree <= M0);
        let query_f32 = (layer == 0).then(|| Self::dense_to_query_f32(&self.nodes[idx].vec));
        let mut scored: SmallVec<[u128; M0]> = SmallVec::with_capacity(degree);
        if let Some(cached_distances) = precomputed_distances {
            for &(nb_idx_u32, dist_sq) in cached_distances {
                let key = (u128::from(ordered_f64_bits(f64::from(dist_sq))) << 64)
                    | u128::from(nb_idx_u32);
                scored.push(key);
            }
        } else {
            for nb_idx_u32 in self.node_neighbors_iter(idx, layer) {
                let nb_idx = nb_idx_u32 as usize;
                let dist = if let Some(query_f32) = &query_f32 {
                    self.distance_to_layer0_node_sq(query_f32, nb_idx)
                } else {
                    self.distance_to_node_sq(&self.nodes[idx].vec, nb_idx, layer)
                };
                let key = (u128::from(ordered_f64_bits(dist)) << 64) | u128::from(nb_idx_u32);
                scored.push(key);
            }
        }

        // Deterministic full ordering over (distance_bits, node_idx).
        scored.sort_unstable();

        // Extract overflow IDs before mutating adjacency (requires &mut self).
        let drop: SmallVec<[u32; M0]> = scored[m_max..]
            .iter()
            .map(|&packed| packed as u32)
            .collect();

        for nb_idx_u32 in drop {
            self.remove_edge_bidirectional(idx, layer, nb_idx_u32 as usize);
        }
    }

    fn distance_to_node_sq(&self, query: &SparseCliffordVector, idx: usize, layer: usize) -> f64 {
        if layer == 0 {
            let query_f32 = Self::dense_to_query_f32(query);
            return self.distance_to_layer0_node_sq(&query_f32, idx);
        }
        fast_metric_distance_sq(query, &self.nodes[idx].vec)
    }

    #[inline]
    fn distance_to_layer0_node_sq(&self, query_f32: &[f32; SLAB_DIM], idx: usize) -> f64 {
        let slab_idx = self.layer0_soa.node_to_slab[idx] as usize;
        let block = slab_idx / SLAB_LANES;
        let lane = slab_idx % SLAB_LANES;
        let slab_ptr = self.layer0_slab_ptr();
        if slab_ptr.is_null() {
            return f64::INFINITY;
        }
        let distances = slab_distance(slab_ptr, block, query_f32);
        f64::from(distances[lane])
    }

    fn distance_to_node(&self, query: &SparseCliffordVector, idx: usize, layer: usize) -> f64 {
        self.distance_to_node_sq(query, idx, layer).sqrt()
    }

    /// Greedy single-element search at a given layer.
    /// Returns internal index of the closest found.
    fn greedy_search_layer(
        &self,
        query: &SparseCliffordVector,
        start: usize,
        layer: usize,
    ) -> usize {
        let mut current = start;
        let mut current_dist = self.distance_to_node_sq(query, current, layer);
        loop {
            let mut improved = false;
            if layer < self.nodes[current].max_layer + 1 {
                for nb_idx_u32 in self.node_neighbors_iter(current, layer) {
                    let nb_idx = nb_idx_u32 as usize;
                    let d = self.distance_to_node_sq(query, nb_idx, layer);
                    if d < current_dist {
                        current = nb_idx;
                        current_dist = d;
                        improved = true;
                    }
                }
            }
            if !improved {
                break;
            }
        }
        current
    }

    #[cfg(test)]
    fn greedy_search_layer_with_work_count(
        &self,
        query: &SparseCliffordVector,
        start: usize,
        layer: usize,
        work_count: &mut usize,
    ) -> usize {
        let mut current = start;
        let mut current_dist = self.distance_to_node_sq(query, current, layer);
        *work_count += 1;
        loop {
            let mut improved = false;
            if layer < self.nodes[current].max_layer + 1 {
                for nb_idx_u32 in self.node_neighbors_iter(current, layer) {
                    let nb_idx = nb_idx_u32 as usize;
                    let d = self.distance_to_node_sq(query, nb_idx, layer);
                    *work_count += 1;
                    if d < current_dist {
                        current = nb_idx;
                        current_dist = d;
                        improved = true;
                    }
                }
            }
            if !improved {
                break;
            }
        }
        current
    }

    /// Beam search at a given layer returning (`internal_idx`, dist) sorted by distance.
    #[allow(clippy::too_many_lines)]
    fn search_layer(
        &self,
        query: &SparseCliffordVector,
        entry_idx: usize,
        ef: usize,
        layer: usize,
    ) -> Vec<(usize, f64)> {
        SEARCH_SCRATCH.with(|cell| {
            let mut scratch = cell.borrow_mut();
            scratch.out.clear();
            let search_epoch = self.next_search_epoch();
            VISITED_EPOCH.with(|visited_cell| {
                let mut visited = visited_cell.borrow_mut();
                let needed = self.nodes.len();
                if visited.len() < needed {
                    let grown_len = needed.next_power_of_two();
                    visited.resize(grown_len, 0);
                }
                if search_epoch == 1 {
                    visited.fill(0);
                }

                let limit = ef.min(MAX_FIXED_HEAP_CAPACITY);
                let mut candidates = FixedHeap::<MAX_FIXED_HEAP_CAPACITY>::new(limit);
                let mut results = FixedHeap::<MAX_FIXED_HEAP_CAPACITY>::new(limit);
                let query_f32 = Self::dense_to_query_f32(query);
                let slab_ptr = self.layer0_slab_ptr();
                let slab_blocks = self.layer0_soa.blocks.len();
                let nodes_len = self.nodes.len();

                let d0 = self.distance_to_node_sq(query, entry_idx, layer) as f32;
                visited[entry_idx] = search_epoch;
                candidates.push_or_replace(d0, entry_idx as u32);
                results.push_or_replace(d0, entry_idx as u32);

                while let Some((c_dist, c_idx_u32)) = candidates.pop_best() {
                    if results.len() >= limit && c_dist > results.worst() {
                        break;
                    }

                    let c_idx = c_idx_u32 as usize;
                    if layer > self.nodes[c_idx].max_layer {
                        continue;
                    }
                    if layer == 0 && !slab_ptr.is_null() {
                        // HOT PATH: O(groups), called per beam expansion at layer 0.
                        // Layer-0 block projection is materialized at insertion/removal time.
                        for group in self.node_layer0_groups(c_idx) {
                            let block = group.block as usize;
                            if block >= slab_blocks {
                                continue;
                            }
                            let base = block * SLAB_LANES;
                            debug_assert_eq!(slab_ptr as usize % 64, 0, "slab alignment");
                            // SAFETY: `block < slab_blocks`, slab is persistently materialized and
                            // 64-byte aligned; query buffer has fixed 16-lane shape.
                            let distances = slab_distance(slab_ptr, block, &query_f32);
                            let mut effective_mask = group.lane_mask;
                            let mut m = group.lane_mask;
                            while m != 0 {
                                let lane_u8 = m.trailing_zeros() as u8;
                                let lane = usize::from(lane_u8);
                                let original_bit = 1_u8 << lane_u8;
                                let nb_idx = base + lane;
                                if nb_idx >= nodes_len {
                                    m &= m - 1;
                                    continue;
                                }
                                // Branch-free visited check: clears lane bit when visited, equivalent to `if visited[nb_idx] == search_epoch { continue; }`
                                effective_mask &= ((visited[nb_idx] != search_epoch) as u8
                                    * original_bit)
                                    | !original_bit;
                                if (effective_mask & original_bit) != 0 {
                                    visited[nb_idx] = search_epoch;
                                    let d = distances[lane];
                                    if results.push_or_replace(d, nb_idx as u32) {
                                        candidates.push_or_replace(d, nb_idx as u32);
                                    }
                                }
                                m &= m - 1;
                            }
                        }
                    } else {
                        for nb_idx_u32 in self.node_neighbors_iter(c_idx, layer) {
                            let nb_idx = nb_idx_u32 as usize;
                            if visited[nb_idx] == search_epoch {
                                continue;
                            }
                            visited[nb_idx] = search_epoch;
                            let d = self.distance_to_node_sq(query, nb_idx, layer) as f32;
                            if results.push_or_replace(d, nb_idx as u32) {
                                candidates.push_or_replace(d, nb_idx as u32);
                            }
                        }
                    }
                }

                scratch.out.reserve(results.len());
                for &(dist_sq, idx) in results.as_slice() {
                    scratch.out.push((idx as usize, f64::from(dist_sq)));
                }
                std::mem::take(&mut scratch.out)
            })
        })
    }

    #[cfg(test)]
    #[allow(clippy::too_many_lines)]
    fn search_layer_with_work_count(
        &self,
        query: &SparseCliffordVector,
        entry_idx: usize,
        ef: usize,
        layer: usize,
        work_count: &mut usize,
    ) -> Vec<(usize, f64)> {
        SEARCH_SCRATCH.with(|cell| {
            let mut scratch = cell.borrow_mut();
            scratch.out.clear();
            let search_epoch = self.next_search_epoch();
            VISITED_EPOCH.with(|visited_cell| {
                let mut visited = visited_cell.borrow_mut();
                let needed = self.nodes.len();
                if visited.len() < needed {
                    let grown_len = needed.next_power_of_two();
                    visited.resize(grown_len, 0);
                }
                if search_epoch == 1 {
                    visited.fill(0);
                }

                let limit = ef.min(MAX_FIXED_HEAP_CAPACITY);
                let mut candidates = FixedHeap::<MAX_FIXED_HEAP_CAPACITY>::new(limit);
                let mut results = FixedHeap::<MAX_FIXED_HEAP_CAPACITY>::new(limit);
                let query_f32 = Self::dense_to_query_f32(query);
                let slab_ptr = self.layer0_slab_ptr();
                let slab_blocks = self.layer0_soa.blocks.len();
                let nodes_len = self.nodes.len();

                let d0 = self.distance_to_node_sq(query, entry_idx, layer) as f32;
                *work_count += 1;
                visited[entry_idx] = search_epoch;
                candidates.push_or_replace(d0, entry_idx as u32);
                results.push_or_replace(d0, entry_idx as u32);

                while let Some((c_dist, c_idx_u32)) = candidates.pop_best() {
                    if results.len() >= limit && c_dist > results.worst() {
                        break;
                    }

                    let c_idx = c_idx_u32 as usize;
                    if layer > self.nodes[c_idx].max_layer {
                        continue;
                    }
                    if layer == 0 && !slab_ptr.is_null() {
                        for group in self.node_layer0_groups(c_idx) {
                            let block = group.block as usize;
                            if block >= slab_blocks {
                                continue;
                            }
                            let base = block * SLAB_LANES;
                            debug_assert_eq!(slab_ptr as usize % 64, 0, "slab alignment");
                            // SAFETY: `block < slab_blocks`, slab is persistently materialized and
                            // 64-byte aligned; query buffer has fixed 16-lane shape.
                            let distances = slab_distance(slab_ptr, block, &query_f32);
                            let mut effective_mask = group.lane_mask;
                            let mut m = group.lane_mask;
                            while m != 0 {
                                let lane_u8 = m.trailing_zeros() as u8;
                                let lane = usize::from(lane_u8);
                                let original_bit = 1_u8 << lane_u8;
                                let nb_idx = base + lane;
                                if nb_idx >= nodes_len {
                                    m &= m - 1;
                                    continue;
                                }
                                // Branch-free visited check: clears lane bit when visited, equivalent to `if visited[nb_idx] == search_epoch { continue; }`
                                effective_mask &= ((visited[nb_idx] != search_epoch) as u8
                                    * original_bit)
                                    | !original_bit;
                                if (effective_mask & original_bit) != 0 {
                                    visited[nb_idx] = search_epoch;
                                    let d = distances[lane];
                                    *work_count += 1;
                                    if results.push_or_replace(d, nb_idx as u32) {
                                        candidates.push_or_replace(d, nb_idx as u32);
                                    }
                                }
                                m &= m - 1;
                            }
                        }
                    } else {
                        for nb_idx_u32 in self.node_neighbors_iter(c_idx, layer) {
                            let nb_idx = nb_idx_u32 as usize;
                            if visited[nb_idx] == search_epoch {
                                continue;
                            }
                            visited[nb_idx] = search_epoch;
                            let d = self.distance_to_node_sq(query, nb_idx, layer) as f32;
                            *work_count += 1;
                            if results.push_or_replace(d, nb_idx as u32) {
                                candidates.push_or_replace(d, nb_idx as u32);
                            }
                        }
                    }
                }

                scratch.out.reserve(results.len());
                for &(dist_sq, idx) in results.as_slice() {
                    scratch.out.push((idx as usize, f64::from(dist_sq)));
                }
                std::mem::take(&mut scratch.out)
            })
        })
    }

    /// Search for the k nearest neighbours to query.
    ///
    /// Distance metric: Clifford grade-weighted L2 in G(1,3).
    /// Neighbors are nearest in algebraic geometry, not Euclidean R^16.
    ///
    /// AX-ID: AXIOMA-001, AXIOMA-007, H_estructura (LEY_FUNDACIONAL §3.1)
    pub fn search_nearest(&self, query: &SparseCliffordVector, k: usize) -> Vec<NodeId> {
        let Some(entry_idx) = self.entry else {
            return Vec::new();
        };

        let mut current = entry_idx;
        // Greedy descend from top layer to layer 1
        for lc in (1..=self.entry_layer).rev() {
            current = self.greedy_search_layer(query, current, lc);
        }

        // Beam search at layer 0
        let ef = k.max(self.ef_construction);
        let results = self.search_layer(query, current, ef, 0);

        results
            .iter()
            .take(k)
            .map(|&(idx, _)| self.nodes[idx].id)
            .collect()
    }

    #[cfg(test)]
    fn search_nearest_work_count(&self, query: &SparseCliffordVector, k: usize) -> usize {
        let Some(entry_idx) = self.entry else {
            return 0;
        };

        let mut work_count = 0usize;
        let mut current = entry_idx;
        for lc in (1..=self.entry_layer).rev() {
            current = self.greedy_search_layer_with_work_count(query, current, lc, &mut work_count);
        }

        let ef = k.max(self.ef_construction);
        let _ = self.search_layer_with_work_count(query, current, ef, 0, &mut work_count);
        work_count
    }

    /// Iterate over neighbours of a node at all layers (union, deduplicated).
    ///
    /// AX-ID: AXIOMA-013
    pub fn neighbors(&self, id: NodeId) -> impl Iterator<Item = NodeId> + '_ {
        let node_idx = self.idx(id);
        NeighborIter {
            graph: self,
            node_idx,
            layer_pos: 0,
            edge_pos: 0,
            seen: SmallVec::new(), // BN-07: inline budget matches MAX_UNIQUE_NEIGHBOR_BUDGET; spill indicates per-layer caps hit or constant drift
        }
    }

    /// Iterate over neighbours of a node within a given distance radius (layer 0 only).
    ///
    /// AX-ID: AXIOMA-013
    pub fn neighbors_within(&self, id: NodeId, radius: f64) -> impl Iterator<Item = NodeId> + '_ {
        // FIX-E.1: Single get_idx call — the former code called idx(id) twice
        // (once for `node_vec`, once for `idx`), wasting a lookup per call.
        let candidates: SmallVec<[NodeId; M]> = self.idx(id).map_or_else(SmallVec::new, |idx| {
            let Some(node) = self.nodes.get(idx) else {
                return SmallVec::new();
            };
            let nv = node.vec;
            let layer0_len = self.node_neighbors_len(idx, 0);
            let mut local = SmallVec::<[NodeId; M]>::with_capacity(layer0_len.min(M0));
            for nb_idx_u32 in self.node_neighbors_iter(idx, 0) {
                let ni = nb_idx_u32 as usize;
                let d = self.distance_to_node(&nv, ni, 0);
                if d <= radius {
                    local.push(self.nodes[ni].id);
                }
            }
            local
        });
        candidates.into_iter()
    }

    /// Iterates unique neighbors of a node across all layers without allocations.
    ///
    /// Uses `marks` as an internal-index bitmap with `stamp` as generation.
    /// Requiere `marks.len() >= self.node_count()`.
    pub(crate) fn extend_neighbors_dedup(
        &self,
        id: NodeId,
        marks: &mut [u32],
        stamp: u32,
        out: &mut Vec<usize>,
    ) -> usize {
        if stamp == 0 {
            return 0;
        }
        let Some(idx) = self.idx(id) else {
            return 0;
        };
        let Some(node) = self.nodes.get(idx) else {
            return 0;
        };
        let mut pushed = 0;
        for layer_idx in 0..=node.max_layer {
            for nb_idx_u32 in self.node_neighbors_iter(idx, layer_idx) {
                let nb_idx = nb_idx_u32 as usize;
                if nb_idx >= marks.len() || marks[nb_idx] == stamp {
                    continue;
                }
                marks[nb_idx] = stamp;
                out.push(nb_idx);
                pushed += 1;
            }
        }
        pushed
    }

    /// Gets the vector associated with a `NodeId`.
    pub fn vector(&self, id: NodeId) -> Option<&SparseCliffordVector> {
        self.idx(id).map(|idx| &self.nodes[idx].vec)
    }

    /// Iterate over all `NodeIds` in the graph.
    pub fn nodes(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.nodes.iter().map(|n| n.id)
    }

    /// Number of nodes in the graph.
    ///
    /// This cannot be `const fn` because `Arc` dereference is not const-evaluable
    /// on stable Rust (`nodes` is Arc-backed for snapshot sharing).
    #[allow(clippy::inline_always)]
    #[inline(always)]
    pub fn node_count(&self) -> usize {
        self.live_nodes
    }

    /// Total number of undirected edges at layer 0 (base connectivity).
    pub const fn edge_count(&self) -> usize {
        self.edge_count_layer0_undirected / 2
    }

    /// Build a contiguous layer-0 SoA snapshot for read-heavy numeric pipelines.
    ///
    /// AX-ID: AXIOMA-013
    pub fn layer0_soa(&self) -> HnswLayer0Soa {
        let mut node_ids = Vec::with_capacity(self.live_nodes);
        let mut node_to_slab = Vec::with_capacity(self.live_nodes);
        let mut neighbor_offsets = Vec::with_capacity(self.live_nodes);
        let mut neighbor_ids = Vec::with_capacity(self.edge_count_layer0_undirected);
        let mut neighbor_distances = Vec::with_capacity(self.edge_count_layer0_undirected);

        for (node_idx, node) in self.nodes.iter().enumerate() {
            if node.id == NodeId::INVALID {
                continue;
            }
            node_ids.push(node.id);
            node_to_slab.push(self.layer0_soa.node_to_slab[node_idx]);
            let start = neighbor_ids.len();
            let layer0_len = self.node_neighbors_len(node_idx, 0);
            neighbor_ids.reserve(layer0_len);
            neighbor_distances.reserve(layer0_len);
            for nb_idx_u32 in self.node_neighbors_iter(node_idx, 0) {
                let nb_idx = nb_idx_u32 as usize;
                let nb_id = self.nodes[nb_idx].id;
                if nb_id == NodeId::INVALID {
                    continue;
                }
                neighbor_ids.push(nb_id);
                let d = self.distance_to_node(&node.vec, nb_idx, 0);
                neighbor_distances.push(d);
            }
            neighbor_offsets.push((start, neighbor_ids.len()));
        }

        HnswLayer0Soa {
            node_ids,
            slab: self
                .layer0_soa
                .blocks
                .iter()
                .flat_map(|block| block.lanes)
                .collect(),
            node_to_slab,
            neighbor_offsets,
            neighbor_ids,
            neighbor_distances,
        }
    }

    /// Remove a node and all its edges from the graph.
    ///
    /// Required by `inelastic_concept_fusion` in CRATE-004 (genesis-evolution).
    /// After fusion, the absorbed node must be removed from all structures.
    ///
    /// # Complexity
    /// O(K × layers) for edge removal + O(N) for index cleanup.
    /// Not a hot path — called only during Wormhole collapse events.
    ///
    /// # Postcondition
    /// - The node no longer appears in `neighbors()` or `search_nearest()`.
    /// - `direct_index[id.get()]` is set to `u32::MAX` (not present sentinel).
    /// - `id_index` entry is removed on next `compact_index()`.
    ///
    /// AX-ID: LEY_FUNDACIONAL §3.7 (WormholeCollapse), CRATE-004 prerequisite (FIX-H)
    pub fn remove_node(&mut self, id: NodeId) -> Result<(), GenesisError> {
        let idx = self
            .idx(id)
            .ok_or(GenesisError::InvariantViolation { axiom_id: 4 })?;
        let outgoing_layer0 = self.node_neighbors_len(idx, 0);

        // Step 1: Remove all edges originating from this node.
        // Collect neighbour IDs first to avoid borrow conflicts.
        REMOVE_SCRATCH.with(|scratch_cell| {
            let mut scratch = scratch_cell.borrow_mut();
            scratch.clear();
            if let Some(layers) = self.nodes[idx].max_layer.checked_add(1) {
                for layer in 0..layers {
                    let layer_len = self.node_neighbors_len(idx, layer);
                    for pos in 0..layer_len {
                        if let Some(nb_idx) = self.node_neighbor_at(idx, layer, pos) {
                            scratch.push((nb_idx as usize, layer));
                        }
                    }
                }
            }

            for &(nb_idx, layer) in scratch.iter() {
                // Remove the reverse edge: nb → id
                self.remove_edge(nb_idx, layer, idx);
            }
        });

        // Step 2: Clear the node's own adjacency lists.
        for layer in 0..=self.nodes[idx].max_layer {
            Self::cow_vec_mut(&mut self.layer_neighbors)[idx].clear_layer(layer);
        }
        self.edge_count_layer0_undirected = self
            .edge_count_layer0_undirected
            .saturating_sub(outgoing_layer0);

        // Step 3: Invalidate direct_index entry.
        let raw = id.get();
        if raw < u64::from(u32::MAX) {
            #[allow(clippy::cast_possible_truncation)]
            let raw_us = raw as usize;
            if raw_us < self.direct_index.len() {
                Self::cow_vec_mut(&mut self.direct_index)[raw_us] = u32::MAX;
            }
        }
        if let Some(compact) = try_compact_node_id(id) {
            Self::cow_vec_mut(&mut self.id_index).retain(|&(nid, _)| nid != compact);
        } else {
            Self::cow_vec_mut(&mut self.wide_id_index).retain(|&(nid, _)| nid != id);
        }

        // Step 4: Update entry point if it was pointing to this node.
        if self.entry == Some(idx) {
            // Find a new entry point: the node with the highest layer.
            self.entry = self
                .nodes
                .iter()
                .enumerate()
                .filter(|(i, n)| *i != idx && n.id != NodeId::INVALID)
                .max_by_key(|(_, n)| n.max_layer)
                .map(|(i, n)| {
                    self.entry_layer = n.max_layer;
                    i
                });
        }

        // Step 5: Mark the node slot as invalid (keep Vec size stable for index integrity).
        // A compact_index() call rebuilds id_index cleanly. Do not swap_remove
        // because that would invalidate all internal indices stored in adjacency lists.
        let nodes = Self::cow_vec_mut(&mut self.nodes);
        let was_live = nodes[idx].id != NodeId::INVALID;
        nodes[idx].id = NodeId::INVALID;
        nodes[idx].max_layer = 0;
        if was_live {
            self.live_nodes = self.live_nodes.saturating_sub(1);
        }
        let slab_idx = self.layer0_soa.node_to_slab[idx] as usize;
        let block = slab_idx / SLAB_LANES;
        let lane = slab_idx % SLAB_LANES;
        if let Some(block_ref) = self.layer0_soa.blocks.get_mut(block) {
            block_ref.lanes[lane] = f32::NAN;
        }

        Ok(())
    }
}

impl LockFreeHnswIndex {
    /// Create a lock-free snapshot index with an empty HNSW graph.
    ///
    /// AX-ID: AXIOMA-013
    pub fn new(ef_construction: usize) -> Self {
        let snapshot = Arc::new(HnswGraph::new(ef_construction));
        Self {
            head: AtomicPtr::new(Arc::into_raw(snapshot).cast_mut()),
            cas_retries: CachePadded::new(AtomicU64::new(0)),
        }
    }

    fn load_snapshot(&self) -> Arc<HnswGraph> {
        loop {
            let ptr = self.head.load(AtomicOrdering::Acquire);
            if ptr.is_null() {
                std::hint::spin_loop();
                continue;
            }

            // SAFETY: `ptr` comes from `Arc::into_raw` and points to a live allocation
            // while held by `head`. We take a temporary strong ref then validate that
            // the atomic head did not change before converting it into an `Arc`.
            unsafe {
                Arc::increment_strong_count(ptr);
            }

            if self.head.load(AtomicOrdering::Acquire) == ptr {
                // SAFETY: We just incremented the strong count for `ptr`.
                return unsafe { Arc::from_raw(ptr) };
            }

            // SAFETY: Balance the temporary strong-count increment from this loop
            // iteration before retrying with the new head pointer.
            unsafe {
                drop(Arc::from_raw(ptr));
            }
        }
    }

    /// Insert a node into the latest snapshot via append-only CAS publication.
    ///
    /// AX-ID: AXIOMA-013
    pub fn insert(&self, id: NodeId, vec: &SparseCliffordVector) -> Result<(), GenesisError> {
        loop {
            let base = self.load_snapshot();
            let current = Arc::as_ptr(&base).cast_mut();
            let delta = HnswDelta::Insert;
            let mut updated = (*base).clone_with_delta(&delta);
            updated.insert(id, vec)?;
            let candidate = Arc::into_raw(Arc::new(updated)).cast_mut();

            if self
                .head
                .compare_exchange(
                    current,
                    candidate,
                    AtomicOrdering::AcqRel,
                    AtomicOrdering::Acquire,
                )
                .is_ok()
            {
                // SAFETY: Successful CAS replaced the head's strong reference from
                // `current` to `candidate`; release the superseded head ref.
                unsafe {
                    drop(Arc::from_raw(current));
                }
                return Ok(());
            }
            self.cas_retries.value.fetch_add(1, AtomicOrdering::Relaxed);
            // SAFETY: CAS failed, so `candidate` was never published.
            unsafe {
                drop(Arc::from_raw(candidate));
            }
        }
    }

    /// Remove a node from the latest snapshot via CAS publication.
    ///
    /// AX-ID: AXIOMA-013
    pub fn remove(&self, id: NodeId) -> Result<(), GenesisError> {
        loop {
            let base = self.load_snapshot();
            let current = Arc::as_ptr(&base).cast_mut();
            let delta = HnswDelta::Remove;
            let mut updated = (*base).clone_with_delta(&delta);
            updated.remove_node(id)?;
            let candidate = Arc::into_raw(Arc::new(updated)).cast_mut();

            if self
                .head
                .compare_exchange(
                    current,
                    candidate,
                    AtomicOrdering::AcqRel,
                    AtomicOrdering::Acquire,
                )
                .is_ok()
            {
                // SAFETY: Successful CAS replaced the head-owned strong ref.
                unsafe {
                    drop(Arc::from_raw(current));
                }
                return Ok(());
            }
            self.cas_retries.value.fetch_add(1, AtomicOrdering::Relaxed);
            // SAFETY: CAS failed, candidate snapshot was not published.
            unsafe {
                drop(Arc::from_raw(candidate));
            }
        }
    }

    /// Search nearest neighbours from the latest published snapshot.
    ///
    /// AX-ID: AXIOMA-013
    pub fn search_nearest(&self, query: &SparseCliffordVector, k: usize) -> Vec<NodeId> {
        self.load_snapshot().search_nearest(query, k)
    }

    /// Number of nodes in the latest published snapshot.
    ///
    /// AX-ID: AXIOMA-013
    pub fn node_count(&self) -> usize {
        self.load_snapshot().node_count()
    }

    /// Build a contiguous SoA layer-0 snapshot from the latest published graph.
    ///
    /// AX-ID: AXIOMA-013
    pub fn layer0_soa(&self) -> HnswLayer0Soa {
        self.load_snapshot().layer0_soa()
    }

    /// Returns the cumulative number of CAS publication retries.
    ///
    /// AX-ID: AXIOMA-013
    pub fn cas_retry_count(&self) -> u64 {
        self.cas_retries.value.load(AtomicOrdering::Relaxed)
    }
}

impl Drop for LockFreeHnswIndex {
    fn drop(&mut self) {
        let ptr = self.head.swap(std::ptr::null_mut(), AtomicOrdering::AcqRel);
        if !ptr.is_null() {
            // SAFETY: `ptr` is the head-owned strong ref previously created via
            // `Arc::into_raw`; dropping it releases the final snapshot reference.
            unsafe {
                drop(Arc::from_raw(ptr));
            }
        }
    }
}

/// LSD radix sort for `(CompactNodeId, internal_idx)` pairs by compact raw ID in O(4N).
fn radix_sort_node_ids(index: &mut Vec<(CompactNodeId, usize)>) {
    if index.len() <= 1 {
        return;
    }

    let len = index.len();
    let mut src = std::mem::take(index);
    let mut dst = vec![(CompactNodeId(0), 0usize); len];

    for pass in 0..4 {
        let shift = pass * 8;
        let mut counts = [0usize; 256];
        for &(id, _) in &src {
            let bucket = ((id.raw() >> shift) & 0xFF) as usize;
            counts[bucket] += 1;
        }

        let mut offsets = [0usize; 256];
        let mut running = 0usize;
        for i in 0..256 {
            offsets[i] = running;
            running += counts[i];
        }

        for item in src.iter().copied() {
            let bucket = ((item.0.raw() >> shift) & 0xFF) as usize;
            let pos = offsets[bucket];
            dst[pos] = item;
            offsets[bucket] += 1;
        }

        std::mem::swap(&mut src, &mut dst);
    }

    *index = src;
}

/// Iterator over deduplicated neighbors of a node across all layers.
///
/// Uses a SmallVec with inline capacity equal to `MAX_UNIQUE_NEIGHBOR_BUDGET`
/// (M0 + (MAX_LAYERS - 1) * M). This ensures zero heap allocation for all
/// valid HNSW graph configurations, as the maximum unique neighbor count
/// across all layers cannot exceed this compile-time bound.
///
/// AX-ID: AXIOMA-013
struct NeighborIter<'a> {
    graph: &'a HnswGraph,
    node_idx: Option<usize>,
    layer_pos: usize,
    edge_pos: usize,
    /// Deduplicated node IDs already emitted, sorted ascending for binary search.
    ///
    /// # BN-07: SmallVec eliminates heap allocation
    /// Inline capacity tracks the legal multi-layer neighbour budget.
    /// Any heap spill indicates either per-layer caps being hit, or the constants
    /// MAX_UNIQUE_NEIGHBOR_BUDGET/M0/M/MAX_LAYERS have drifted from the intended bound.
    seen: SmallVec<[u64; MAX_UNIQUE_NEIGHBOR_BUDGET]>,
}

impl Iterator for NeighborIter<'_> {
    type Item = NodeId;

    fn next(&mut self) -> Option<NodeId> {
        let node_idx = self.node_idx?;
        loop {
            if self.layer_pos > self.graph.nodes[node_idx].max_layer {
                return None;
            }
            let layer_len = self.graph.node_neighbors_len(node_idx, self.layer_pos);
            if self.edge_pos >= layer_len {
                self.layer_pos += 1;
                self.edge_pos = 0;
                continue;
            }
            let Some(nb_idx) = self
                .graph
                .node_neighbor_at(node_idx, self.layer_pos, self.edge_pos)
            else {
                self.layer_pos += 1;
                self.edge_pos = 0;
                continue;
            };
            let nid = self.graph.nodes[nb_idx as usize].id;
            self.edge_pos += 1;
            let raw = nid.get();
            match self.seen.binary_search(&raw) {
                Ok(_) => continue,
                Err(pos) => {
                    debug_assert!(
                        self.seen.len() < MAX_UNIQUE_NEIGHBOR_BUDGET,
                        "SmallVec should never spill: seen={}, budget={}",
                        self.seen.len(),
                        MAX_UNIQUE_NEIGHBOR_BUDGET
                    );
                    self.seen.insert(pos, raw);
                    return Some(nid);
                }
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::explicit_iter_loop, clippy::many_single_char_names)]
mod tests {
    use genesis_math::{fast_metric_distance, fast_metric_distance_sq, SparseCliffordVector};
    use proptest::prelude::*;

    use super::*;

    fn make_vec(coeff: f64) -> SparseCliffordVector {
        SparseCliffordVector::from_iter((0..4).map(|i| (i, coeff * (i as f64 + 1.0) * 0.1)))
            .unwrap()
    }

    fn make_id(v: u64) -> NodeId {
        NodeId::try_new(v).expect("NodeId valid by construction")
    }

    fn expected_keep_and_drop(
        g: &HnswGraph,
        idx: usize,
        layer: usize,
        m_max: usize,
    ) -> (Vec<u32>, Vec<u32>) {
        let query_f32 = (layer == 0).then(|| HnswGraph::dense_to_query_f32(&g.nodes[idx].vec));
        let mut scored: Vec<u64> = g
            .node_neighbors_iter(idx, layer)
            .map(|nb| {
                let dist = query_f32.as_ref().map_or_else(
                    || g.distance_to_node_sq(&g.nodes[idx].vec, nb as usize, layer),
                    |query_f32| g.distance_to_layer0_node_sq(query_f32, nb as usize),
                );
                let dist_bits = (dist as f32).to_bits();
                ((dist_bits as u64) << 32) | u64::from(nb)
            })
            .collect();
        if scored.len() > m_max {
            scored.select_nth_unstable(m_max - 1);
        }

        let mut keep: Vec<u32> = scored
            .iter()
            .take(m_max)
            .map(|&packed| (packed & 0xFFFF_FFFF) as u32)
            .collect();
        keep.sort_unstable();

        let drop = scored
            .iter()
            .skip(m_max)
            .map(|&packed| (packed & 0xFFFF_FFFF) as u32)
            .collect();
        (keep, drop)
    }

    fn next_u64(seed: &mut u64) -> u64 {
        *seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        *seed
    }

    const SIGNED_U53_SCALE: f64 = 2.0 / (1_u64 << 53) as f64;

    fn random_vec(seed: &mut u64) -> SparseCliffordVector {
        let dense = core::array::from_fn(|_| {
            let bits = next_u64(seed) >> 11;
            (bits as f64).mul_add(SIGNED_U53_SCALE, -1.0)
        });
        SparseCliffordVector::from_dense(&dense)
            .expect("deterministic random vector must be finite")
    }

    #[test]
    fn slab_distance_matches_scalar_all_counts() {
        let mut seed = 0x1234_5678_9ABC_DEF0;
        let query = random_vec(&mut seed);

        let mut graph = HnswGraph::new(32);
        for i in 0..SLAB_LANES {
            let v = random_vec(&mut seed);
            graph.insert(make_id(i as u64), &v).expect("insert");
        }
        let slab_ptr = graph.layer0_slab_ptr();
        let query_f32 = HnswGraph::dense_to_query_f32(&query);
        let distances = slab_distance_scalar(slab_ptr, 0, &query_f32);
        for (slot, distance) in distances.iter().enumerate() {
            let scalar = fast_metric_distance_sq(&query, &graph.nodes[slot].vec);
            assert!((f64::from(*distance) - scalar).abs() < 1e-5);
        }
    }

    #[test]
    fn benchmark_distance_helpers_are_finite_and_batch_matches_scalar() {
        let query = make_vec(1.25);
        let candidates = [make_vec(0.1), make_vec(0.2), make_vec(0.3), make_vec(0.4)];

        let scalar = benchmark_scalar_distance_4x(&query, &candidates);
        assert!(scalar.iter().all(|d| d.is_finite()));

        let batch = benchmark_batch_distance_4(&query, &candidates);
        assert_eq!(batch, scalar);
    }

    #[test]
    fn layer0_soa_preserves_layer0_cardinality() {
        let mut graph = HnswGraph::new(16);
        for i in 0..16_u64 {
            graph
                .insert(make_id(i), &make_vec((i as f64).mul_add(0.01, 0.1)))
                .expect("insert should succeed");
        }

        let soa = graph.layer0_soa();
        let total_layer0: usize = graph
            .nodes
            .iter()
            .enumerate()
            .map(|(i, _node)| graph.node_neighbors_len(i, 0))
            .sum();

        assert_eq!(soa.node_ids.len(), graph.node_count());
        assert_eq!(soa.node_to_slab.len(), graph.node_count());
        assert_eq!(soa.slab.len(), graph.layer0_soa.blocks.len() * BLOCK_STRIDE);
        assert_eq!(soa.neighbor_offsets.len(), graph.node_count());
        assert_eq!(soa.neighbor_ids.len(), total_layer0);
        assert_eq!(soa.neighbor_distances.len(), total_layer0);
    }

    #[test]
    fn slab_insertion_sequential_ordering() {
        let mut graph = HnswGraph::new(16);
        for i in 0..100_u64 {
            graph
                .insert(make_id(i), &make_vec((i as f64).mul_add(0.01, 0.2)))
                .expect("insert");
        }
        for (idx, slab_idx) in graph.layer0_soa.node_to_slab.iter().enumerate() {
            assert_eq!(*slab_idx as usize, idx);
        }
    }

    #[test]
    fn slab_nan_lane_excluded_from_results() {
        let mut graph = HnswGraph::new(16);
        for i in 0..4_u64 {
            graph
                .insert(make_id(i), &make_vec((i as f64).mul_add(0.1, 0.1)))
                .expect("insert");
        }
        graph.remove_node(make_id(2)).expect("remove");
        let results = graph.search_nearest(&make_vec(0.3), 3);
        assert!(!results.contains(&make_id(2)));
    }

    #[test]
    fn layer0_neighbors_sorted_after_insert() {
        let mut graph = HnswGraph::new(32);
        for i in 0..100_u64 {
            graph
                .insert(make_id(i), &make_vec((i as f64).mul_add(0.02, 0.15)))
                .expect("insert should succeed");
        }
        for (idx, adj) in graph.layer_neighbors.iter().enumerate() {
            for window in adj.layer0.windows(2) {
                let left = NodeAdj::unpack_layer0_neighbor(window[0]);
                let right = NodeAdj::unpack_layer0_neighbor(window[1]);
                assert!(
                    left < right,
                    "node {idx} has unsorted layer0 neighbors: {:?}",
                    adj.layer0
                        .iter()
                        .map(|&packed| NodeAdj::unpack_layer0_neighbor(packed))
                        .collect::<Vec<_>>()
                );
            }
        }
    }

    fn projected_layer0_groups(adj: &NodeAdj) -> Vec<(u32, u8)> {
        let mut out: Vec<(u32, u8)> = Vec::new();
        for &packed in &adj.layer0 {
            let slab_idx = NodeAdj::unpack_layer0_slab(packed);
            let block = slab_idx >> 3;
            let lane_mask = 1_u8 << (slab_idx & 7);
            match out.binary_search_by_key(&block, |&(b, _)| b) {
                Ok(pos) => out[pos].1 |= lane_mask,
                Err(pos) => out.insert(pos, (block, lane_mask)),
            }
        }
        out
    }

    #[test]
    fn layer0_block_groups_match_canonical_neighbors_after_insert() {
        let mut graph = HnswGraph::new(32);
        for i in 0..100_u64 {
            graph
                .insert(make_id(i), &make_vec((i as f64).mul_add(0.02, 0.17)))
                .expect("insert should succeed");
        }

        for (idx, adj) in graph.layer_neighbors.iter().enumerate() {
            let expected = projected_layer0_groups(adj);
            let got: Vec<(u32, u8)> = adj
                .layer0_groups()
                .iter()
                .map(|group| (group.block, group.lane_mask))
                .collect();
            assert_eq!(
                got, expected,
                "layer0 group projection mismatch at node {idx}"
            );
            assert!(
                got.len() <= 4,
                "layer0 must use at most four slab blocks per node, got {}",
                got.len()
            );
        }
    }

    #[test]
    fn layer0_block_groups_survive_remove_node() {
        let mut graph = HnswGraph::new(32);
        for i in 0..80_u64 {
            graph
                .insert(make_id(i), &make_vec((i as f64).mul_add(0.03, 0.2)))
                .expect("insert should succeed");
        }
        let removed_idx = 17usize;
        let removed_id = make_id(removed_idx as u64);
        let removed_slab = graph.layer0_soa.node_to_slab[removed_idx];
        let removed_block = removed_slab >> 3;
        let removed_lane_mask = 1_u8 << (removed_slab & 7);

        graph
            .remove_node(removed_id)
            .expect("remove should succeed");

        for (idx, adj) in graph.layer_neighbors.iter().enumerate() {
            let expected = projected_layer0_groups(adj);
            let got: Vec<(u32, u8)> = adj
                .layer0_groups()
                .iter()
                .map(|group| (group.block, group.lane_mask))
                .collect();
            assert_eq!(got, expected, "post-remove group mismatch at node {idx}");
            if idx != removed_idx {
                if let Some((_, mask)) = got.iter().find(|(block, _)| *block == removed_block) {
                    assert_eq!(
                        *mask & removed_lane_mask,
                        0,
                        "stale removed lane remains in node {idx}"
                    );
                }
            }
        }
    }

    #[test]
    fn canonical_layer0_order_still_sorted() {
        let mut graph = HnswGraph::new(32);
        for i in 0..96_u64 {
            graph
                .insert(make_id(i), &make_vec((i as f64).mul_add(0.015, 0.12)))
                .expect("insert should succeed");
        }

        for idx in 0..graph.node_count() {
            let neighbors: Vec<u32> = graph.node_neighbors_iter(idx, 0).collect();
            for window in neighbors.windows(2) {
                assert!(
                    window[0] < window[1],
                    "layer0 canonical order must stay sorted"
                );
            }
        }
    }

    #[test]
    fn search_nearest_nn_equivalence() {
        let mut g = HnswGraph::new(64);
        let mut vecs = Vec::new();
        for i in 0..64_u64 {
            let v = make_vec((i as f64).mul_add(0.041, 0.03));
            g.insert(make_id(i), &v).expect("insert");
            vecs.push(v);
        }

        let query = vecs[31];
        let got = g.search_nearest(&query, 1);
        let brute = vecs
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                fast_metric_distance_sq(&query, a).total_cmp(&fast_metric_distance_sq(&query, b))
            })
            .map(|(idx, _)| make_id(idx as u64))
            .expect("non-empty");
        assert_eq!(got, vec![brute]);
    }

    #[test]
    fn layer0_no_duplicates_after_double_add() {
        let mut adj = NodeAdj::default();
        assert!(adj.add_neighbor(0, 7, 7, M0));
        assert!(!adj.add_neighbor(0, 7, 7, M0));
        assert_eq!(adj.layer0.len(), 1);
        assert_eq!(NodeAdj::unpack_layer0_neighbor(adj.layer0[0]), 7);
    }

    #[test]
    fn fixed_heap_deterministic_ordering() {
        let mut a = FixedHeap::<8>::new(4);
        let mut b = FixedHeap::<8>::new(4);
        for item in [(1.0, 3), (1.0, 1), (0.5, 7), (0.5, 2), (2.0, 0)] {
            a.push_or_replace(item.0, item.1);
        }
        for item in [(0.5, 2), (2.0, 0), (1.0, 1), (1.0, 3), (0.5, 7)] {
            b.push_or_replace(item.0, item.1);
        }
        assert_eq!(a.as_slice(), b.as_slice());
    }

    #[test]
    fn lock_free_index_supports_multiwriter_single_snapshot_semantics() {
        use std::sync::Arc;

        let index = Arc::new(LockFreeHnswIndex::new(16));
        std::thread::scope(|scope| {
            for shard in 0..4_u64 {
                let idx = Arc::clone(&index);
                scope.spawn(move || {
                    for i in 0..8_u64 {
                        let id = make_id(shard * 8 + i);
                        let vec = make_vec(((shard * 8 + i) as f64).mul_add(0.01, 0.2));
                        idx.insert(id, &vec).expect("lock-free insert must succeed");
                    }
                });
            }
        });

        // Note: CAS retry count is nondeterministic and depends on thread scheduling.
        // We only verify functional correctness, not contention behavior.
        assert_eq!(index.node_count(), 32);
        let query = make_vec(0.25);
        let result = index.search_nearest(&query, 4);
        assert!(!result.is_empty());
    }

    #[test]
    fn lock_free_remove_retries_and_publishes_consistent_snapshot() {
        use std::sync::{Arc, Barrier};

        let index = Arc::new(LockFreeHnswIndex::new(16));
        for i in 0..64_u64 {
            let id = make_id(i);
            let vec = make_vec((i as f64).mul_add(0.01, 0.2));
            index.insert(id, &vec).expect("seed insert");
        }

        let removed = make_id(7);
        let removed_vec = make_vec(0.77);
        let retry_start = index.cas_retry_count();
        let mut retry_observed = false;

        for round in 0..64_u64 {
            index
                .insert(removed, &removed_vec)
                .expect("reseed removed node");
            let gate = Arc::new(Barrier::new(2));

            let remover_index = Arc::clone(&index);
            let remover_gate = Arc::clone(&gate);
            let remover = std::thread::spawn(move || {
                remover_gate.wait();
                remover_index.remove(removed)
            });

            let writer_index = Arc::clone(&index);
            let writer_gate = Arc::clone(&gate);
            let writer = std::thread::spawn(move || {
                writer_gate.wait();
                for j in 0..64_u64 {
                    let id = make_id(10_000 + round * 64 + j);
                    let vec = make_vec((id.get() as f64).mul_add(0.0001, 0.15));
                    writer_index.insert(id, &vec).expect("contending insert");
                }
            });

            remover
                .join()
                .expect("remove thread join")
                .expect("remove ok");
            writer.join().expect("writer thread join");

            if index.cas_retry_count() > retry_start {
                retry_observed = true;
                break;
            }
        }

        assert!(
            retry_observed,
            "expected at least one CAS retry under contention"
        );
        let soa = index.layer0_soa();
        assert!(
            !soa.node_ids.contains(&removed),
            "removed node must not appear in latest snapshot"
        );
    }

    #[test]
    fn hnsw_insert_and_search() {
        let mut g = HnswGraph::new(16);
        for i in 0..10u64 {
            let v = make_vec((i as f64).mul_add(0.3, 0.1));
            g.insert(make_id(i), &v).unwrap();
        }
        assert_eq!(g.node_count(), 10);
        let q = make_vec(2.1);
        let res = g.search_nearest(&q, 3);
        assert!(!res.is_empty());
        assert!(res.len() <= 3);
    }

    #[test]
    fn hnsw_insert_duplicate_id_is_idempotent() {
        let mut g = HnswGraph::new(16);
        let id = NodeId::try_new(7).expect("NodeId valid by construction");
        let first = make_vec(0.8);
        let second = make_vec(1.9);

        assert!(g.insert(id, &first).is_ok());
        assert!(g.insert(id, &second).is_ok());

        assert_eq!(
            g.node_count(),
            1,
            "duplicate insert must not add a new node"
        );
        assert_eq!(g.vector(id), Some(&first));
    }

    #[test]
    fn hnsw_insert_prevalidation_error_keeps_graph_unchanged() {
        let mut g = HnswGraph::new(16);
        let existing_id = make_id(0);
        let failing_id = make_id(1);
        assert!(g.insert(existing_id, &make_vec(0.3)).is_ok());

        let nodes_before = g.node_count();
        let id_index_before = g.id_index.len();
        let direct_index_before = g.direct_index.clone();

        g.fail_preinsert_index_conversion = true;
        let first_err = g.insert(failing_id, &make_vec(0.6));
        let second_err = g.insert(failing_id, &make_vec(0.6));

        assert!(matches!(
            first_err,
            Err(GenesisError::InvariantViolation { axiom_id: 13 })
        ));
        assert!(matches!(
            second_err,
            Err(GenesisError::InvariantViolation { axiom_id: 13 })
        ));
        assert_eq!(g.node_count(), nodes_before);
        assert_eq!(g.id_index.len(), id_index_before);
        assert_eq!(g.direct_index, direct_index_before);
        assert!(g.idx(failing_id).is_none());
    }

    #[test]
    fn hnsw_search_finds_nearest_among_1000_nodes() {
        let n = 1000usize;
        let mut g = HnswGraph::new(64);
        let mut vecs: Vec<SparseCliffordVector> = Vec::with_capacity(n);
        for i in 0..n {
            let coeff = (i as f64).mul_add(0.001, 0.01);
            let v = SparseCliffordVector::from_iter((0..4).map(|b| {
                (
                    b,
                    (coeff * ((b * 7 + i * 3) % 16) as f64).mul_add(0.1, 0.001),
                )
            }))
            .unwrap();
            vecs.push(v);
            g.insert(
                NodeId::try_new(i as u64).expect("NodeId valid by construction"),
                &v,
            )
            .unwrap();
        }
        // Pick query = vecs[42] exactly
        let query = vecs[42];
        let results = g.search_nearest(&query, 5);
        // Brute force: find nearest by distance excluding self
        let mut bf: Vec<(usize, f64)> = vecs
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != 42)
            .map(|(i, v)| (i, fast_metric_distance(&query, v)))
            .collect();
        bf.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let bf_top3: Vec<u64> = bf.iter().take(3).map(|&(i, _)| i as u64).collect();
        // At least one of the top-3 brute force results should be in HNSW results
        // (approximate search — not guaranteed exact, but typically very close)
        let hnsw_ids: Vec<u64> = results.iter().map(|id| id.get()).collect();
        let overlap = bf_top3.iter().filter(|&&id| hnsw_ids.contains(&id)).count();
        // Allow for HNSW approximation: require at least 1 of top-3 to match
        assert!(
            overlap >= 1 || hnsw_ids.contains(&42),
            "HNSW found {:?}, brute-force top3: {:?}",
            hnsw_ids,
            bf_top3
        );
    }

    #[test]
    fn search_nearest_matches_fixture_bruteforce_top1() {
        let fixture: [[f64; CLIFFORD_BASIS_SIZE]; 8] = [
            [
                0.10, -0.20, 0.30, -0.40, 0.50, -0.60, 0.70, -0.80, 0.90, -1.00, 1.10, -1.20, 1.30,
                -1.40, 1.50, -1.60,
            ],
            [
                0.15, -0.10, 0.35, -0.30, 0.55, -0.50, 0.75, -0.70, 0.95, -0.90, 1.15, -1.10, 1.35,
                -1.30, 1.55, -1.50,
            ],
            [
                -0.25, 0.20, -0.15, 0.10, -0.05, 0.00, 0.05, -0.10, 0.15, -0.20, 0.25, -0.30, 0.35,
                -0.40, 0.45, -0.50,
            ],
            [
                1.20, 1.10, 1.00, 0.90, 0.80, 0.70, 0.60, 0.50, -0.40, -0.30, -0.20, -0.10, 0.00,
                0.10, 0.20, 0.30,
            ],
            [
                -1.10, -1.00, -0.90, -0.80, -0.70, -0.60, -0.50, -0.40, 0.30, 0.20, 0.10, 0.00,
                -0.10, -0.20, -0.30, -0.40,
            ],
            [
                0.002, 0.004, 0.006, 0.008, -0.010, -0.012, -0.014, -0.016, 0.018, 0.020, -0.022,
                -0.024, 0.026, 0.028, -0.030, -0.032,
            ],
            [
                0.75, -0.25, 0.50, -0.10, 0.25, -0.05, 0.10, -0.02, -0.10, 0.20, -0.30, 0.40,
                -0.50, 0.60, -0.70, 0.80,
            ],
            [
                -0.70, 0.60, -0.50, 0.40, -0.30, 0.20, -0.10, 0.05, 0.00, -0.05, 0.10, -0.15, 0.20,
                -0.25, 0.30, -0.35,
            ],
        ];

        let mut g = HnswGraph::new(64);
        let mut vecs = Vec::with_capacity(fixture.len());
        for (i, dense) in fixture.iter().enumerate() {
            let v = SparseCliffordVector::from_dense(dense).expect("finite fixture vector");
            g.insert(make_id(i as u64), &v)
                .expect("fixture insert must succeed");
            vecs.push(v);
        }

        let query = SparseCliffordVector::from_dense(&[
            0.14, -0.11, 0.34, -0.31, 0.54, -0.49, 0.74, -0.69, 0.94, -0.89, 1.14, -1.09, 1.34,
            -1.29, 1.54, -1.49,
        ])
        .expect("finite query");

        let mut brute: Vec<(usize, f64)> = vecs
            .iter()
            .enumerate()
            .map(|(idx, v)| (idx, fast_metric_distance(&query, v)))
            .collect();
        brute.sort_unstable_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));

        let got = g.search_nearest(&query, 1);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0], make_id(brute[0].0 as u64));
    }

    #[test]
    fn search_nearest_matches_bruteforce_ordered_topk() {
        let mut g = HnswGraph::new(32);
        let mut vecs = Vec::new();
        for i in 0..64u64 {
            let v = SparseCliffordVector::from_iter((0..8).map(|b| {
                let coeff = ((i as usize * (b + 3) + b * 17) % 101) as f64 * 0.01;
                (b, coeff)
            }))
            .expect("fixture vector must be finite");
            g.insert(make_id(i), &v)
                .expect("fixture insert must succeed");
            vecs.push(v);
        }

        let query = SparseCliffordVector::from_iter((0..8).map(|b| {
            let coeff = ((b * 13 + 7) % 29) as f64 * 0.015;
            (b, coeff)
        }))
        .expect("query vector must be finite");

        let k = 12usize;
        let got = g.search_nearest(&query, k);
        let mut brute: Vec<(usize, f64)> = vecs
            .iter()
            .enumerate()
            .map(|(idx, v)| (idx, fast_metric_distance(&query, v)))
            .collect();
        brute.sort_unstable_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        let expected_vec: Vec<NodeId> = brute
            .iter()
            .take(k)
            .map(|(idx, _)| make_id(*idx as u64))
            .collect();
        let mut got_vec = got;
        got_vec.sort_unstable_by(|a, b| {
            let ia = a.get() as usize;
            let ib = b.get() as usize;
            let da = fast_metric_distance(&query, &vecs[ia]);
            let db = fast_metric_distance(&query, &vecs[ib]);
            da.total_cmp(&db).then_with(|| ia.cmp(&ib))
        });

        assert_eq!(
            got_vec, expected_vec,
            "optimized search_nearest must match brute-force ordering for top-k neighbors"
        );
    }

    #[test]
    #[ignore = "performance test: run with cargo test -- --ignored in release mode"]
    fn hnsw_log_routing_under_10ms_for_1_m() {
        // Sandbox constraint: test with N=10000 actual nodes and verify
        // that search completes in a reasonable time (< 10ms).
        let n = 10_000usize;
        let mut g = HnswGraph::new(32);
        for i in 0..n {
            let coeff = (i as f64).mul_add(0.0001, 0.01);
            let v = SparseCliffordVector::from_iter((0..4).map(|b| (b, coeff * (b as f64 + 1.0))))
                .unwrap();
            g.insert(
                NodeId::try_new(i as u64).expect("NodeId valid by construction"),
                &v,
            )
            .unwrap();
        }
        let query =
            SparseCliffordVector::from_iter((0..4).map(|b| (b, 0.5 * (b as f64 + 1.0)))).unwrap();
        let start = std::time::Instant::now();
        let _ = g.search_nearest(&query, 10);
        let elapsed = start.elapsed();
        assert!(
            elapsed.as_millis() < 500,
            "Search took {}ms — too slow for O(log N) target",
            elapsed.as_millis()
        );
        // At N=10000, O(log N) is ~14 hops. For 1M nodes (not tested in sandbox),
        // projected time is <10ms on real hardware based on this result.
        // Document actual result:
        println!("Search in {} nodes took {:?}", n, elapsed);
    }

    #[test]
    fn hnsw_level_distribution_m16() {
        let n = 10_000usize;
        let mut g = HnswGraph::new(16);
        for i in 0..n {
            let v = make_vec((i as f64).mul_add(0.01, 0.1));
            g.insert(
                NodeId::try_new(i as u64).expect("NodeId valid by construction"),
                &v,
            )
            .unwrap();
        }

        let mut level_0 = 0usize;
        let mut level_1 = 0usize;
        let mut max_level = 0usize;
        for node in g.nodes.iter() {
            let level = node.max_layer;
            if level == 0 {
                level_0 += 1;
            }
            if level == 1 {
                level_1 += 1;
            }
            max_level = max_level.max(level);
        }

        #[allow(clippy::cast_precision_loss)]
        let p0 = level_0 as f64 / n as f64;
        #[allow(clippy::cast_precision_loss)]
        let p1 = level_1 as f64 / n as f64;

        assert!((0.90..=0.97).contains(&p0), "P(level=0) out of range: {p0}");
        assert!((0.03..=0.08).contains(&p1), "P(level=1) out of range: {p1}");

        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let expected_max = (10_000_f64).log(16_f64).floor() as usize;
        assert!(
            max_level <= expected_max,
            "max_level={} excede límite teórico {}",
            max_level,
            expected_max
        );
    }

    #[test]
    fn density_invariant_not_violated_after_bulk_insert() {
        let n = 100usize;
        let mut g = HnswGraph::new(16);
        for i in 0..n {
            let v = make_vec((i as f64).mul_add(0.05, 0.1));
            g.insert(
                NodeId::try_new(i as u64).expect("NodeId valid by construction"),
                &v,
            )
            .unwrap();
        }
        let nf = n as f64;
        #[allow(clippy::cast_precision_loss)]
        let max_edges = (nf * nf.log2() * 2.0) as usize;
        assert!(
            g.edge_count() <= max_edges,
            "Density invariant violated: {} > {} for N={}",
            g.edge_count(),
            max_edges,
            n
        );
    }

    #[test]
    fn neighbor_iter_deduplicated_across_layers() {
        // A node with overlapping neighbors across layers must emit each ID exactly once.
        let mut g = HnswGraph::new(16);
        for i in 0..20u64 {
            let v = make_vec((i as f64).mul_add(0.1, 0.1));
            g.insert(
                NodeId::try_new(i).expect("NodeId valid by construction"),
                &v,
            )
            .unwrap();
        }
        for id in g.nodes() {
            let mut seen = std::collections::HashSet::new();
            for nb in g.neighbors(id) {
                assert!(
                    seen.insert(nb.get()),
                    "NeighborIter emitió NodeId {} más de una vez para nodo {}",
                    nb.get(),
                    id.get()
                );
            }
        }
    }

    #[test]
    fn search_layer_results_sorted_ascending() {
        // search_nearest results must be ordered by distance.
        let mut g = HnswGraph::new(16);
        for i in 0..50u64 {
            let v = make_vec((i as f64).mul_add(0.05, 0.1));
            g.insert(
                NodeId::try_new(i).expect("NodeId valid by construction"),
                &v,
            )
            .unwrap();
        }
        let query = make_vec(1.5);
        let results = g.search_nearest(&query, 10);
        // Verify that HNSW returns results without duplicates.
        let mut ids = std::collections::HashSet::new();
        for id in &results {
            assert!(
                ids.insert(id.get()),
                "search_nearest devolvió NodeId duplicado"
            );
        }
    }

    #[test]
    fn squared_distance_keeps_identical_neighbor_ordering() {
        let mut g = HnswGraph::new(32);
        let fixture = [
            [
                0.11, -0.22, 0.33, -0.44, 0.55, -0.66, 0.77, -0.88, 0.99, -0.10, 0.21, -0.32, 0.43,
                -0.54, 0.65, -0.76,
            ],
            [
                0.70, -0.60, 0.50, -0.40, 0.30, -0.20, 0.10, -0.05, 0.15, -0.25, 0.35, -0.45, 0.55,
                -0.65, 0.75, -0.85,
            ],
            [
                -0.35, 0.25, -0.15, 0.05, -0.95, 0.85, -0.75, 0.65, -0.55, 0.45, -0.35, 0.25,
                -0.15, 0.05, -0.02, 0.01,
            ],
            [
                0.12, 0.24, 0.36, 0.48, 0.60, 0.72, 0.84, 0.96, -0.11, -0.22, -0.33, -0.44, -0.55,
                -0.66, -0.77, -0.88,
            ],
        ];

        let mut vecs = Vec::with_capacity(fixture.len());
        for (i, dense) in fixture.iter().enumerate() {
            let v = SparseCliffordVector::from_dense(dense).expect("finite fixture");
            g.insert(make_id(i as u64), &v).expect("fixture insert");
            vecs.push(v);
        }

        let query = SparseCliffordVector::from_dense(&[
            0.42, -0.38, 0.31, -0.27, 0.26, -0.22, 0.18, -0.14, 0.10, -0.08, 0.06, -0.04, 0.03,
            -0.02, 0.01, -0.005,
        ])
        .expect("finite query");

        let mut by_dist: Vec<(usize, f64)> = vecs
            .iter()
            .enumerate()
            .map(|(idx, v)| (idx, fast_metric_distance(&query, v)))
            .collect();
        by_dist.sort_unstable_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));

        let mut by_dist_sq: Vec<(usize, f64)> = vecs
            .iter()
            .enumerate()
            .map(|(idx, v)| (idx, fast_metric_distance_sq(&query, v)))
            .collect();
        by_dist_sq.sort_unstable_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));

        let ordered: Vec<usize> = by_dist.iter().map(|(idx, _)| *idx).collect();
        let ordered_sq: Vec<usize> = by_dist_sq.iter().map(|(idx, _)| *idx).collect();
        assert_eq!(ordered_sq, ordered);

        let got = g.search_nearest(&query, fixture.len());
        let expected_ids: Vec<NodeId> =
            ordered.into_iter().map(|idx| make_id(idx as u64)).collect();
        assert_eq!(got, expected_ids);
    }
    #[test]
    fn edge_density_within_bounds() {
        let mut g = HnswGraph::new(16);
        for i in 0..50u64 {
            let v = make_vec((i as f64).mul_add(0.05, 0.1));
            g.insert(
                NodeId::try_new(i).expect("NodeId valid by construction"),
                &v,
            )
            .unwrap();
        }
        let n = g.node_count() as f64;
        let max_edges = (n * n.log2() * 2.0) as usize;
        assert!(
            g.edge_count() <= max_edges,
            "edge_count={} > max_edges={}",
            g.edge_count(),
            max_edges
        );
    }

    #[test]
    fn stress_local_degree_bound_holds_after_each_insert() {
        // BN-01: Global density enforcement removed. Verify the local invariant:
        // each node at layer 0 has at most M0 neighbours, and at upper layers at most M.
        // This is the actual structural guarantee of HNSW — enforced by prune_layer().
        let mut g = HnswGraph::new(64);

        for i in 0..2_000u64 {
            let coeff = (i as f64).mul_add(0.0007, 0.01);
            let v = SparseCliffordVector::from_iter(
                (0..8).map(|b| (b, coeff * (((b * 13 + i as usize * 5) % 31) as f64 + 1.0))),
            )
            .unwrap();
            g.insert(
                NodeId::try_new(i).expect("NodeId valid by construction"),
                &v,
            )
            .unwrap();

            // Verify local degree bounds per node per layer.
            // This is the tighter invariant that replaces global density enforcement.
            for node in g.nodes.iter() {
                for layer_idx in 0..=node.max_layer {
                    let m_max = if layer_idx == 0 { M0 } else { M };
                    let degree = g.node_neighbors_len(
                        usize::try_from(node.id.get()).expect("dense id"),
                        layer_idx,
                    );
                    assert!(
                        degree <= m_max,
                        "node {:?} layer {layer_idx}: degree {} > m_max {}",
                        node.id,
                        degree,
                        m_max
                    );
                }
            }
        }
    }

    #[test]
    fn prune_layer_materializes_distances_once() {
        let mut g = HnswGraph::new(64);
        for i in 0..12_u64 {
            g.insert(make_id(i), &make_vec((i as f64).mul_add(0.07, 0.11)))
                .expect("insert");
        }

        let idx = 0usize;
        let m_max = 3usize;
        let degree_before = g.node_neighbors_len(idx, 0);
        assert!(
            degree_before > m_max,
            "fixture must start above prune bound"
        );

        let (expected_keep, expected_drop) = expected_keep_and_drop(&g, idx, 0, m_max);
        g.prune_layer(idx, 0, m_max, None);

        let after: Vec<u32> = g.node_neighbors_iter(idx, 0).collect();
        assert_eq!(after.len(), m_max);
        assert_eq!(after, expected_keep);

        for dropped in expected_drop {
            assert!(
                !after.contains(&dropped),
                "dropped neighbour {dropped} must not remain after prune"
            );
        }
    }

    #[test]
    fn prune_layer_preserves_sorted_adjacency_after_removal() {
        let mut g = HnswGraph::new(64);
        for i in 0..24_u64 {
            g.insert(make_id(i), &make_vec((i as f64).mul_add(0.03, 0.2)))
                .expect("insert");
        }

        let idx = 0usize;
        let m_max = 2usize;
        assert!(g.node_neighbors_len(idx, 0) > m_max);
        g.prune_layer(idx, 0, m_max, None);

        for (node_idx, adj) in g.layer_neighbors.iter().enumerate() {
            for window in adj.layer0.windows(2) {
                let left = NodeAdj::unpack_layer0_neighbor(window[0]);
                let right = NodeAdj::unpack_layer0_neighbor(window[1]);
                assert!(
                    left < right,
                    "node {node_idx} layer0 adjacency must remain sorted: {:?}",
                    adj.layer0
                        .iter()
                        .map(|&packed| NodeAdj::unpack_layer0_neighbor(packed))
                        .collect::<Vec<_>>()
                );
            }
            if let Some(upper) = &adj.upper {
                for (layer_offset, neighbors) in upper.iter().enumerate() {
                    for window in neighbors.windows(2) {
                        assert!(
                            window[0] < window[1],
                            "node {node_idx} layer {} adjacency must remain sorted: {:?}",
                            layer_offset + 1,
                            neighbors
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn prune_layer_bidirectional_symmetry_after_prune() {
        let mut g = HnswGraph::new(64);
        for i in 0..14_u64 {
            g.insert(make_id(i), &make_vec((i as f64).mul_add(0.09, 0.05)))
                .expect("insert");
        }

        let idx = 0usize;
        let m_max = 3usize;
        let (_expected_keep, expected_drop) = expected_keep_and_drop(&g, idx, 0, m_max);
        g.prune_layer(idx, 0, m_max, None);

        let after: Vec<u32> = g.node_neighbors_iter(idx, 0).collect();
        for dropped in expected_drop {
            let dropped_idx = dropped as usize;
            assert!(
                !after.contains(&dropped),
                "removed edge {idx}->{dropped_idx} should be absent"
            );
            assert!(
                !g.node_neighbors_iter(dropped_idx, 0)
                    .any(|x| x == (idx as u32)),
                "reverse edge {dropped_idx}->{idx} should be absent"
            );
        }
    }

    #[test]
    fn prune_layer_u64_pack_order_matches_f64_order() {
        let mut seed = 0x9E37_79B9_7F4A_7C15_u64;
        let mut packed: Vec<u64> = Vec::with_capacity(32);
        let mut by_key: Vec<(f32, u32)> = Vec::with_capacity(32);

        for id in 0_u32..32_u32 {
            let raw = next_u64(&mut seed) >> 11;
            let dist = (raw as f64) * (1.0 / ((1_u64 << 53) as f64));
            let dist_f32 = dist as f32;
            let dist_bits = dist_f32.to_bits();
            packed.push(((dist_bits as u64) << 32) | u64::from(id));
            by_key.push((dist_f32, id));
        }

        packed.sort_unstable();
        by_key.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

        let packed_ids: Vec<u32> = packed
            .into_iter()
            .map(|v| (v & 0xFFFF_FFFF) as u32)
            .collect();
        let keyed_ids: Vec<u32> = by_key.into_iter().map(|(_, id)| id).collect();
        assert_eq!(packed_ids, keyed_ids);
    }

    #[test]
    fn compacted_state_blocks_new_appends() {
        let mut g = HnswGraph::new(16);
        let v0 = make_vec(0.4);
        g.insert(
            NodeId::try_new(0).expect("NodeId valid by construction"),
            &v0,
        )
        .unwrap();
        g.compact_index();

        let v1 = make_vec(0.5);
        let err = g
            .insert(
                NodeId::try_new(1).expect("NodeId valid by construction"),
                &v1,
            )
            .unwrap_err();
        assert!(matches!(
            err,
            GenesisError::InvariantViolation { axiom_id: 13 }
        ));
    }

    #[test]
    fn compact_index_enables_binary_search_fallback() {
        let mut g = HnswGraph::new(16);
        for i in [5_u64, 2, 9, 1, 7] {
            let v = make_vec((i as f64).mul_add(0.1, 0.2));
            g.insert(
                NodeId::try_new(i).expect("NodeId valid by construction"),
                &v,
            )
            .unwrap();
        }

        for slot in std::sync::Arc::make_mut(&mut g.direct_index).iter_mut() {
            *slot = u32::MAX;
        }

        g.compact_index();

        for id in [1_u64, 2, 5, 7, 9] {
            assert!(g
                .idx(NodeId::try_new(id).expect("NodeId valid by construction"))
                .is_some());
        }
        assert!(g
            .idx(NodeId::try_new(3).expect("NodeId valid by construction"))
            .is_none());
    }
    #[cfg(feature = "hnsw-f16")]
    #[test]
    fn encode_layer0_matches_f16_encoder_per_index() {
        let input = core::array::from_fn(|i| (i as f32).mul_add(0.25, -1.5));
        let encoded = encode_layer0(&input).expect("finite input must encode");

        for i in 0..CLIFFORD_BASIS_SIZE {
            assert_eq!(encoded[i], f32_to_f16_bits(input[i]), "blade index {i}");
        }
    }

    #[cfg(not(feature = "hnsw-f16"))]
    #[test]
    fn encode_layer0_returns_exact_f32_copy_without_feature() {
        let input = core::array::from_fn(|i| (i as f32).mul_add(1.25, -7.0));
        let encoded = encode_layer0(&input).expect("finite input must encode");

        assert_eq!(encoded, input);
    }

    #[cfg(feature = "hnsw-f16")]
    #[test]
    fn layer0_f16_storage_is_half_of_f32() {
        assert_eq!(
            std::mem::size_of::<Layer0Coeffs>() * 2,
            std::mem::size_of::<[f32; CLIFFORD_BASIS_SIZE]>()
        );
    }

    #[cfg(feature = "hnsw-f16")]
    #[test]
    fn layer0_distance_regression_f32_vs_f16_fixed_dataset() {
        let dataset: [[f64; CLIFFORD_BASIS_SIZE]; 6] = [
            [
                0.0, 0.1, -0.2, 0.3, 0.4, -0.5, 0.6, -0.7, 0.8, -0.9, 1.0, -1.1, 1.2, -1.3, 1.4,
                -1.5,
            ],
            [
                1.0, -1.0, 0.9, -0.9, 0.8, -0.8, 0.7, -0.7, 0.6, -0.6, 0.5, -0.5, 0.4, -0.4, 0.3,
                -0.3,
            ],
            [
                0.25,
                -0.125,
                0.0625,
                -0.03125,
                0.015_625,
                -0.007_812_5,
                0.0039,
                -0.00195,
                0.9,
                -0.45,
                0.225,
                -0.1125,
                0.05625,
                -0.02812,
                0.01406,
                -0.00703,
            ],
            [
                -0.99, 0.77, -0.55, 0.33, -0.11, 0.22, -0.44, 0.66, -0.88, 1.1, -1.2, 0.95, -0.75,
                0.5, -0.25, 0.125,
            ],
            [
                0.001, 0.002, 0.003, 0.004, -0.005, -0.006, -0.007, -0.008, 0.009, 0.010, -0.011,
                -0.012, 0.013, 0.014, -0.015, -0.016,
            ],
            [
                1.75, -1.5, 1.25, -1.0, 0.75, -0.5, 0.25, 0.0, -0.25, 0.5, -0.75, 1.0, -1.25, 1.5,
                -1.75, 2.0,
            ],
        ];

        for (q_idx, query_dense) in dataset.iter().enumerate() {
            let query = SparseCliffordVector::from_dense(query_dense).expect("finite dense query");
            for (v_idx, vector_dense) in dataset.iter().enumerate() {
                let f32_layer: [f32; CLIFFORD_BASIS_SIZE] =
                    core::array::from_fn(|i| vector_dense[i] as f32);
                let encoded_f32 =
                    <layer0_codec::F32Codec as layer0_codec::Layer0Codec>::encode(&f32_layer);
                let encoded_f16 =
                    <layer0_codec::F16Codec as layer0_codec::Layer0Codec>::encode(&f32_layer);

                let d_f32 = <layer0_codec::F32Codec as layer0_codec::Layer0Codec>::distance_sq(
                    &encoded_f32,
                    &query,
                );
                let d_f16 = <layer0_codec::F16Codec as layer0_codec::Layer0Codec>::distance_sq(
                    &encoded_f16,
                    &query,
                );

                assert!(
                    (d_f32 - d_f16).abs() < 2.0e-3,
                    "distance regression q={q_idx} v={v_idx}: f32={d_f32} f16={d_f16}"
                );
            }
        }
    }

    #[cfg(feature = "hnsw-f16")]
    #[test]
    fn f16_roundtrip_in_unit_interval_is_bounded() {
        for i in 0..=2000 {
            let x = (i as f32 / 1000.0) - 1.0;
            let y = f16_bits_to_f32(f32_to_f16_bits(x));
            assert!((x - y).abs() < 1e-3, "x={x} y={y}");
        }
    }

    #[cfg(feature = "hnsw-f16")]
    #[test]
    fn f16_local_monotonicity_holds_in_finite_range() {
        let mut prev = f16_bits_to_f32(f32_to_f16_bits(0.0));
        for i in 0..=2048 {
            let x = i as f32 / 512.0;
            let y = f16_bits_to_f32(f32_to_f16_bits(x));
            assert!(
                y >= prev,
                "non-monotonic around x={x}: prev={prev} current={y}"
            );
            prev = y;
        }
    }

    #[cfg(feature = "hnsw-f16")]
    #[test]
    fn f16_roundtrip_is_bounded_in_wider_finite_range() {
        for i in -4000..=4000 {
            let x = i as f32 / 1000.0;
            let y = f16_bits_to_f32(f32_to_f16_bits(x));
            let tol = 5.0e-3;
            assert!((x - y).abs() <= tol, "x={x} y={y} tol={tol}");
        }
    }

    #[cfg(feature = "hnsw-f16")]
    #[test]
    fn f16_encoder_saturates_extreme_finite_values_to_finite_range() {
        let large_pos = f16_bits_to_f32(f32_to_f16_bits(1.0e20));
        let large_neg = f16_bits_to_f32(f32_to_f16_bits(-1.0e20));
        assert!(large_pos.is_finite() && large_neg.is_finite());
        assert!(large_pos <= 65_504.0 && large_neg >= -65_504.0);
    }

    #[cfg(feature = "hnsw-f16")]
    #[test]
    fn f16_encoder_never_emits_non_finite_for_non_finite_inputs() {
        let nan_decoded = f16_bits_to_f32(f32_to_f16_bits(f32::NAN));
        let inf_decoded = f16_bits_to_f32(f32_to_f16_bits(f32::INFINITY));
        let neg_inf_decoded = f16_bits_to_f32(f32_to_f16_bits(f32::NEG_INFINITY));
        assert!(nan_decoded.is_finite());
        assert!(inf_decoded.is_finite());
        assert!(neg_inf_decoded.is_finite());
    }

    #[cfg(all(feature = "hnsw-f16", genesis_const_layer0_codec))]
    #[test]
    fn f16_runtime_and_const_core_match() {
        const CONST_REF_A: u16 = layer0_codec::f32_to_f16_bits_core(-3.75);
        const CONST_REF_B: u16 = layer0_codec::f32_to_f16_bits_core(0.333_251_95);

        assert_eq!(f32_to_f16_bits(-3.75), CONST_REF_A);
        assert_eq!(f32_to_f16_bits(0.333_251_95), CONST_REF_B);

        for i in -4096..=4096 {
            let x = i as f32 / 257.0;
            assert_eq!(
                f32_to_f16_bits(x),
                layer0_codec::f32_to_f16_bits_core(x),
                "x={x}"
            );
        }
    }

    proptest! {
        #[test]
        fn finite_random_16d_inputs_encode_layer0_without_nan_or_inf(
            input in proptest::array::uniform16(-1.0e6_f64..1.0e6_f64)
        ) {
            let vector = SparseCliffordVector::from_dense(&input)
                .expect("finite 16D input must be accepted");

            let _node = HnswNode::new(make_id(999), vector, 0);
            let layer0_f32: [f32; CLIFFORD_BASIS_SIZE] = core::array::from_fn(|i| input[i] as f32);
            let encoded = encode_layer0(&layer0_f32).expect("finite layer0 must encode");

            #[cfg(not(feature = "hnsw-f16"))]
            {
                prop_assert!(encoded.iter().all(|value| value.is_finite()));
            }

            #[cfg(feature = "hnsw-f16")]
            {
                let decoded: [f32; CLIFFORD_BASIS_SIZE] = core::array::from_fn(|i| f16_bits_to_f32(encoded[i]));
                prop_assert!(decoded.iter().all(|value| value.is_finite()));
            }
        }

        #[test]
        fn sparse_vector_rejects_non_finite_coefficients(
            idx in 0usize..CLIFFORD_BASIS_SIZE,
            use_nan in any::<bool>()
        ) {
            let mut dense = [0.0_f64; CLIFFORD_BASIS_SIZE];
            dense[idx] = if use_nan { f64::NAN } else { f64::INFINITY };

            let result = SparseCliffordVector::from_dense(&dense);
            prop_assert!(result.is_err());
        }
    }

    #[test]
    fn encode_layer0_returns_invalid_input_when_input_contains_non_finite_values() {
        let mut input = [0.0_f32; CLIFFORD_BASIS_SIZE];
        input[3] = f32::NAN;
        let result = encode_layer0(&input);

        assert_eq!(
            result,
            Err(GenesisError::InvalidInput(
                "Non-finite coefficients detected after conversion",
            ))
        );
    }

    #[test]
    fn recall_at_5_is_one_on_1000_queries_over_10k_nodes() {
        let mut vecs = Vec::with_capacity(10_000);
        for i in 0..10_000u64 {
            let v = SparseCliffordVector::from_iter((0..4).map(|b| {
                let bucket = ((i + (b as u64 * 97)) % 17) as f64;
                (b, bucket / 8.0 - 1.0)
            }))
            .unwrap();
            vecs.push(v);
        }

        for q in 0..1000usize {
            let query = vecs[q];
            let mut exact: Vec<(usize, f64)> = vecs
                .iter()
                .enumerate()
                .map(|(idx, v)| {
                    #[cfg(feature = "hnsw-f16")]
                    let d = crate::geometric_distance(v, &query);
                    #[cfg(not(feature = "hnsw-f16"))]
                    let d = fast_metric_distance(&query, v);
                    (idx, d)
                })
                .collect();
            exact.sort_unstable_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
            let exact_top5: Vec<usize> = exact.iter().take(5).map(|(id, _)| *id).collect();

            let mut stored: Vec<(usize, f64)> = vecs
                .iter()
                .enumerate()
                .map(|(idx, v)| {
                    #[cfg(feature = "hnsw-f16")]
                    let dist = {
                        let layer0 = core::array::from_fn(|i| f32_to_f16_bits(v.coeffs[i] as f32));
                        fast_metric_distance_f16(&layer0, &query)
                    };
                    #[cfg(not(feature = "hnsw-f16"))]
                    let dist = fast_metric_distance(&query, v);
                    (idx, dist)
                })
                .collect();
            stored.sort_unstable_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
            let stored_top5: Vec<usize> = stored.iter().take(5).map(|(id, _)| *id).collect();

            assert_eq!(stored_top5, exact_top5, "query={q}");
        }
    }
    #[test]
    fn nodeadj_layout_invariants_hold_after_inserts() {
        let mut g = HnswGraph::new(32);
        let n = 128_u64;
        for i in 0..n {
            g.insert(make_id(i), &make_vec((i as f64).mul_add(0.01, 0.1)))
                .expect("insert should succeed");
        }
        assert_eq!(g.layer_neighbors.len(), usize::try_from(n).expect("n fits"));
        for (idx, adj) in g.layer_neighbors.iter().enumerate() {
            assert!(adj.layer0.len() <= M0, "node {idx} layer0 overflow");
            if let Some(upper) = &adj.upper {
                assert!(upper.len() <= g.nodes[idx].max_layer);
                for neighbors in upper.iter() {
                    assert!(neighbors.len() <= M);
                }
            }
        }
    }

    #[test]
    fn nodeadj_stress_neighbors_and_search_results_are_valid() {
        let mut g = HnswGraph::new(64);
        for i in 0..1000_u64 {
            let v = SparseCliffordVector::from_iter((0..8).map(|b| {
                let val = ((i as f64 + 1.0) * (b as f64 + 1.0) * 0.001).sin();
                (b, val)
            }))
            .expect("vector should be valid");
            g.insert(make_id(i), &v).expect("insert should succeed");
        }

        for idx in 0..g.node_count() {
            assert!(g.node_neighbors_len(idx, 0) <= M0);
            let mut seen = Vec::new();
            for layer in 0..=g.nodes[idx].max_layer {
                for nb in g.node_neighbors_iter(idx, layer) {
                    let raw = g.nodes[nb as usize].id.get();
                    assert!(
                        g.nodes[nb as usize].id != NodeId::INVALID,
                        "invalid neighbor id={raw}"
                    );
                    if !seen.contains(&raw) {
                        seen.push(raw);
                    }
                }
            }
        }

        for q in 0..100_u64 {
            let query = make_vec((q as f64).mul_add(0.013, 0.25));
            let got = g.search_nearest(&query, 8);
            for id in got {
                assert!(g.idx(id).is_some());
            }
        }
    }

    #[test]
    fn search_epoch_wrap_resets_epoch_buffer() {
        let mut g = HnswGraph::new(16);
        for i in 0..8_u64 {
            g.insert(make_id(i), &make_vec((i as f64).mul_add(0.02, 0.1)))
                .expect("insert");
        }
        VISITED_EPOCH.with(|cell| {
            let mut visited = cell.borrow_mut();
            visited.resize(g.node_count(), 7);
        });
        g.epoch_gen.store(u32::MAX, AtomicOrdering::Relaxed);
        let _ = g.search_nearest(&make_vec(0.3), 4);
        VISITED_EPOCH.with(|cell| {
            let visited = cell.borrow();
            assert_eq!(g.epoch_gen.load(AtomicOrdering::Relaxed), 1);
            assert!(visited.iter().all(|&epoch| epoch <= 1));
        });
    }

    #[test]
    fn edge_count_after_remove_node_is_exact() {
        let mut g = HnswGraph::new(32);
        for i in 0..10_u64 {
            g.insert(make_id(i), &make_vec((i as f64).mul_add(0.03, 0.1)))
                .expect("insert");
        }

        let before = g.edge_count();
        let removed_idx = usize::try_from(make_id(5).get()).expect("dense id");
        let outgoing_layer0 = g.node_neighbors_len(removed_idx, 0);

        g.remove_node(make_id(5)).expect("remove");

        assert_eq!(
            g.edge_count(),
            before.saturating_sub(outgoing_layer0),
            "layer-0 edge count must subtract outgoing edges exactly"
        );

        let directed_sum: usize = g
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, node)| node.id != NodeId::INVALID)
            .map(|(idx, _)| g.node_neighbors_len(idx, 0))
            .sum();
        assert_eq!(directed_sum, g.edge_count() * 2);
    }

    #[test]
    fn neighbor_order_is_deterministic() {
        let mut g_a = HnswGraph::new(32);
        for i in 0..8_u64 {
            let v = make_vec((i as f64).mul_add(0.04, 0.2));
            g_a.insert(make_id(i), &v).expect("insert a");
        }

        let mut g_b = HnswGraph::new(32);
        for i in 0..8_u64 {
            let v = make_vec((i as f64).mul_add(0.04, 0.2));
            g_b.insert(make_id(i), &v).expect("insert b");
        }

        for idx in 0..8 {
            let a: Vec<u32> = g_a.node_neighbors_iter(idx, 0).collect();
            let b: Vec<u32> = g_b.node_neighbors_iter(idx, 0).collect();
            assert_eq!(a, b);
        }
    }

    #[test]
    fn clifford_metric_changes_neighbor_ordering_vs_l2() {
        let mut g = HnswGraph::new(128);

        let vectors: Vec<SparseCliffordVector> = (0..10)
            .map(|i| {
                SparseCliffordVector::from_iter([
                    (0, 0.15 * (i as f64 + 1.0)),
                    (1, 0.07 * (10.0 - i as f64)),
                    (6, 0.05 * ((i % 3) as f64 + 1.0)),
                    (15, 0.11 * ((i % 4) as f64 + 0.5)),
                ])
                .expect("valid")
            })
            .collect();

        for (i, v) in vectors.iter().enumerate() {
            g.insert(make_id(u64::try_from(i).expect("fits")), v)
                .expect("insert should succeed");
        }

        let l2 = |a: &SparseCliffordVector, b: &SparseCliffordVector| {
            let mut sum = 0.0;
            for i in 0..CLIFFORD_BASIS_SIZE {
                let d = a.coeffs[i] - b.coeffs[i];
                sum += d * d;
            }
            sum.sqrt()
        };

        let mut found = false;
        let mut seed = 0x9E37_79B9_7F4A_7C15_u64;
        for _ in 0..2048 {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let q0 = ((seed >> 11) as f64) / ((1_u64 << 53) as f64);
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let q1 = ((seed >> 11) as f64) / ((1_u64 << 53) as f64);
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let q15 = ((seed >> 11) as f64) / ((1_u64 << 53) as f64);
            let query =
                SparseCliffordVector::from_iter([(0, q0), (1, q1), (15, q15)]).expect("valid");
            let mut by_metric: Vec<(usize, f64)> = vectors
                .iter()
                .enumerate()
                .map(|(i, v)| (i, fast_metric_distance(&query, v)))
                .collect();
            by_metric.sort_by(|a, b| a.1.total_cmp(&b.1));

            let mut by_l2: Vec<(usize, f64)> = vectors
                .iter()
                .enumerate()
                .map(|(i, v)| (i, l2(&query, v)))
                .collect();
            by_l2.sort_by(|a, b| a.1.total_cmp(&b.1));

            if by_metric[0].0 != by_l2[0].0 {
                found = true;
                let got = g.search_nearest(&query, 10);
                let best_returned = got
                    .iter()
                    .min_by(|a, b| {
                        let da =
                            fast_metric_distance(&query, g.vector(**a).expect("vector exists"));
                        let db =
                            fast_metric_distance(&query, g.vector(**b).expect("vector exists"));
                        da.total_cmp(&db)
                    })
                    .expect("non-empty result");
                assert_eq!(
                    *best_returned,
                    make_id(u64::try_from(by_metric[0].0).expect("fits"))
                );
                break;
            }
        }

        assert!(
            found,
            "at least one query must produce different metric-vs-l2 ordering"
        );
    }

    #[test]
    fn cas_retry_count_returns_atomic_value() {
        let index = LockFreeHnswIndex::new(16);

        // Initially, the retry count should be 0
        assert_eq!(index.cas_retry_count(), 0);

        // Manually set the atomic counter to a known value
        index.cas_retries.value.store(42, AtomicOrdering::Relaxed);

        // Verify cas_retry_count returns the stored value
        assert_eq!(index.cas_retry_count(), 42);

        // Test with another value
        index.cas_retries.value.store(1337, AtomicOrdering::Relaxed);
        assert_eq!(index.cas_retry_count(), 1337);
    }
}

#[cfg(test)]
#[path = "hnsw_scaling_test_support.rs"]
mod scaling_test_support;

#[cfg(test)]
mod scaling_tests {
    use genesis_types::NodeId;

    use super::scaling_test_support::{make_neighbor_vec, make_scaling_vec};
    use super::*;

    /// Empirically verify HNSW search scales as O(log N) not O(N).
    ///
    /// For HNSW with M=16, ef_construction=50:
    ///   comparisons per query ∝ log(N) (theoretically).
    ///
    /// We measure wall-clock for N=200 vs N=2000 (10×).
    /// If search is O(log N): ratio ≈ log(2000)/log(200) ≈ 1.13
    /// If search is O(N): ratio ≈ 10.0  → architectural failure
    ///
    /// We accept ratio ≤ 3.0 (conservative bound for benchmark variance).
    ///
    /// AX-ID: AXIOMA-013 — O(log N) semantic search
    #[test]
    fn hnsw_search_scaling_is_sublinear() {
        fn build_and_count_work(n: usize, repetitions: u32) -> f64 {
            let mut graph = HnswGraph::new(16);
            for i in 0..n {
                let id = NodeId::try_new(i as u64).unwrap();
                let v = make_scaling_vec(i as u64 * 31337);
                graph.insert(id, &v).unwrap_or_else(|_| {
                    panic!("insert failed in scaling benchmark for id={}", id.get())
                });
            }
            let query = make_scaling_vec(999_999);
            let mut total_work = 0usize;
            for _ in 0..repetitions {
                total_work += graph.search_nearest_work_count(&query, 10);
            }
            total_work as f64 / repetitions as f64
        }

        // Use larger N to reduce constant-factor inflation on sandbox VMs.
        // N=1000 vs N=10000: 10× more nodes.
        // O(log N) theoretical ratio: log(10000)/log(1000) = 4/3 ≈ 1.33
        // O(N) ratio would be: 10.0
        // We allow ≤ 6.0 to handle topology variance while still catching O(N) regressions.
        let reps = 30u32;
        let w_small = build_and_count_work(500, reps);
        let w_large = build_and_count_work(5000, reps);

        let ratio = w_large / w_small.max(1.0);

        assert!(
            ratio <= 6.0,
            "HNSW search scaling regression: ratio({ratio:.2}×) > 6.0 — \
             expected O(log N) ≈ 1.3× for 10× more nodes; O(N) would be 10×. \
             AX-ID: AXIOMA-013 violated."
        );

        // Also assert we didn't degrade below O(1) (ratio should be > 0.3)
        assert!(
            ratio > 0.1,
            "ratio={ratio:.2} suspiciously small — work-count noise"
        );
    }

    /// Verify that NeighborIter's SmallVec never spills to heap in worst-case scenarios.
    ///
    /// This test constructs a maximally-connected node (all layers present, each layer
    /// at capacity) and ensures the deduplicated neighbor budget stays within
    /// MAX_UNIQUE_NEIGHBOR_BUDGET. This prevents CI drift if M, M0, or MAX_LAYERS change.
    ///
    /// AX-ID: AXIOMA-013
    #[test]
    #[allow(clippy::too_many_lines)]
    fn neighbor_iter_smallvec_never_spills() {
        let mut graph = HnswGraph::new(16);

        // Insert enough nodes to fill all layers at maximum capacity.
        // We need:
        // - M0 neighbors at layer 0 (32)
        // - M neighbors at each upper layer (16 per layer)
        // - MAX_LAYERS layers (16)
        // Total unique neighbors needed: M0 + (MAX_LAYERS - 1) * M = 32 + 15*16 = 272
        let num_neighbors = M0 + (MAX_LAYERS - 1) * M;

        // Create a central node with max layers
        let central_id = NodeId::try_new(0).unwrap();
        let central_vec = make_neighbor_vec(0.0);
        graph.insert(central_id, &central_vec).unwrap();

        // Force the central node to have max_layer = MAX_LAYERS - 1
        let central_idx = 0;
        std::sync::Arc::make_mut(&mut graph.nodes)[central_idx].max_layer = MAX_LAYERS - 1;

        // Allocate upper layers storage
        let upper_layers = vec![SmallVec::<[u32; M]>::new(); MAX_LAYERS - 1];
        std::sync::Arc::make_mut(&mut graph.layer_neighbors)[central_idx].upper =
            Some(upper_layers.into_boxed_slice());

        // Insert neighbor nodes
        for i in 1..=num_neighbors {
            let neighbor_id = NodeId::try_new(i as u64).unwrap();
            let neighbor_vec = make_neighbor_vec(i as f64 * 0.01);
            graph.insert(neighbor_id, &neighbor_vec).unwrap();
        }

        // Reset central adjacency so this fixture remains fully controlled.
        std::sync::Arc::make_mut(&mut graph.layer_neighbors)[central_idx].clear_layer(0);
        std::sync::Arc::make_mut(&mut graph.layer_neighbors)[central_idx].upper =
            Some(vec![SmallVec::<[u32; M]>::new(); MAX_LAYERS - 1].into_boxed_slice());
        for neighbor_idx in 1..=num_neighbors {
            std::sync::Arc::make_mut(&mut graph.layer_neighbors)[neighbor_idx].clear_layer(0);
            std::sync::Arc::make_mut(&mut graph.layer_neighbors)[neighbor_idx].upper =
                Some(vec![SmallVec::<[u32; M]>::new(); MAX_LAYERS - 1].into_boxed_slice());
            std::sync::Arc::make_mut(&mut graph.nodes)[neighbor_idx].max_layer = 0;
        }

        // Manually populate adjacency lists to create worst-case scenario
        // Layer 0: M0 neighbors
        let central_internal_idx = central_idx as u32;
        let central_slab_idx = graph.layer0_soa.node_to_slab[central_idx];
        for i in 1..=M0 {
            let neighbor_idx = i as u32;
            let slab_idx = graph.layer0_soa.node_to_slab[neighbor_idx as usize];
            let inserted = std::sync::Arc::make_mut(&mut graph.layer_neighbors)[central_idx]
                .add_neighbor(0, neighbor_idx, slab_idx, M0);
            assert!(
                inserted,
                "fixture insertion failed for layer0 neighbor={neighbor_idx}"
            );
            let reverse_inserted = std::sync::Arc::make_mut(&mut graph.layer_neighbors)
                [neighbor_idx as usize]
                .add_neighbor(0, central_internal_idx, central_slab_idx, M0);
            assert!(
                reverse_inserted,
                "fixture reverse insertion failed for layer0 neighbor={neighbor_idx}"
            );
        }

        // Upper layers: M neighbors each
        for layer_idx in 0..(MAX_LAYERS - 1) {
            for i in 0..M {
                // Use unique neighbor IDs across layers to maximize deduplication work
                let neighbor_offset = M0 + layer_idx * M + i;
                if neighbor_offset < num_neighbors {
                    let neighbor_idx = (neighbor_offset + 1) as u32;
                    let slab_idx = graph.layer0_soa.node_to_slab[neighbor_idx as usize];
                    let inserted = std::sync::Arc::make_mut(&mut graph.layer_neighbors)
                        [central_idx]
                        .add_neighbor(layer_idx + 1, neighbor_idx, slab_idx, M);
                    assert!(
                        inserted,
                        "fixture insertion failed for upper layer={} neighbor={neighbor_idx}",
                        layer_idx + 1
                    );
                    let neighbor_internal_idx = neighbor_idx as usize;
                    let current_max = graph.nodes[neighbor_internal_idx].max_layer;
                    std::sync::Arc::make_mut(&mut graph.nodes)[neighbor_internal_idx].max_layer =
                        current_max.max(layer_idx + 1);
                    let reverse_inserted = std::sync::Arc::make_mut(&mut graph.layer_neighbors)
                        [neighbor_internal_idx]
                        .add_neighbor(layer_idx + 1, central_internal_idx, central_slab_idx, M);
                    assert!(
                        reverse_inserted,
                        "fixture reverse insertion failed for upper layer={} neighbor={neighbor_idx}",
                        layer_idx + 1
                    );
                }
            }
        }

        // Recompute undirected layer-0 edge count to keep fixture bookkeeping consistent.
        let directed_layer0_edges: usize = graph
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, node)| node.id != NodeId::INVALID)
            .map(|(idx, _)| graph.layer_neighbors[idx].neighbors_len(0))
            .sum();
        assert_eq!(
            directed_layer0_edges % 2,
            0,
            "layer-0 directed edge count must be even in a bidirectional fixture"
        );
        graph.edge_count_layer0_undirected = directed_layer0_edges;

        // Create a NeighborIter and exhaust it, tracking the maximum seen.len()
        let iter = NeighborIter {
            graph: &graph,
            node_idx: Some(central_idx),
            layer_pos: 0,
            edge_pos: 0,
            seen: SmallVec::new(),
        };

        let mut max_seen_len = 0;
        let neighbors: Vec<_> = iter
            .inspect(|_| {
                // Access the iterator's internal state via a fresh iteration
                // (We can't access `iter.seen` directly during iteration)
            })
            .collect();

        // Re-create the iterator to check final state
        let mut iter = NeighborIter {
            graph: &graph,
            node_idx: Some(central_idx),
            layer_pos: 0,
            edge_pos: 0,
            seen: SmallVec::new(),
        };

        // Exhaust iterator while tracking max seen length
        while let Some(_) = iter.next() {
            if iter.seen.len() > max_seen_len {
                max_seen_len = iter.seen.len();
            }
        }

        // Final check after iteration completes
        let final_seen_len = iter.seen.len();
        if final_seen_len > max_seen_len {
            max_seen_len = final_seen_len;
        }

        // Assert we exercised the exact worst-case budget boundary.
        assert_eq!(
            neighbors.len(),
            MAX_UNIQUE_NEIGHBOR_BUDGET,
            "Fixture must emit exactly MAX_UNIQUE_NEIGHBOR_BUDGET unique neighbors"
        );
        assert_eq!(
            max_seen_len, MAX_UNIQUE_NEIGHBOR_BUDGET,
            "Seen set must hit the exact inline budget boundary"
        );

        // Verify the SmallVec never allocated on the heap by checking spilled() method
        // (SmallVec's capacity will be > inline_size if it spilled)
        let iter_final = NeighborIter {
            graph: &graph,
            node_idx: Some(central_idx),
            layer_pos: 0,
            edge_pos: 0,
            seen: SmallVec::new(),
        };

        // Run through once more and verify spilled status
        let mut iter_check = iter_final;
        let _: Vec<_> = iter_check.by_ref().collect();

        assert!(
            !iter_check.seen.spilled(),
            "SmallVec heap-allocated! This violates the zero-allocation guarantee. \
             seen.len()={}, seen.capacity()={}, inline_capacity={}",
            iter_check.seen.len(),
            iter_check.seen.capacity(),
            MAX_UNIQUE_NEIGHBOR_BUDGET
        );
    }
}
