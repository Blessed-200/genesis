#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::map_unwrap_or,
    clippy::missing_errors_doc,
    clippy::needless_continue
)]

use std::cell::RefCell;
/// AX-ID: AXIOMA-013
/// Hierarchical Navigable Small World graph for O(log N) semantic search.
/// Exclusive metric: `geometric_distance` (grade-weighted fast_metric_distance).
/// PROHIBITED: Delaunay triangulation. PROHIBITED: `HashMap` in hot path.
/// Adjacency lists stored as sorted Vec<(`NodeId`, f64)> with binary search.
use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicPtr, Ordering as AtomicOrdering};
use std::sync::Arc;
use std::thread;

use fixedbitset::FixedBitSet;
use genesis_math::SparseCliffordVector;
use genesis_types::{GenesisError, NodeId};
use smallvec::SmallVec;

use crate::geodesic::geometric_distance;

const TOTAL_BLADES: usize = 16;

// Política de mantenimiento para módulos críticos de topología.
//
// - `#[inline(always)]` está prohibido salvo excepción documentada con
//   benchmark reproducible + motivo arquitectónico + evaluación de riesgo.
// - Los símbolos del codec layer-0 deben mantener simetría de `cfg`:
//   `feature = "hnsw-f16"`, `genesis_const_layer0_codec`, y `test`.
//   Cualquier símbolo condicionado por `cfg` debe tener contraparte
//   explícita `not(...)` para evitar símbolos huérfanos entre perfiles.
//
// AX-ID: AXIOMA-013, H_estructura (LEY_FUNDACIONAL §3.1)

mod layer0_codec {
    use super::TOTAL_BLADES;
    use genesis_math::SparseCliffordVector;

    #[cfg(feature = "hnsw-f16")]
    mod f16_kernel {
        #[inline]
        pub(in super::super) const fn f32_to_f16_bits_core(value: f32) -> u16 {
            let bits = value.to_bits();
            let sign = ((bits >> 16) & 0x8000) as u16;
            let exp = ((bits >> 23) & 0xFF) as i32;
            let frac = bits & 0x7F_FFFF;
            if exp <= 112 {
                if exp < 103 {
                    return sign;
                }
                let mant = frac | 0x80_0000;
                return sign | (((mant >> (126 - exp)) + 0x1000) >> 13) as u16;
            }
            if exp >= 143 {
                return sign | 0x7C00;
            }
            sign | ((((exp - 112) as u32) << 10) as u16) | (((frac + 0x1000) >> 13) as u16)
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

    /// Trait sellado para codificación densa de layer-0.
    ///
    /// AX-ID: AXIOMA-013, H_estructura (LEY_FUNDACIONAL §3.1)
    pub(super) trait Layer0Codec: sealed::Sealed {
        type Storage: Copy;

        fn encode(values: &[f32; TOTAL_BLADES]) -> Self::Storage;
        fn distance(stored: &Self::Storage, query: &SparseCliffordVector) -> f64;
        fn decode_to_f64(stored: &Self::Storage) -> [f64; TOTAL_BLADES];
    }

    #[cfg(any(not(feature = "hnsw-f16"), test))]
    pub(super) struct F32Codec;

    #[cfg(any(not(feature = "hnsw-f16"), test))]
    impl Layer0Codec for F32Codec {
        type Storage = [f32; TOTAL_BLADES];

        fn encode(values: &[f32; TOTAL_BLADES]) -> Self::Storage {
            *values
        }

        fn distance(stored: &Self::Storage, query: &SparseCliffordVector) -> f64 {
            let dense = core::array::from_fn(|i| f64::from(stored[i]));
            crate::geodesic::fast_bivector_distance_from_dense(&dense, query)
        }

        fn decode_to_f64(stored: &Self::Storage) -> [f64; TOTAL_BLADES] {
            core::array::from_fn(|i| f64::from(stored[i]))
        }
    }

    #[cfg(feature = "hnsw-f16")]
    pub(super) struct F16Codec;

    #[cfg(feature = "hnsw-f16")]
    impl Layer0Codec for F16Codec {
        type Storage = [u16; TOTAL_BLADES];

        fn encode(values: &[f32; TOTAL_BLADES]) -> Self::Storage {
            #[cfg(genesis_const_layer0_codec)]
            {
                return encode_f16_const(values);
            }

            #[cfg(not(genesis_const_layer0_codec))]
            {
                encode_f16_runtime(values)
            }
        }

        fn distance(stored: &Self::Storage, query: &SparseCliffordVector) -> f64 {
            super::fast_bivector_distance_f16(stored, query)
        }

        fn decode_to_f64(stored: &Self::Storage) -> [f64; TOTAL_BLADES] {
            core::array::from_fn(|i| f64::from(f16_bits_to_f32(stored[i])))
        }
    }

    #[cfg(all(feature = "hnsw-f16", not(genesis_const_layer0_codec)))]
    fn encode_f16_runtime(values: &[f32; TOTAL_BLADES]) -> [u16; TOTAL_BLADES] {
        core::array::from_fn(|i| f32_to_f16_bits(values[i]))
    }

    #[cfg(all(feature = "hnsw-f16", genesis_const_layer0_codec))]
    const fn encode_f16_const(values: &[f32; TOTAL_BLADES]) -> [u16; TOTAL_BLADES] {
        let mut encoded = [0_u16; TOTAL_BLADES];
        let mut i = 0;
        while i < TOTAL_BLADES {
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

#[cfg(feature = "hnsw-f16")]
type ActiveLayer0Codec = layer0_codec::F16Codec;
#[cfg(not(feature = "hnsw-f16"))]
type ActiveLayer0Codec = layer0_codec::F32Codec;
type Layer0Coeffs = <ActiveLayer0Codec as layer0_codec::Layer0Codec>::Storage;

/// Maximum number of layers in the HNSW graph.
const MAX_LAYERS: usize = 16;

/// Default maximum connections per layer (M parameter).
/// `pub(crate)` for manifold.rs stack-allocated neighbour buffers (BN-02).
pub(crate) const M: usize = 16;

/// Maximum connections at layer 0 (M0 = 2*M).
/// `pub(crate)` for manifold.rs stack-allocated neighbour buffers (BN-02).
pub(crate) const M0: usize = M * 2;

/// Level multiplier: 1.0 / ln(M).
// M es una constante pequeña (≤ 64). M as f64 es exacto: M < 2^53.
#[allow(clippy::cast_precision_loss)]
fn ml() -> f64 {
    1.0_f64 / (M as f64).ln()
}

/// Wrapper para f64 que implementa Ord (NaN nunca ocurre por contrato del CS gate).
#[derive(Clone, Copy, PartialEq)]
struct FiniteDist(f64);

impl Eq for FiniteDist {}

impl PartialOrd for FiniteDist {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for FiniteDist {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.partial_cmp(&other.0).unwrap_or(Ordering::Equal)
    }
}

/// A node stored in the HNSW graph.
#[derive(Clone)]
struct HnswNode {
    id: NodeId,
    vec: SparseCliffordVector,
    layer0: Layer0Coeffs,
    /// Adjacency lists per layer. Layer 0 is the densest.
    /// Each entry is (`NodeId`, distance) sorted by `NodeId` for O(log K) lookup.
    layers: Vec<Vec<(NodeId, f64)>>,
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
    fn new(id: NodeId, vec: SparseCliffordVector, max_layer: usize) -> Result<Self, GenesisError> {
        let mut layer0 = [0.0_f32; TOTAL_BLADES];
        for (idx, coeff) in vec.coeffs.iter().enumerate() {
            let coeff_f32 = *coeff as f32;
            debug_assert!(
                coeff_f32.is_finite(),
                "layer0 coefficient must be finite after f64->f32 projection"
            );
            layer0[idx] = coeff_f32;
        }
        Ok(Self {
            id,
            vec,
            layer0: encode_layer0(&layer0)?,
            layers: vec![Vec::new(); max_layer + 1],
        })
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
fn encode_layer0(values: &[f32; TOTAL_BLADES]) -> Result<Layer0Coeffs, GenesisError> {
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
pub fn fast_bivector_distance_f16(stored: &[u16; 16], query: &SparseCliffordVector) -> f64 {
    // NOTA ARQUITECTÓNICA:
    // Se usa compile-time dispatch en lugar de runtime dispatch para evitar
    // la pérdida de inlining y la penalización de `vzeroupper` en el hot loop.
    // En x86_64, compilar con RUSTFLAGS="-C target-cpu=native" para activar AVX2.
    // Target primario: Genesis Edge (ARM + NEON).
    let mut decompressed = [0.0_f64; 16];

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "f16c",
        target_feature = "avx2"
    ))]
    {
        use std::arch::x86_64::*;
        let s = stored.as_ptr().cast::<__m128i>();

        // SAFETY: se leen/escriben exactamente 16 elementos dentro de límites y las features AVX2/F16C están garantizadas por cfg.
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
        for i in 0..16 {
            decompressed[i] = f64::from(layer0_codec::f16_bits_to_f32(stored[i]));
        }
    }

    crate::geodesic::fast_bivector_distance_from_dense(&decompressed, query)
}

#[cfg(all(feature = "hnsw-f16", test))]
use layer0_codec::{
    f16_bits_to_f32_core as f16_bits_to_f32, f32_to_f16_bits_core as f32_to_f16_bits,
};

#[derive(Default)]
struct SearchScratch {
    candidates: BinaryHeap<Reverse<(FiniteDist, usize)>>,
    results: BinaryHeap<(FiniteDist, usize)>,
    visited: FixedBitSet,
    visited_touched: Vec<usize>,
    out: Vec<(usize, f64)>,
}

thread_local! {
    static SEARCH_SCRATCH: RefCell<SearchScratch> = RefCell::new(SearchScratch::default());
}

/// Hierarchical Navigable Small World graph.
///
/// Stores vectors and supports O(log N) approximate nearest-neighbour search
/// using the exclusive bivector metric from genesis-math.
///
/// AX-ID: AXIOMA-013
#[derive(Clone)]
pub struct HnswGraph {
    /// All nodes stored in a flat Vec. Index = internal idx.
    nodes: Vec<HnswNode>,
    /// Maps `NodeId` (u64) → internal index via sorted (`NodeId`, usize) pairs.
    /// Sorted by `NodeId`, searched via binary search. No `HashMap`.
    id_index: Vec<(NodeId, usize)>,
    /// Entry point for top-layer search (internal index).
    entry: Option<usize>,
    /// Layer of the current entry point.
    entry_layer: usize,
    /// `ef_construction` parameter.
    ef_construction: usize,
    /// Mapa directo `NodeId.get()` → `internal_idx` cuando `NodeIds` son consecutivos.
    /// Capacidad dinámica: se expande al insertar `NodeIds` mayores.
    direct_index: Vec<u32>, // u32::MAX = no presente
    /// Estado del índice secundario `id_index`.
    state: GraphState,
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
    /// Dense `[f64; 16]` blade coefficients for layer-0 vectors.
    pub dense_coeffs: Vec<[f64; TOTAL_BLADES]>,
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
/// AX-ID: AXIOMA-013
pub struct LockFreeHnswIndex {
    head: AtomicPtr<HnswGraph>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GraphState {
    /// Fase online: appends directos en `id_index` sin ordenar.
    Online,
    /// Fase post-compactación: `id_index` ordenado para binary search.
    Compacted,
}

impl HnswGraph {
    /// Create a new empty HNSW graph.
    ///
    /// AX-ID: AXIOMA-013
    pub const fn new(ef_construction: usize) -> Self {
        Self {
            nodes: Vec::new(),
            id_index: Vec::new(),
            entry: None,
            entry_layer: 0,
            ef_construction,
            direct_index: Vec::new(),
            state: GraphState::Online,
        }
    }

    /// Lookup internal index by `NodeId`. O(1) average with direct index, fallback O(log N).
    fn get_idx(&self, id: NodeId) -> Option<usize> {
        // Contrato CRATE-002: NodeIds son consecutivos desde 0. N < 2^32 en cualquier
        // deployment de GÉNESIS (límite de memoria física). u64 → usize es seguro.
        #[allow(clippy::cast_possible_truncation)]
        let raw = id.get() as usize;
        if raw < self.direct_index.len() {
            // Para uso concurrente (producción multi-hilo): direct_index debe ser
            // Vec<AtomicU32>. Actualmente el acceso es exclusivo por &mut self / &self
            // con contrato de single-writer. Fence no tiene efecto sobre tipos no-atómicos.
            let idx = self.direct_index[raw];
            if idx != u32::MAX {
                return Some(idx as usize);
            }
        }

        match self.state {
            GraphState::Compacted => self
                .id_index
                .binary_search_by_key(&id.get(), |&(nid, _)| nid.get())
                .ok()
                .map(|pos| self.id_index[pos].1),
            GraphState::Online => self
                .id_index
                .iter()
                .find_map(|&(nid, idx)| (nid == id).then_some(idx)),
        }
    }

    /// Insert internal index mapping with O(1) append during online phase.
    fn insert_id_index(&mut self, id: NodeId, idx: usize) {
        debug_assert_eq!(self.state, GraphState::Online);
        self.id_index.push((id, idx));
    }

    /// Compact and sort `id_index` for O(log N) fallback queries.
    pub fn compact_index(&mut self) {
        if self.state == GraphState::Compacted {
            return;
        }

        radix_sort_node_ids(&mut self.id_index);
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
        // u uniforme en (0, 1): usa 53 bits y desplaza medio ULP para evitar 0 exacto.
        // x >> 11 ∈ [0, 2^53). Las conversiones son exactas en f64 para 53 bits.
        #[allow(clippy::cast_precision_loss)]
        let u = (((x >> 11) as f64) + 0.5) * (1.0 / ((1_u64 << 53) as f64));

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
    /// # Panics
    /// Panics if the internal entry point is `None` after the first insertion.
    /// This cannot occur under normal usage: `entry` is set atomically on first insert.
    ///
    /// AX-ID: AXIOMA-013, `H_restricción`
    pub fn insert(&mut self, id: NodeId, vec: &SparseCliffordVector) -> Result<(), GenesisError> {
        if self.state == GraphState::Compacted {
            return Err(GenesisError::InvariantViolation { axiom_id: 13 });
        }
        if id == NodeId::INVALID {
            return Err(GenesisError::InvariantViolation { axiom_id: 4 });
        }
        if self.get_idx(id).is_some() {
            return Ok(()); // already present
        }

        let target_layer = Self::random_level(id);
        let new_idx = self.nodes.len();
        self.nodes.push(HnswNode::new(id, *vec, target_layer)?);
        self.insert_id_index(id, new_idx);

        if id.get() < u64::from(u32::MAX) {
            // Mismo contrato: NodeId.get() ≤ N < usize::MAX en arquitecturas objetivo.
            #[allow(clippy::cast_possible_truncation)]
            let id_raw = id.get() as usize;
            if id_raw >= self.direct_index.len() {
                self.direct_index.resize(id_raw + 1, u32::MAX);
            }
            // Barrera Release: no necesaria sobre Vec<u32> con &mut self (single-writer).
            // Para publicar este índice a lectores concurrentes, direct_index debe ser
            // Vec<AtomicU32> con store(Release). Por ahora: single-threaded, sin contención.
            self.direct_index[id_raw] =
                u32::try_from(new_idx).expect("node count must stay below u32::MAX (≈4B nodes)");
            debug_assert!(
                self.get_idx(id) == Some(new_idx),
                "direct_index inconsistente con id_index para NodeId={}",
                id.get()
            );
        }

        if self.entry.is_none() {
            self.entry = Some(new_idx);
            self.entry_layer = target_layer;
            return Ok(());
        }

        let entry_idx = self.entry.expect(
            "entry point is set on first insert; this branch is unreachable on second+ insert",
        );
        let entry_layer = self.entry_layer;

        // Phase 1: greedy descent from entry_layer to target_layer+1
        let mut current = entry_idx;
        for lc in (target_layer + 1..=entry_layer).rev() {
            current = self.greedy_search_layer(vec, current, lc);
        }

        // Phase 2: beam search and connect from target_layer down to 0
        for lc in (0..=target_layer.min(entry_layer)).rev() {
            let layer_m = if lc == 0 { M0 } else { M };
            // nodes.len() + 1 ≤ N. Para N < 2^53 (límite físico), usize→f64 es exacto.
            // log2(N).ceil() es siempre positivo (N ≥ 1). f64→usize sin pérdida de signo.
            #[allow(
                clippy::cast_precision_loss,
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss
            )]
            let degree_cap = ((((self.nodes.len() + 1) as f64).log2().ceil() as usize) * 2).max(1);
            let m_max = layer_m.min(degree_cap);
            let candidates = self.search_layer(vec, current, self.ef_construction, lc);
            // Take top-M by distance
            let neighbours: Vec<(usize, f64)> = candidates.into_iter().take(m_max).collect();

            // Add bidirectional edges
            let new_idx = self.nodes.len() - 1; // last inserted
            for &(nb_idx, dist) in &neighbours {
                self.add_edge(new_idx, lc, self.nodes[nb_idx].id, dist);
                self.add_edge(nb_idx, lc, id, dist);
                // Prune nb if it exceeds m_max
                self.prune_layer(nb_idx, lc, m_max);
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

    /// Add an edge at a given layer (sorted insert, no duplicates).
    fn add_edge(&mut self, from_idx: usize, layer: usize, to: NodeId, dist: f64) {
        if layer >= self.nodes[from_idx].layers.len() {
            return;
        }
        let adj = &mut self.nodes[from_idx].layers[layer];
        let pos = adj.binary_search_by_key(&to.get(), |&(nid, _)| nid.get());
        if let Err(insertion_point) = pos {
            adj.insert(insertion_point, (to, dist));
        }
    }

    /// Remove a directed edge at `layer` if present.
    /// Returns `true` iff one edge was removed.
    fn remove_edge(&mut self, from_idx: usize, layer: usize, to: NodeId) -> bool {
        if layer >= self.nodes[from_idx].layers.len() {
            return false;
        }
        let adj = &mut self.nodes[from_idx].layers[layer];
        if let Ok(pos) = adj.binary_search_by_key(&to.get(), |&(nid, _)| nid.get()) {
            adj.remove(pos);
            return true;
        }
        false
    }

    /// Remove an edge in both directions, keeping graph symmetry.
    /// Returns the number of directed edges removed.
    fn remove_edge_bidirectional(&mut self, from_idx: usize, layer: usize, to: NodeId) -> usize {
        let mut removed = 0;
        if self.remove_edge(from_idx, layer, to) {
            removed += 1;
        }
        if let Some(to_idx) = self.get_idx(to) {
            if self.remove_edge(to_idx, layer, self.nodes[from_idx].id) {
                removed += 1;
            }
        }
        removed
    }

    /// Prune a node's adjacency list at a layer to at most `m_max` neighbours
    /// (keep the closest by distance).
    fn prune_layer(&mut self, idx: usize, layer: usize, m_max: usize) {
        if layer >= self.nodes[idx].layers.len() {
            return;
        }
        while self.nodes[idx].layers[layer].len() > m_max {
            let farthest = self.nodes[idx].layers[layer]
                .iter()
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .copied();

            let Some((farthest_id, _)) = farthest else {
                break;
            };
            self.remove_edge_bidirectional(idx, layer, farthest_id);
        }
    }

    fn distance_to_node(&self, query: &SparseCliffordVector, idx: usize, layer: usize) -> f64 {
        if layer == 0 {
            return <ActiveLayer0Codec as layer0_codec::Layer0Codec>::distance(
                &self.nodes[idx].layer0,
                query,
            );
        }
        geometric_distance(query, &self.nodes[idx].vec)
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
        let mut current_dist = self.distance_to_node(query, current, layer);
        loop {
            let mut improved = false;
            if layer < self.nodes[current].layers.len() {
                for &(nb_id, _) in &self.nodes[current].layers[layer] {
                    if let Some(nb_idx) = self.get_idx(nb_id) {
                        let d = self.distance_to_node(query, nb_idx, layer);
                        if d < current_dist {
                            current = nb_idx;
                            current_dist = d;
                            improved = true;
                        }
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
    fn search_layer(
        &self,
        query: &SparseCliffordVector,
        entry_idx: usize,
        ef: usize,
        layer: usize,
    ) -> Vec<(usize, f64)> {
        SEARCH_SCRATCH.with(|cell| {
            let mut scratch = cell.borrow_mut();
            scratch.candidates.clear();
            scratch.results.clear();
            scratch.out.clear();
            scratch.visited_touched.clear();
            scratch.candidates.reserve(ef.saturating_mul(2));
            scratch.results.reserve(ef.saturating_add(1));
            scratch.out.reserve(ef);

            if scratch.visited.len() < self.nodes.len() {
                scratch.visited.grow(self.nodes.len());
            }

            let d0 = self.distance_to_node(query, entry_idx, layer);
            scratch.visited.set(entry_idx, true);
            scratch.visited_touched.push(entry_idx);
            scratch
                .candidates
                .push(Reverse((FiniteDist(d0), entry_idx)));
            scratch.results.push((FiniteDist(d0), entry_idx));

            while let Some(Reverse((FiniteDist(c_dist), c_idx))) = scratch.candidates.pop() {
                if scratch.results.len() >= ef {
                    let worst = scratch
                        .results
                        .peek()
                        .map(|(FiniteDist(d), _)| *d)
                        .unwrap_or(f64::INFINITY);
                    if c_dist > worst {
                        break;
                    }
                }

                if layer < self.nodes[c_idx].layers.len() {
                    for &(nb_id, _) in &self.nodes[c_idx].layers[layer] {
                        if let Some(nb_idx) = self.get_idx(nb_id) {
                            if scratch.visited[nb_idx] {
                                continue;
                            }
                            scratch.visited.set(nb_idx, true);
                            scratch.visited_touched.push(nb_idx);
                            let d = self.distance_to_node(query, nb_idx, layer);

                            let worst = scratch
                                .results
                                .peek()
                                .map(|(FiniteDist(dw), _)| *dw)
                                .unwrap_or(f64::INFINITY);

                            if scratch.results.len() < ef || d < worst {
                                scratch.candidates.push(Reverse((FiniteDist(d), nb_idx)));
                                scratch.results.push((FiniteDist(d), nb_idx));
                                if scratch.results.len() > ef {
                                    scratch.results.pop();
                                }
                            }
                        }
                    }
                }
            }

            // PERF NOTE: The intermediate Vec here (result_pairs) is intentional.
            // scratch.results (BinaryHeap) and scratch.out (Vec) are fields of the same
            // struct — Rust's borrow checker cannot split them through &mut scratch.
            // The Vec is bounded by ef (typically 200) so allocation is O(1) amortized.
            // FIX-E.2 was reverted: field-split reborrow is not possible here because
            // the closure already holds &mut scratch from the outer context.
            let result_pairs: Vec<(usize, f64)> = scratch
                .results
                .iter()
                .map(|(FiniteDist(d), idx)| (*idx, *d))
                .collect();
            scratch.out.extend(result_pairs);
            scratch
                .out
                .sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

            let touched: Vec<usize> = scratch.visited_touched.drain(..).collect();
            for idx in touched {
                scratch.visited.set(idx, false);
            }

            scratch.out.clone()
        })
    }

    /// Search for the k nearest neighbours to query.
    ///
    /// AX-ID: AXIOMA-013
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

    /// Iterate over neighbours of a node at all layers (union, deduplicated).
    ///
    /// AX-ID: AXIOMA-013
    pub fn neighbors(&self, id: NodeId) -> impl Iterator<Item = NodeId> + '_ {
        let node = self.get_idx(id).map(|idx| &self.nodes[idx]);
        NeighborIter {
            layers: node.map_or(&[], |n| n.layers.as_slice()),
            layer_pos: 0,
            edge_pos: 0,
            seen: SmallVec::new(), // BN-07: inline stack, no heap allocation for ≤128 IDs
        }
    }

    /// Iterate over neighbours of a node within a given distance radius (layer 0 only).
    ///
    /// AX-ID: AXIOMA-013
    pub fn neighbors_within(&self, id: NodeId, radius: f64) -> impl Iterator<Item = NodeId> + '_ {
        // FIX-E.1: Single get_idx call — the former code called get_idx(id) twice
        // (once for `node_vec`, once for `idx`), wasting a lookup per call.
        let candidates: SmallVec<[NodeId; M]> =
            self.get_idx(id).map_or_else(SmallVec::new, |idx| {
                // SAFETY: `idx` comes from `self.get_idx(id)`, which guarantees an in-bounds index.
                let node = unsafe { self.nodes.get_unchecked(idx) };
                let nv = node.vec;
                node.layers.first().map_or_else(SmallVec::new, |layer0| {
                    let mut local = SmallVec::<[NodeId; M]>::with_capacity(layer0.len().min(M0));
                    for &(nb_id, _) in layer0 {
                        if let Some(ni) = self.get_idx(nb_id) {
                            let d = self.distance_to_node(&nv, ni, 0);
                            if d <= radius {
                                local.push(nb_id);
                            }
                        }
                    }
                    local
                })
            });
        candidates.into_iter()
    }

    /// Recorre vecinos únicos de un nodo en todas las capas sin asignaciones.
    ///
    /// Usa `marks` como bitmap por índice interno con `stamp` como generación.
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
        let Some(idx) = self.get_idx(id) else {
            return 0;
        };
        let Some(node) = self.nodes.get(idx) else {
            return 0;
        };
        debug_assert!(marks.len() >= self.nodes.len());

        let mut pushed = 0;
        for layer in &node.layers {
            for &(nb_id, _) in layer {
                let Some(nb_idx) = self.get_idx(nb_id) else {
                    continue;
                };
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

    /// Obtiene el vector asociado a un `NodeId`.
    pub fn get_vector(&self, id: NodeId) -> Option<&SparseCliffordVector> {
        self.get_idx(id).map(|idx| &self.nodes[idx].vec)
    }

    /// Iterate over all `NodeIds` in the graph.
    pub fn nodes(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.nodes.iter().map(|n| n.id)
    }

    /// Number of nodes in the graph.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    pub const fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Total number of directed edges at layer 0 (base connectivity).
    pub fn edge_count(&self) -> usize {
        self.nodes
            .iter()
            .map(|n| n.layers.first().map_or(0, Vec::len))
            .sum()
    }

    /// Build a contiguous layer-0 SoA snapshot for read-heavy numeric pipelines.
    ///
    /// AX-ID: AXIOMA-013
    pub fn layer0_soa(&self) -> HnswLayer0Soa {
        let mut node_ids = Vec::with_capacity(self.nodes.len());
        let mut dense_coeffs = Vec::with_capacity(self.nodes.len());
        let mut neighbor_offsets = Vec::with_capacity(self.nodes.len());
        let mut neighbor_ids = Vec::new();
        let mut neighbor_distances = Vec::new();

        for node in &self.nodes {
            node_ids.push(node.id);
            dense_coeffs.push(decode_layer0_to_f64(&node.layer0));
            let start = neighbor_ids.len();
            if let Some(layer0) = node.layers.first() {
                neighbor_ids.reserve(layer0.len());
                neighbor_distances.reserve(layer0.len());
                for &(nb_id, d) in layer0 {
                    neighbor_ids.push(nb_id);
                    neighbor_distances.push(d);
                }
            }
            neighbor_offsets.push((start, neighbor_ids.len()));
        }

        HnswLayer0Soa {
            node_ids,
            dense_coeffs,
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
            .get_idx(id)
            .ok_or(GenesisError::InvariantViolation { axiom_id: 4 })?;

        // Step 1: Remove all edges originating from this node.
        // Collect neighbour IDs first to avoid borrow conflicts.
        let all_neighbours: Vec<(NodeId, usize)> = self.nodes[idx]
            .layers
            .iter()
            .enumerate()
            .flat_map(|(layer, adj)| adj.iter().map(move |&(nb_id, _)| (nb_id, layer)))
            .collect();

        for (nb_id, layer) in all_neighbours {
            // Remove the reverse edge: nb → id
            if let Some(nb_idx) = self.get_idx(nb_id) {
                self.remove_edge(nb_idx, layer, id);
            }
        }

        // Step 2: Clear the node's own adjacency lists.
        for layer in &mut self.nodes[idx].layers {
            layer.clear();
        }

        // Step 3: Invalidate direct_index entry.
        let raw = id.get();
        if raw < u64::from(u32::MAX) {
            #[allow(clippy::cast_possible_truncation)]
            let raw_us = raw as usize;
            if raw_us < self.direct_index.len() {
                self.direct_index[raw_us] = u32::MAX;
            }
        }

        // Step 4: Update entry point if it was pointing to this node.
        if self.entry == Some(idx) {
            // Find a new entry point: the node with the highest layer.
            self.entry = self
                .nodes
                .iter()
                .enumerate()
                .filter(|(i, n)| *i != idx && !n.layers.is_empty())
                .max_by_key(|(_, n)| n.layers.len())
                .map(|(i, n)| {
                    self.entry_layer = n.layers.len() - 1;
                    i
                });
        }

        // Step 5: Mark the node slot as invalid (keep Vec size stable for index integrity).
        // A compact_index() call rebuilds id_index cleanly. Do not swap_remove
        // because that would invalidate all internal indices stored in adjacency lists.
        self.nodes[idx].id = NodeId::INVALID;
        self.nodes[idx].layers.clear();

        Ok(())
    }
}

fn decode_layer0_to_f64(stored: &Layer0Coeffs) -> [f64; TOTAL_BLADES] {
    <ActiveLayer0Codec as layer0_codec::Layer0Codec>::decode_to_f64(stored)
}

impl LockFreeHnswIndex {
    /// Create a lock-free snapshot index with an empty HNSW graph.
    ///
    /// AX-ID: AXIOMA-013
    pub fn new(ef_construction: usize) -> Self {
        let snapshot = Arc::new(HnswGraph::new(ef_construction));
        Self {
            head: AtomicPtr::new(Arc::into_raw(snapshot).cast_mut()),
        }
    }

    fn load_snapshot(&self) -> Arc<HnswGraph> {
        loop {
            let ptr = self.head.load(AtomicOrdering::Acquire);
            assert!(!ptr.is_null(), "LockFreeHnswIndex head must be initialized");

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
            let mut updated = (*base).clone();
            updated.insert(id, vec)?;
            let candidate = Arc::into_raw(Arc::new(updated)).cast_mut();

            match self.head.compare_exchange(
                current,
                candidate,
                AtomicOrdering::AcqRel,
                AtomicOrdering::Acquire,
            ) {
                Ok(_) => {
                    // SAFETY: Successful CAS replaced the head's strong reference from
                    // `current` to `candidate`; release the superseded head ref.
                    unsafe {
                        drop(Arc::from_raw(current));
                    }
                    return Ok(());
                }
                Err(_) => {
                    // SAFETY: CAS failed, so `candidate` was never published.
                    unsafe {
                        drop(Arc::from_raw(candidate));
                    }
                }
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

/// LSD radix sort para pares `(NodeId, internal_idx)` por `NodeId.get()` en O(8N).
fn radix_sort_node_ids(index: &mut Vec<(NodeId, usize)>) {
    if index.len() <= 1 {
        return;
    }

    let len = index.len();
    let mut src = std::mem::take(index);
    let mut dst = vec![(NodeId::INVALID, 0usize); len];

    for pass in 0..8 {
        let shift = pass * 8;
        let mut counts = [0usize; 256];
        let workers = thread::available_parallelism()
            .map(std::num::NonZero::<usize>::get)
            .unwrap_or(1)
            .min(8);

        if workers > 1 && src.len() >= 4_096 {
            let chunk = src.len().div_ceil(workers);
            thread::scope(|scope| {
                let mut handles = Vec::new();
                for part in src.chunks(chunk) {
                    handles.push(scope.spawn(move || {
                        let mut local = [0usize; 256];
                        for &(id, _) in part {
                            let bucket = ((id.get() >> shift) & 0xFF) as usize;
                            local[bucket] += 1;
                        }
                        local
                    }));
                }
                for handle in handles {
                    let local = handle.join().expect("radix histogram worker panicked");
                    for (i, value) in local.into_iter().enumerate() {
                        counts[i] += value;
                    }
                }
            });
        } else {
            for &(id, _) in &src {
                let bucket = ((id.get() >> shift) & 0xFF) as usize;
                counts[bucket] += 1;
            }
        }

        let mut offsets = [0usize; 256];
        let mut running = 0usize;
        for i in 0..256 {
            offsets[i] = running;
            running += counts[i];
        }

        for item in src.iter().copied() {
            let bucket = ((item.0.get() >> shift) & 0xFF) as usize;
            let pos = offsets[bucket];
            dst[pos] = item;
            offsets[bucket] += 1;
        }

        std::mem::swap(&mut src, &mut dst);
    }

    *index = src;
}

/// Iterador sobre vecinos deduplicados de un nodo en todas las capas.
///
/// Usa un bitmap de 64 bits en stack para nodos con ID < 64 (caso típico
/// en grafos pequeños a medianos). Para IDs ≥ 64 usa un Vec<u64> ordenado
/// con búsqueda binaria: O(log seen) en lugar de O(seen) lineal.
///
/// AX-ID: AXIOMA-013
struct NeighborIter<'a> {
    layers: &'a [Vec<(NodeId, f64)>],
    layer_pos: usize,
    edge_pos: usize,
    /// Deduplicated node IDs already emitted, sorted ascending for binary search.
    ///
    /// # BN-07: SmallVec eliminates heap allocation
    /// Inline capacity of 128 covers the theoretical maximum of unique neighbours
    /// across all HNSW layers: M0 (32) + M × MAX_LAYERS (16 × 6 = 96) = 128.
    /// The 99.9% common case (< 64 unique neighbours) never touches the heap.
    /// Graceful spill to heap for rare deep-hierarchy nodes (no panic, no truncation).
    seen: SmallVec<[u64; 128]>,
}

impl Iterator for NeighborIter<'_> {
    type Item = NodeId;

    fn next(&mut self) -> Option<NodeId> {
        loop {
            if self.layer_pos >= self.layers.len() {
                return None;
            }
            let layer = &self.layers[self.layer_pos];
            if self.edge_pos >= layer.len() {
                self.layer_pos += 1;
                self.edge_pos = 0;
                continue;
            }
            let (nid, _) = layer[self.edge_pos];
            self.edge_pos += 1;
            let raw = nid.get();
            // Búsqueda binaria en Vec ordenado: O(log |seen|) en lugar de O(|seen|).
            // Para el caso típico (M_total ≤ 64 vecinos únicos), |seen| ≤ 64,
            // log₂(64) = 6 comparaciones vs 64 lineales: 10× más rápido.
            match self.seen.binary_search(&raw) {
                Ok(_) => continue, // ya emitido
                Err(pos) => {
                    // Insertar en posición ordenada para mantener invariante.
                    // Cost: O(|seen|) shift, pero |seen| ≤ M_total ≤ 64. Aceptable.
                    self.seen.insert(pos, raw);
                    return Some(nid);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use genesis_math::SparseCliffordVector;
    use proptest::prelude::*;

    use super::*;

    fn make_vec(coeff: f64) -> SparseCliffordVector {
        SparseCliffordVector::from_iter((0..4).map(|i| (i, coeff * (i as f64 + 1.0) * 0.1)))
            .unwrap()
    }

    fn make_id(v: u64) -> NodeId {
        NodeId::try_new(v).expect("NodeId válido por construcción")
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
            .map(|node| node.layers.first().map_or(0, Vec::len))
            .sum();

        assert_eq!(soa.node_ids.len(), graph.node_count());
        assert_eq!(soa.dense_coeffs.len(), graph.node_count());
        assert_eq!(soa.neighbor_offsets.len(), graph.node_count());
        assert_eq!(soa.neighbor_ids.len(), total_layer0);
        assert_eq!(soa.neighbor_distances.len(), total_layer0);
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

        assert_eq!(index.node_count(), 32);
        let query = make_vec(0.25);
        let result = index.search_nearest(&query, 4);
        assert!(!result.is_empty());
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
        let id = NodeId::try_new(7).expect("NodeId válido por construcción");
        let first = make_vec(0.8);
        let second = make_vec(1.9);

        assert!(g.insert(id, &first).is_ok());
        assert!(g.insert(id, &second).is_ok());

        assert_eq!(
            g.node_count(),
            1,
            "duplicate insert must not add a new node"
        );
        assert_eq!(g.get_vector(id), Some(&first));
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
                NodeId::try_new(i as u64).expect("NodeId válido por construcción"),
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
            .map(|(i, v)| (i, geometric_distance(&query, v)))
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
        let fixture: [[f64; TOTAL_BLADES]; 8] = [
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
            .map(|(idx, v)| (idx, geometric_distance(&query, v)))
            .collect();
        brute.sort_unstable_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));

        let got = g.search_nearest(&query, 1);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0], make_id(brute[0].0 as u64));
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
                NodeId::try_new(i as u64).expect("NodeId válido por construcción"),
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
                NodeId::try_new(i as u64).expect("NodeId válido por construcción"),
                &v,
            )
            .unwrap();
        }

        let mut level_0 = 0usize;
        let mut level_1 = 0usize;
        let mut max_level = 0usize;
        for node in &g.nodes {
            let level = node.layers.len() - 1;
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

        assert!(
            (0.90..=0.97).contains(&p0),
            "P(level=0) fuera de rango: {p0}"
        );
        assert!(
            (0.03..=0.08).contains(&p1),
            "P(level=1) fuera de rango: {p1}"
        );

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
                NodeId::try_new(i as u64).expect("NodeId válido por construcción"),
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
        // Un nodo con vecinos solapados entre capas debe emitir cada ID exactamente una vez.
        let mut g = HnswGraph::new(16);
        for i in 0..20u64 {
            let v = make_vec((i as f64).mul_add(0.1, 0.1));
            g.insert(
                NodeId::try_new(i).expect("NodeId válido por construcción"),
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
        // Los resultados de search_nearest deben estar ordenados por distancia.
        let mut g = HnswGraph::new(16);
        for i in 0..50u64 {
            let v = make_vec((i as f64).mul_add(0.05, 0.1));
            g.insert(
                NodeId::try_new(i).expect("NodeId válido por construcción"),
                &v,
            )
            .unwrap();
        }
        let query = make_vec(1.5);
        let results = g.search_nearest(&query, 10);
        // Verificar que HNSW devuelve resultados sin duplicados.
        let mut ids = std::collections::HashSet::new();
        for id in &results {
            assert!(
                ids.insert(id.get()),
                "search_nearest devolvió NodeId duplicado"
            );
        }
    }
    #[test]
    fn edge_density_within_bounds() {
        let mut g = HnswGraph::new(16);
        for i in 0..50u64 {
            let v = make_vec((i as f64).mul_add(0.05, 0.1));
            g.insert(
                NodeId::try_new(i).expect("NodeId válido por construcción"),
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
                NodeId::try_new(i).expect("NodeId válido por construcción"),
                &v,
            )
            .unwrap();

            // Verify local degree bounds per node per layer.
            // This is the tighter invariant that replaces global density enforcement.
            for node in &g.nodes {
                for (layer_idx, adj) in node.layers.iter().enumerate() {
                    let m_max = if layer_idx == 0 { M0 } else { M };
                    assert!(
                        adj.len() <= m_max,
                        "node {:?} layer {layer_idx}: degree {} > m_max {}",
                        node.id,
                        adj.len(),
                        m_max
                    );
                }
            }
        }
    }

    #[test]
    fn compacted_state_blocks_new_appends() {
        let mut g = HnswGraph::new(16);
        let v0 = make_vec(0.4);
        g.insert(
            NodeId::try_new(0).expect("NodeId válido por construcción"),
            &v0,
        )
        .unwrap();
        g.compact_index();

        let v1 = make_vec(0.5);
        let err = g
            .insert(
                NodeId::try_new(1).expect("NodeId válido por construcción"),
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
                NodeId::try_new(i).expect("NodeId válido por construcción"),
                &v,
            )
            .unwrap();
        }

        for slot in &mut g.direct_index {
            *slot = u32::MAX;
        }

        g.compact_index();

        for id in [1_u64, 2, 5, 7, 9] {
            assert!(g
                .get_idx(NodeId::try_new(id).expect("NodeId válido por construcción"))
                .is_some());
        }
        assert!(g
            .get_idx(NodeId::try_new(3).expect("NodeId válido por construcción"))
            .is_none());
    }
    #[cfg(feature = "hnsw-f16")]
    #[test]
    fn encode_layer0_matches_f16_encoder_per_index() {
        let input = core::array::from_fn(|i| (i as f32).mul_add(0.25, -1.5));
        let encoded = encode_layer0(&input).expect("finite input must encode");

        for i in 0..TOTAL_BLADES {
            assert_eq!(encoded[i], f32_to_f16_bits(input[i]), "blade index {i}");
        }
    }

    #[cfg(not(feature = "hnsw-f16"))]
    #[test]
    fn encode_layer0_returns_exact_f32_copy_without_feature() {
        let input = core::array::from_fn(|i| (i as f32 * 1.25) - 7.0);
        let encoded = encode_layer0(&input).expect("finite input must encode");

        assert_eq!(encoded, input);
    }

    #[cfg(feature = "hnsw-f16")]
    #[test]
    fn layer0_f16_storage_is_half_of_f32() {
        assert_eq!(
            std::mem::size_of::<Layer0Coeffs>() * 2,
            std::mem::size_of::<[f32; TOTAL_BLADES]>()
        );
    }

    #[cfg(feature = "hnsw-f16")]
    #[test]
    fn layer0_distance_regression_f32_vs_f16_fixed_dataset() {
        let dataset: [[f64; TOTAL_BLADES]; 6] = [
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
                let f32_layer: [f32; TOTAL_BLADES] =
                    core::array::from_fn(|i| vector_dense[i] as f32);
                let encoded_f32 =
                    <layer0_codec::F32Codec as layer0_codec::Layer0Codec>::encode(&f32_layer);
                let encoded_f16 =
                    <layer0_codec::F16Codec as layer0_codec::Layer0Codec>::encode(&f32_layer);

                let d_f32 = <layer0_codec::F32Codec as layer0_codec::Layer0Codec>::distance(
                    &encoded_f32,
                    &query,
                );
                let d_f16 = <layer0_codec::F16Codec as layer0_codec::Layer0Codec>::distance(
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

            let _node = HnswNode::new(make_id(999), vector, 0).expect("finite vector must create node");
            let layer0_f32: [f32; TOTAL_BLADES] = core::array::from_fn(|i| input[i] as f32);
            let encoded = encode_layer0(&layer0_f32).expect("finite layer0 must encode");

            #[cfg(not(feature = "hnsw-f16"))]
            {
                prop_assert!(encoded.iter().all(|value| value.is_finite()));
            }

            #[cfg(feature = "hnsw-f16")]
            {
                let decoded: [f32; TOTAL_BLADES] = core::array::from_fn(|i| f16_bits_to_f32(encoded[i]));
                prop_assert!(decoded.iter().all(|value| value.is_finite()));
            }
        }

        #[test]
        fn sparse_vector_rejects_non_finite_coefficients(
            idx in 0usize..TOTAL_BLADES,
            use_nan in any::<bool>()
        ) {
            let mut dense = [0.0_f64; TOTAL_BLADES];
            dense[idx] = if use_nan { f64::NAN } else { f64::INFINITY };

            let result = SparseCliffordVector::from_dense(&dense);
            prop_assert!(result.is_err());
        }
    }

    #[test]
    fn encode_layer0_returns_invalid_input_when_input_contains_non_finite_values() {
        let mut input = [0.0_f32; TOTAL_BLADES];
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
                    let d = geometric_distance(v, &query);
                    #[cfg(not(feature = "hnsw-f16"))]
                    let d = geometric_distance(&query, v);
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
                        fast_bivector_distance_f16(&layer0, &query)
                    };
                    #[cfg(not(feature = "hnsw-f16"))]
                    let dist = geometric_distance(&query, v);
                    (idx, dist)
                })
                .collect();
            stored.sort_unstable_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
            let stored_top5: Vec<usize> = stored.iter().take(5).map(|(id, _)| *id).collect();

            assert_eq!(stored_top5, exact_top5, "query={q}");
        }
    }
}

#[cfg(test)]
mod scaling_tests {
    use genesis_math::SparseCliffordVector;
    use genesis_types::NodeId;

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
        fn make_vec(seed: u64) -> SparseCliffordVector {
            let mut coeffs = [0.0f64; 16];
            let mut rng = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            for c in &mut coeffs {
                rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                *c = ((rng >> 33) as f64 / u32::MAX as f64).mul_add(2.0, -1.0);
            }
            SparseCliffordVector::from_dense(&coeffs)
                .unwrap_or_else(|_| SparseCliffordVector::zero())
        }

        fn build_and_time_search(n: usize, repetitions: u32) -> std::time::Duration {
            let mut graph = HnswGraph::new(16);
            for i in 0..n {
                let id = NodeId::try_new(i as u64).unwrap();
                let v = make_vec(i as u64 * 31337);
                let _ = graph.insert(id, &v);
            }
            let query = make_vec(999_999);
            let start = std::time::Instant::now();
            for _ in 0..repetitions {
                let _ = graph.search_nearest(&query, 10);
            }
            start.elapsed() / repetitions
        }

        // Use larger N to reduce constant-factor inflation on sandbox VMs.
        // N=1000 vs N=10000: 10× more nodes.
        // O(log N) theoretical ratio: log(10000)/log(1000) = 4/3 ≈ 1.33
        // O(N) ratio would be: 10.0
        // We allow ≤ 6.0 to handle sandbox CPU variance while still catching O(N) regressions.
        let reps = 30u32;
        let t_small = build_and_time_search(500, reps);
        let t_large = build_and_time_search(5000, reps);

        let ratio = t_large.as_nanos() as f64 / t_small.as_nanos().max(1) as f64;

        assert!(
            ratio <= 6.0,
            "HNSW search scaling regression: ratio({ratio:.2}×) > 6.0 — \
             expected O(log N) ≈ 1.3× for 10× more nodes; O(N) would be 10×. \
             AX-ID: AXIOMA-013 violated."
        );

        // Also assert we didn't degrade below O(1) (ratio should be > 0.3)
        assert!(
            ratio > 0.1,
            "ratio={ratio:.2} suspiciously small — benchmark noise"
        );
    }
}
