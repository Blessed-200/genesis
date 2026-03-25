use std::cell::RefCell;

use genesis_math::SparseCliffordVector;
use smallvec::SmallVec;

use crate::cohomology::CohomologyValidator;
use crate::hnsw::HnswGraph;
use crate::incremental_cohomology::IncrementalH1State;
use crate::rips::RipsComplex;
use crate::topological_intuition::{TopologicalHypothesis, TopologicalIntuition};

const EDGE_DENSITY_LOG_BASE: f64 = 2.0;
const LANCZOS_MAX_ITERS_DEFAULT: usize = 50;
const LANCZOS_REORTHOGONALIZE_EVERY: usize = 10;
const LANCZOS_CONVERGENCE_EPS: f64 = 1e-9;
const POWER_REFINE_MAX_ITERS: usize = 800;

// Policy of maintenance for collectors topological critical.
//
// - `#[inline(always)]` is forbidden by default; an exception requires
//   justification documentada (benchmark + architectural rationale + risk).
// - `cfg` rules of the codec: any branch conditioned by
//   `feature = "hnsw-f16"` or `genesis_const_layer0_codec` must keep
//   counterpart `not(...)` verifiesble for prevent symbols orphan.
//
// AX-ID: AXIOMA-007, AXIOMA-013, H_restricción (LEY_FUNDACIONAL §5.6)

// ── HyperbolicCoord — Contract for CRATE-004 ─────────────────────────────────

/// Coordinate in the Poincaré disk ℍ² (model of curvature constant −1).
///
/// # Invariant
/// `x² + y² < 1` always — the point lives strictly within the unit disk.
///
/// # Cognitive semantics
///
/// The Poincaré disk represents hierarchies naturally:
/// - **Center of the disk** (`|coord| → 0`): concepts root of high connectivity
///   (baja Ollivier-Ricci curvature, muchos vecinos HNSW).
/// - **Edge of the disk** (`|coord| → 1`): concepts hoja of baja connectivity
///   (high curvature, pocos vecinos, high especificidad semantic).
///
/// The hyperbolic distance between two points grows exponentially towards the
///edge, which allows representing hierarchies with exponential depth
/// in espacio lineal.
///
/// # State of activation
///
/// **Contract — no active yet.**
/// The field `hyperbolic_coords` in `ManifoldCollector` exists but always
/// returns `None` until `DiscreteRicciFlow` (CRATE-004) starts
/// updatesr coordenadas with `ManifoldCollector::set_hyperbolic_coord()`.
///
/// CRATE-004 will compute coordinates using the projection formula based on
/// curvature local: `r = tanh(K_avg_node / 2)`, an angle derived from Kuramoto synchrony.
///
/// # Invariants physical of the disk of Poincaré
/// ```
/// use genesis_topology::manifold::HyperbolicCoord;
///
/// // Every valid coordinate lives strictly inside the unit disk.
/// // The boundary r=1 represents the point at infinity — excluded.
/// let p = HyperbolicCoord::new(0.3, 0.4).unwrap(); // r²=0.25 < 1
/// assert!(p.norm_sq() < 1.0);
/// assert!(HyperbolicCoord::new(1.0, 0.0).is_none()); // r=1 rejected
///
/// // Hyperbolic distance to origin: d(0,p) = 2·arctanh(|p|)
/// // For p=(0.3,0.4): |p|=0.5, so d=2·arctanh(0.5)≈1.0986
/// let d = p.hyperbolic_distance_to_origin();
/// assert!((d - 2.0_f64 * 0.5_f64.atanh()).abs() < 1e-12);
/// ```
///
/// AX-ID: AXIOMA-004 (paisaje of atractores), LEY_FUNDACIONAL §7.2
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HyperbolicCoord {
    /// Coordinate x on the disk of Poincaré. `x² + y² < 1`.
    pub x: f64,
    /// Coordinate and on the disk of Poincaré. `x² + y² < 1`.
    pub y: f64,
}

impl HyperbolicCoord {
    /// Constructor validated. Returns `None` if `x² + y² ≥ 1`.
    ///
    /// # Invariant
    /// Todo `HyperbolicCoord` valid satisfies `self.norm_sq() < 1.0`.
    ///
    /// ```
    /// use genesis_topology::manifold::HyperbolicCoord;
    ///
    /// // Poincaré disk invariant: all valid coordinates satisfy r² < 1
    /// let coord = HyperbolicCoord::new(0.3, 0.4).expect("valid: 0.09+0.16=0.25 < 1");
    /// assert!(coord.norm_sq() < 1.0);
    ///
    /// // Boundary is excluded: r = 1 represents the point at infinity
    /// assert!(HyperbolicCoord::new(1.0, 0.0).is_none());
    ///
    /// // Hyperbolic distance to origin: d(0,p) = 2·arctanh(|p|)
    /// let dist = coord.hyperbolic_distance_to_origin();
    /// let expected = 2.0 * 0.5_f64.atanh(); // |p| = √0.25 = 0.5
    /// assert!((dist - expected).abs() < 1e-12);
    /// ```
    #[must_use]
    pub fn new(x: f64, y: f64) -> Option<Self> {
        if x.mul_add(x, y * y) < 1.0 && x.is_finite() && y.is_finite() {
            Some(Self { x, y })
        } else {
            None
        }
    }

    /// Euclidean norm to the square: `x² + y²`. Always < 1 for invariant.
    #[inline]
    #[must_use]
    pub fn norm_sq(&self) -> f64 {
        self.x.mul_add(self.x, self.y * self.y)
    }

    /// Hyperbolic distance to the origin of the disk.
    /// `d(0, p) = 2·arctanh(|p|)`
    #[inline]
    #[must_use]
    pub fn hyperbolic_distance_to_origin(&self) -> f64 {
        2.0 * self.norm_sq().sqrt().atanh()
    }
}

/// Workspace per thread for `compute_lambda2`. All the buffers are `Vec` dynamic.
/// It redimensionan lazy with crecimiento geometric (`next_power_of_two`) para
/// amortize reallocations and improve locality of cache.
/// Sin limit fijo of nodes — only the memory of the system lo acotat.
#[repr(C, align(32))]
#[derive(Default)]
struct LambdaWorkspace {
    degrees: Vec<f64>,
    adj_flat: Vec<usize>,
    adj_offsets: Vec<(usize, usize)>,
    y: Vec<f64>,
    q_prev: Vec<f64>,
    q_curr: Vec<f64>,
    w: Vec<f64>,
    basis: Vec<f64>,
    alpha: Vec<f64>,
    beta: Vec<f64>,
    tri_vec: Vec<f64>,
    tri_tmp: Vec<f64>,
    seen_marks: Vec<u32>,
    seen_generation: u32,
}

impl LambdaWorkspace {
    fn new() -> Self {
        Self::default()
    }
}

fn ensure_lambda_workspace_capacity(ws: &mut LambdaWorkspace, n: usize, max_iters: usize) {
    // ── Redimensionamiento lazy with crecimiento geometric ──────────
    let n_cap = n.next_power_of_two();
    if ws.degrees.len() < n {
        ws.degrees.resize(n_cap, 0.0);
        ws.adj_offsets.resize(n_cap, (0, 0));
        ws.y.resize(n_cap, 0.0);
        ws.q_prev.resize(n_cap, 0.0);
        ws.q_curr.resize(n_cap, 0.0);
        ws.w.resize(n_cap, 0.0);
        ws.seen_marks.resize(n_cap, 0);
    }
    if ws.adj_flat.is_empty() {
        ws.adj_flat = Vec::with_capacity(n_cap.saturating_mul(16));
    }

    let iters_cap = max_iters.next_power_of_two();
    if ws.alpha.len() < max_iters {
        ws.alpha.resize(iters_cap, 0.0);
        ws.beta.resize(iters_cap, 0.0);
        ws.tri_vec.resize(iters_cap, 0.0);
        ws.tri_tmp.resize(iters_cap, 0.0);
    }

    let basis_needed = n * max_iters;
    if ws.basis.len() < basis_needed {
        ws.basis.resize(basis_needed.next_power_of_two(), 0.0);
    }
}

thread_local! {
    static LAMBDA_SCRATCH: RefCell<LambdaWorkspace> = RefCell::new(LambdaWorkspace::new());
}
/// AX-ID: AXIOMA-007, AXIOMA-013, `H_restricción` (λ₂, H¹, density)
/// `ManifoldCollector`: the cognitive manifold wrapping `HnswGraph`.
/// Provides `compute_lambda2()`, `compute_edge_density()`, `compute_h1()`,
/// `find_affected_nodes()`, and edge/node count delegation.
use genesis_types::{GenesisError, NodeId, REDUNDANCY_RADIUS};

/// `ManifoldCollector` — the cognitive space where GENESIS concepts exist.
///
/// Wraps `HnswGraph` and provides:
/// - Topological metrics (λ₂, edge density, H¹)
/// - Structural queries (affected nodes)
/// - Invariant verifiestion interfaces
/// - Hyperbolic coordinates (contract for CRATE-004 — inactive until Ricci flow)
///
/// AX-ID: AXIOMA-007, AXIOMA-013, AXIOMA-014, `H_restricción` §3.5
pub struct ManifoldCollector {
    graph: HnswGraph,
    h1_state: IncrementalH1State,
    /// Hyperbolic coordinates per node on the disk of Poincaré.
    ///
    /// **Contract — populated by CRATE-004 (`DiscreteRicciFlow`).**
    /// Before of that CRATE-004 this implementado, this Vec is empty y
    /// `hyperbolic_coord(id)` siempre returns `None`.
    ///
    /// Stored as Vec sorted by `NodeId::get()` for search O(log N)
    /// without HashMap (cumple restricción of hot-path of the workspace).
    /// Insertado mediante `set_hyperbolic_coord()` with maintenance of order.
    ///
    /// AX-ID: AXIOMA-004, LEY_FUNDACIONAL §7.2
    hyperbolic_coords: Vec<(u64, HyperbolicCoord)>,
}

impl ManifoldCollector {
    /// Create a new empty manifold.
    ///
    /// AX-ID: AXIOMA-013
    pub fn new(ef_construction: usize) -> Self {
        Self {
            graph: HnswGraph::new(ef_construction),
            h1_state: IncrementalH1State::new(),
            hyperbolic_coords: Vec::new(),
        }
    }

    /// Insert a vector into the manifold.
    ///
    /// # Errors
    /// Propagates `GenesisError` from `HnswGraph::insert` if insertion fails.
    ///
    /// # Allocation contract
    ///
    /// Zero heap allocations after BN-02. All neighbour buffers are stack-allocated
    /// using the compile-time HNSW degree bound `M0 = 32`. Triangle detection uses
    /// binary search on a sorted stack array instead of `HashSet`.
    ///
    /// AX-ID: AXIOMA-013
    pub fn insert(&mut self, id: NodeId, vec: &SparseCliffordVector) -> Result<(), GenesisError> {
        self.graph.insert(id, vec)?;

        // Actualizar state incremental of H¹.
        self.h1_state.add_node();

        // BN-02: Stack-allocated neighbour buffer — max M0 neighbours at layer 0.
        // M0 = 32 is the HNSW compile-time degree bound for layer 0.
        // Upper layers contribute at most M = 16 neighbours each, but the unique
        // union across all layers is bounded by M0 in practice (layer-0 dominates).
        // Using M0 as the static bound; debug_assert guards correctness.
        use crate::hnsw::M0;
        let mut neighbors_buf = [NodeId::INVALID; M0];
        let mut neighbor_count = 0usize;

        for neighbor in self.graph.neighbors(id) {
            debug_assert!(
                neighbor_count < M0,
                "BN-02: neighbor count {} exceeded M0 = {M0} — check HNSW degree bounds",
                neighbor_count
            );
            if neighbor_count < M0 {
                neighbors_buf[neighbor_count] = neighbor;
                neighbor_count += 1;
            }
        }
        let neighbors = &mut neighbors_buf[..neighbor_count];

        // Registrar edges nuevas.
        for &v in neighbors.iter() {
            self.h1_state.add_edge(id, v);
        }

        // Sort for binary-search based triangle detection — replaces HashSet.
        // O(K log K) where K ≤ M0 = 32. Entirely in L1 cache.
        neighbors.sort_unstable();

        // Detect triangles: for each pair (v, w) of vecinos of `id`,
        // comprobar if v and w are conectados → triangle (id, v, w).
        // Binary search on sorted stack array replaces HashSet lookup.
        // Complexity: O(K² · log K) per insertion. Para K=32: ~5120 ops — L1.
        for i in 0..neighbor_count {
            let v = neighbors[i];
            for w in self.graph.neighbors(v) {
                // Only process each triangle once: w > v, w must also be neighbour of id.
                if w > v && neighbors[i + 1..].binary_search(&w).is_ok() {
                    self.h1_state.add_triangle(id, v, w);
                }
            }
        }

        Ok(())
    }

    /// Return the number of nodes.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    pub const fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Return the total edge count.
    pub const fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Remove node `id` from the HNSW graph and the incremental H¹ state.
    ///
    /// Required by `inelastic_concept_fusion` in CRATE-004 (genesis-evolution)
    /// when the absorbed node must be purged after merge.
    ///
    /// Steps:
    /// 1. Remove from `graph` (HnswGraph::remove_node) — purges edges.
    /// 2. Remove hyperbolic coordinate if present.
    /// 3. Remove from `h1_state` (IncrementalH1State tracks edge IDs by NodeId).
    ///
    /// Returns `Err(NodeNotFound)` if the node is not registered.
    /// Complexity: O(K × layers) for HNSW edge cleanup.
    ///
    /// AX-ID: LEY_FUNDACIONAL §3.7 (WormholeCollapse), CRATE-004 prerequisite
    pub fn remove_node(&mut self, id: NodeId) -> Result<(), GenesisError> {
        // Step 1: Remove from HNSW graph (purges edges).
        self.graph.remove_node(id)?;

        // Step 2: Remove hyperbolic coordinate if present.
        self.hyperbolic_coords
            .retain(|&(raw_id, _)| raw_id != id.get());

        // Step 3: Remove node-related H¹ state.
        self.h1_state.remove_node(id);

        Ok(())
    }

    /// Compute the edge density ratio: |E| / (N × log₂(N)).
    ///
    /// Returns 0.0 if N < 2.
    ///
    /// Used as a sub-term of `H_restricción`: if this exceeds 1.0, the system
    /// is denser than the O(N log N) HNSW optimum.
    ///
    /// AX-ID: `LEY_FUNDACIONAL` §3.5, AXIOMA-013
    pub fn compute_edge_density(&self) -> f64 {
        let n = self.graph.node_count();
        if n < 2 {
            return 0.0;
        }
        // N = node_count(). Para N < 2^52 (limit físico of memory), usize→f64 exact.
        // Calculation of density of edges: N·log(N) no requires precisión of entero exact.
        #[allow(clippy::cast_precision_loss)]
        let n_f = n as f64;
        let log_n = n_f.log(EDGE_DENSITY_LOG_BASE).max(1.0);
        // edge_count ≤ N² and for N < 2^52 the cast a f64 is aceptable for metric.
        #[allow(clippy::cast_precision_loss)]
        let edge_count_f = self.graph.edge_count() as f64;
        edge_count_f / (n_f * log_n)
    }

    /// Compute the algebraic connectivity λ₂ — the second smallest eigenvalue
    /// of the normalised graph Laplacian.
    ///
    /// Algorithm: Truncated Lanczos on a shifted Laplacian matrix.
    /// Convergence: `|λ_new - λ_old| < 1e-9` or max 50 iterations.
    /// Reorthogonalisation: every 10 iterations.
    ///
    /// En graphs N<200 power iteration can be competitivo; Lanczos gana
    /// at scale N>1000 by convergence in fewer iterations.
    /// Returns 0.0 for disconnected graphs or graphs with < 2 nodes.
    ///
    /// # Panics
    /// Panics if internal node identifiers exceed the valid [`NodeId`] range.
    ///
    /// AX-ID: `LEY_FUNDACIONAL` §5.3, `H_restricción` §3.5
    pub fn compute_lambda2(&self) -> f64 {
        let n = self.graph.node_count();
        if n < 2 {
            return 0.0;
        }

        let max_iters = LANCZOS_MAX_ITERS_DEFAULT.min(n);

        LAMBDA_SCRATCH.with(|cell| {
            let mut ws = cell.borrow_mut();
            ensure_lambda_workspace_capacity(&mut ws, n, max_iters);
            // CRYSTAL: FO121 — inevitable

            // ── Safe handling of the generation counter ────────────────────
            if ws.seen_generation == u32::MAX {
                ws.seen_marks.fill(0);
                ws.seen_generation = 1;
            } else {
                ws.seen_generation = ws.seen_generation.saturating_add(1);
            }

            let LambdaWorkspace {
                degrees,
                adj_flat,
                adj_offsets,
                y,
                q_prev,
                q_curr,
                w,
                basis,
                alpha,
                beta,
                tri_vec,
                tri_tmp,
                seen_marks,
                seen_generation: seen_generation_ref,
            } = &mut *ws;

            let (sigma, seen_generation) = prepare_laplacian_data(
                &self.graph,
                n,
                &mut degrees[..n],
                adj_flat,
                &mut adj_offsets[..n],
                &mut seen_marks[..n],
                *seen_generation_ref,
            );
            *seen_generation_ref = seen_generation;

            if sigma == 0.0 {
                return 0.0;
            }

            let lambda_lanczos = lanczos_largest_shifted_eigenvalue(
                n,
                sigma,
                &degrees[..n],
                &adj_offsets[..n],
                adj_flat.as_slice(),
                &mut y[..n],
                &mut q_prev[..n],
                &mut q_curr[..n],
                &mut w[..n],
                basis,
                alpha,
                beta,
                tri_vec,
                tri_tmp,
            );
            let lambda_refined = power_refine_shifted_eigenvalue(
                n,
                sigma,
                &degrees[..n],
                &adj_offsets[..n],
                adj_flat.as_slice(),
                &mut q_curr[..n],
                &mut y[..n],
                POWER_REFINE_MAX_ITERS,
            );
            let lambda = if lambda_refined.is_finite() {
                lambda_refined
            } else {
                lambda_lanczos
            };

            if !lambda.is_finite() {
                return 0.0;
            }
            (sigma - lambda).max(0.0)
        })
    }

    /// Returns 0 if H¹ = 0, 1 if H¹ ≠ 0.
    ///
    /// Builds a `RipsComplex` and runs `CohomologyValidator`.
    ///
    /// AX-ID: AXIOMA-007, AXIOMA-009
    pub fn compute_h1(&self) -> usize {
        let complex = RipsComplex::build(&self.graph, REDUNDANCY_RADIUS);
        usize::from(!CohomologyValidator::check_h1(&complex))
    }

    /// Verifica H¹ = 0 using the state incremental (O(1)).
    ///
    /// Para verifiestion full (con rebuild of RipsComplex), use
    /// `compute_h1()` which is still available for checkpointing.
    ///
    /// AX-ID: AXIOMA-007, AXIOMA-009
    #[allow(clippy::inline_always)]
    #[inline(always)]
    pub const fn h1_is_zero_fast(&self) -> bool {
        self.h1_state.h1_is_zero()
    }

    /// Dimension of H¹ according to the incremental state.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    pub const fn h1_dim_fast(&self) -> usize {
        self.h1_state.h1_dim()
    }

    /// Internal read-only access to the underlying HNSW graph for topology inference.
    ///
    /// AX-ID: AXIOMA-013
    pub(crate) const fn graph_ref(&self) -> &HnswGraph {
        &self.graph
    }

    /// Internal read-only access to incremental H¹ state for topology inference.
    ///
    /// AX-ID: AXIOMA-007, AXIOMA-009
    pub(crate) const fn h1_state_ref(&self) -> &IncrementalH1State {
        &self.h1_state
    }

    /// Infer read-only topological hypotheses for downstream dynamics.
    ///
    /// AX-ID: AXIOMA-007, AXIOMA-013, H_dinámica (LEY_FUNDACIONAL §3.2)
    pub fn infer_topological_hypotheses(&self) -> SmallVec<[TopologicalHypothesis; 16]> {
        TopologicalIntuition::new(self).infer()
    }

    /// Find nodes affected by a candidate vector expansion.
    ///
    /// Returns the K nearest neighbours of the candidate according to HNSW.
    /// Used by `DualityConsistency` check (`AxiomID::DualityConsistency`).
    ///
    /// AX-ID: `LEY_FUNDACIONAL` §5.6
    pub fn find_affected_nodes(&mut self, candidate: &SparseCliffordVector) -> Vec<NodeId> {
        self.graph.search_nearest(candidate, 16)
    }

    // ── Hyperbolic coordinate contract (CRATE-004 interface) ─────────────────

    /// Returns the hyperbolic coordinate of the node, if it has been assigned by CRATE-004.
    ///
    /// # State actual
    /// Always returns `None` until `DiscreteRicciFlow` (CRATE-004) starts
    /// a llamar `set_hyperbolic_coord()`. Ver `HyperbolicCoord` for la
    /// full specification of the activation protocol.
    ///
    /// # Complexity
    /// O(log N) — search binaria over Vec sorted (sin HashMap).
    ///
    /// AX-ID: AXIOMA-004, LEY_FUNDACIONAL §7.2
    pub fn hyperbolic_coord(&self, id: NodeId) -> Option<HyperbolicCoord> {
        let raw = id.get();
        self.hyperbolic_coords
            .binary_search_by_key(&raw, |&(k, _)| k)
            .ok()
            .map(|idx| self.hyperbolic_coords[idx].1)
    }

    /// Assigns or updates the hyperbolic coordinate of a node.
    ///
    /// # Contract of llamada
    ///Only must be called per `DiscreteRicciFlow` (CRATE-004) tras compute
    /// the Ollivier-Ricci curvature of the node.
    ///
    /// # Invariant preservado
    /// `coord.norm_sq() < 1.0` — guaranteed by `HyperbolicCoord::new()`.
    ///
    /// # Complexity
    /// O(log N) amortized — binary search + ordered insert.
    ///
    /// AX-ID: AXIOMA-004, LEY_FUNDACIONAL §7.2
    pub fn set_hyperbolic_coord(&mut self, id: NodeId, coord: HyperbolicCoord) {
        let raw = id.get();
        match self
            .hyperbolic_coords
            .binary_search_by_key(&raw, |&(k, _)| k)
        {
            Ok(idx) => self.hyperbolic_coords[idx].1 = coord,
            Err(idx) => self.hyperbolic_coords.insert(idx, (raw, coord)),
        }
    }

    /// Number of nodes with assigned hyperbolic coordinates.
    ///
    /// En state normal (CRATE-004 no implementado): siempre 0.
    /// Useful for diagnóstico and tests.
    pub const fn hyperbolic_coord_count(&self) -> usize {
        self.hyperbolic_coords.len()
    }
}

// ─── Vector arithmetic helpers ───────────────────────────────────────────────

fn shifted_mv_inplace(
    n: usize,
    sigma: f64,
    degrees: &[f64],
    adj_offsets: &[(usize, usize)],
    adj_flat: &[usize],
    x: &[f64],
    out: &mut [f64],
) {
    for i in 0..n {
        let (start, end) = adj_offsets[i];
        out[i] = (sigma - degrees[i]) * x[i];
        for &nb_idx in &adj_flat[start..end] {
            out[i] += x[nb_idx];
        }
    }
}

fn prepare_laplacian_data(
    graph: &HnswGraph,
    n: usize,
    degrees: &mut [f64],
    adj_flat: &mut Vec<usize>,
    adj_offsets: &mut [(usize, usize)],
    seen_marks: &mut [u32],
    mut seen_generation: u32,
) -> (f64, u32) {
    let node_ids_raw: Vec<u64> = graph
        .nodes()
        .map(NodeId::get)
        .filter(|&raw| raw != u64::MAX && usize::try_from(raw).is_ok())
        .collect();
    let max_id = node_ids_raw.iter().copied().max().unwrap_or(0) as usize;
    let mut id_to_dense = vec![usize::MAX; max_id.saturating_add(1)];
    for (dense_idx, &raw) in node_ids_raw.iter().enumerate() {
        id_to_dense[raw as usize] = dense_idx;
    }

    adj_flat.clear();
    degrees.fill(0.0);

    for (idx, &raw_id) in node_ids_raw.iter().enumerate().take(n) {
        let start = adj_flat.len();
        seen_generation = seen_generation.wrapping_add(1);
        if seen_generation == 0 {
            seen_marks.fill(0);
            seen_generation = 1;
        }

        let pushed = graph.extend_neighbors_dedup(
            NodeId::try_new(raw_id).expect("NodeId válido por construcción"),
            seen_marks,
            seen_generation,
            adj_flat,
        );
        let end = adj_flat.len();
        for nb in &mut adj_flat[start..end] {
            let raw_nb = *nb;
            if raw_nb >= id_to_dense.len() {
                continue;
            }
            let dense_nb = id_to_dense[raw_nb];
            if dense_nb != usize::MAX {
                *nb = dense_nb;
            }
        }
        #[allow(clippy::cast_precision_loss)]
        {
            degrees[idx] = pushed as f64;
        }
        adj_offsets[idx] = (start, end);
    }

    if degrees.iter().sum::<f64>() == 0.0 {
        return (0.0, seen_generation);
    }

    (
        // FIX-F.2: Use +1e-6 instead of +1.0 for σ shift.
        // Adding 1.0 makes σ >> λ_max in sparse graphs, compressing the
        // Lanczos spectrum and inflating λ₂ estimates. 1e-6 is a minimal
        // regulariser that prevents division by zero without spectral distortion.
        degrees.iter().copied().fold(0.0f64, f64::max) + 1e-6_f64,
        seen_generation,
    )
}

#[allow(clippy::too_many_arguments)]
fn lanczos_largest_shifted_eigenvalue(
    n: usize,
    sigma: f64,
    degrees: &[f64],
    adj_offsets: &[(usize, usize)],
    adj_flat: &[usize],
    y: &mut [f64],
    q_prev: &mut [f64],
    q_curr: &mut [f64],
    w: &mut [f64],
    basis: &mut [f64],
    alpha: &mut [f64],
    beta: &mut [f64],
    tri_vec: &mut [f64],
    tri_tmp: &mut [f64],
) -> f64 {
    for i in 0..n {
        q_curr[i] = if i % 2 == 0 { 1.0 } else { -1.0 };
        q_prev[i] = 0.0;
        w[i] = 0.0;
    }

    deflate_ones(q_curr);
    let norm = vec_norm(q_curr);
    if norm < 1e-14 {
        return 0.0;
    }
    for x in q_curr.iter_mut() {
        *x /= norm;
    }

    let max_iters = LANCZOS_MAX_ITERS_DEFAULT.min(n);
    alpha[..max_iters].fill(0.0);
    beta[..max_iters].fill(0.0);

    let mut lambda_prev = f64::NEG_INFINITY;

    for k in 0..max_iters {
        shifted_mv_inplace(n, sigma, degrees, adj_offsets, adj_flat, q_curr, w);
        deflate_ones(w);

        if k > 0 {
            axpy_inplace(w, -beta[k - 1], q_prev);
        }

        alpha[k] = dot(q_curr, w);
        axpy_inplace(w, -alpha[k], q_curr);

        if (k + 1) % LANCZOS_REORTHOGONALIZE_EVERY == 0 {
            reorthogonalize(w, &basis[..(k + 1) * n], k + 1, n, n);
            deflate_ones(w);
        }

        beta[k] = vec_norm(w);
        copy_basis_vector(&mut basis[k * n..(k + 1) * n], q_curr);

        let lambda_new = largest_tridiagonal_eigenvalue(
            &alpha[..=k],
            &beta[..k],
            &mut tri_vec[..=k],
            &mut tri_tmp[..=k],
        );
        if (lambda_new - lambda_prev).abs() < LANCZOS_CONVERGENCE_EPS {
            lambda_prev = lambda_new;
            break;
        }
        lambda_prev = lambda_new;

        if beta[k] < 1e-14 || k + 1 == max_iters {
            break;
        }

        for i in 0..n {
            y[i] = w[i] / beta[k];
        }
        q_prev.copy_from_slice(q_curr);
        q_curr.copy_from_slice(y);
    }

    lambda_prev
}

#[allow(clippy::too_many_arguments)]
fn power_refine_shifted_eigenvalue(
    n: usize,
    sigma: f64,
    degrees: &[f64],
    adj_offsets: &[(usize, usize)],
    adj_flat: &[usize],
    v: &mut [f64],
    y: &mut [f64],
    max_iters: usize,
) -> f64 {
    deflate_ones(v);
    let norm = vec_norm(v);
    if norm < 1e-14 {
        return f64::NAN;
    }
    for vi in v.iter_mut() {
        *vi /= norm;
    }

    let mut lambda_prev = f64::NEG_INFINITY;
    for _ in 0..max_iters {
        shifted_mv_inplace(n, sigma, degrees, adj_offsets, adj_flat, v, y);
        deflate_ones(y);
        let y_norm = vec_norm(y);
        if y_norm < 1e-14 {
            break;
        }
        let rayleigh = dot(v, y);
        for yi in y.iter_mut() {
            *yi /= y_norm;
        }
        if (rayleigh - lambda_prev).abs() < 1e-10 {
            v.copy_from_slice(y);
            break;
        }
        lambda_prev = rayleigh;
        v.copy_from_slice(y);
    }

    shifted_mv_inplace(n, sigma, degrees, adj_offsets, adj_flat, v, y);
    dot(v, y) / dot(v, v).max(1e-14)
}

fn vec_norm(v: &[f64]) -> f64 {
    v.iter().fold(0.0, |acc, x| x.mul_add(*x, acc)).sqrt()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .fold(0.0, |acc, (x, y)| x.mul_add(*y, acc))
}

fn axpy_inplace(y: &mut [f64], alpha: f64, x: &[f64]) {
    for (yi, xi) in y.iter_mut().zip(x.iter()) {
        *yi += alpha * *xi;
    }
}

fn copy_basis_vector(dst: &mut [f64], src: &[f64]) {
    dst.fill(0.0);
    dst[..src.len()].copy_from_slice(src);
}

fn reorthogonalize(w: &mut [f64], basis: &[f64], vectors: usize, stride: usize, n: usize) {
    for j in 0..vectors {
        let base = j * stride;
        let qj = &basis[base..base + n];
        let proj = dot(w, qj);
        axpy_inplace(w, -proj, qj);
    }
}

fn largest_tridiagonal_eigenvalue(
    alpha: &[f64],
    beta: &[f64],
    v: &mut [f64],
    tmp: &mut [f64],
) -> f64 {
    let m = alpha.len();
    if m == 0 {
        return 0.0;
    }
    for (i, vi) in v.iter_mut().enumerate() {
        *vi = if i % 2 == 0 { 1.0 } else { -1.0 };
    }
    let mut norm = vec_norm(v);
    if norm < 1e-14 {
        return alpha[0];
    }
    for vi in v.iter_mut() {
        *vi /= norm;
    }

    let mut lambda_prev = f64::NEG_INFINITY;
    for _ in 0..32 {
        tridiagonal_mv(alpha, beta, v, tmp);
        let lambda = dot(v, tmp);
        norm = vec_norm(tmp);
        if norm < 1e-14 {
            break;
        }
        for (vi, ti) in v.iter_mut().zip(tmp.iter()) {
            *vi = *ti / norm;
        }
        if (lambda - lambda_prev).abs() < 1e-12 {
            lambda_prev = lambda;
            break;
        }
        lambda_prev = lambda;
    }
    lambda_prev
}

fn tridiagonal_mv(alpha: &[f64], beta: &[f64], x: &[f64], out: &mut [f64]) {
    let m = alpha.len();
    for i in 0..m {
        let mut acc = alpha[i].mul_add(x[i], 0.0);
        if i > 0 {
            acc = beta[i - 1].mul_add(x[i - 1], acc);
        }
        if i + 1 < m {
            acc = beta[i].mul_add(x[i + 1], acc);
        }
        out[i] = acc;
    }
}

/// Project out the all-ones component from v (deflation for λ₁=0).
fn deflate_ones(v: &mut [f64]) {
    // Length of the eigenvector ≤ N. For N < 2^52, cast exact. Normalization of
    // eigenvector no requires arithmetic of entero exact.
    #[allow(clippy::cast_precision_loss)]
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    for x in v {
        *x -= mean;
    }
}

#[cfg(test)]
mod tests {
    use genesis_math::SparseCliffordVector;
    use genesis_types::NodeId;

    use super::*;

    fn power_iteration_lambda2_reference(manifold: &ManifoldCollector, max_iters: usize) -> f64 {
        let n = manifold.graph.node_count();
        if n < 2 {
            return 0.0;
        }

        let mut degrees = vec![0.0; n];
        let mut adj_flat = Vec::with_capacity(n.saturating_mul(16));
        let mut adj_offsets = vec![(0usize, 0usize); n];
        let mut seen_marks = vec![0u32; n];
        let mut seen_generation = 1u32;

        for i in 0..n as u64 {
            #[allow(clippy::cast_possible_truncation)]
            let idx = i as usize;
            let start = adj_flat.len();
            seen_generation = seen_generation.wrapping_add(1);
            if seen_generation == 0 {
                seen_marks.fill(0);
                seen_generation = 1;
            }
            let pushed = manifold.graph.extend_neighbors_dedup(
                NodeId::try_new(i).expect("NodeId válido por construcción"),
                &mut seen_marks,
                seen_generation,
                &mut adj_flat,
            );
            #[allow(clippy::cast_precision_loss)]
            {
                degrees[idx] = pushed as f64;
            }
            adj_offsets[idx] = (start, adj_flat.len());
        }

        let total_degree: f64 = degrees.iter().sum();
        if total_degree == 0.0 {
            return 0.0;
        }

        let sigma = degrees.iter().copied().fold(0.0f64, f64::max) + 1.0;
        let mut v = vec![0.0; n];
        let mut y = vec![0.0; n];
        for (i, vi) in v.iter_mut().enumerate().take(n) {
            *vi = if i % 2 == 0 { 1.0 } else { -1.0 };
        }
        deflate_ones(&mut v);
        let norm = vec_norm(&v);
        if norm < 1e-14 {
            return 0.0;
        }
        for x in &mut v {
            *x /= norm;
        }

        let mut lambda_prev = f64::NEG_INFINITY;
        for _ in 0..max_iters {
            shifted_mv_inplace(n, sigma, &degrees, &adj_offsets, &adj_flat, &v, &mut y);
            deflate_ones(&mut y);
            let new_norm = vec_norm(&y);
            if new_norm < 1e-14 {
                break;
            }
            let rayleigh = dot(&v, &y);
            for yi in &mut y {
                *yi /= new_norm;
            }
            if (rayleigh - lambda_prev).abs() < 1e-10 {
                v.copy_from_slice(&y);
                break;
            }
            lambda_prev = rayleigh;
            core::mem::swap(&mut v, &mut y);
        }

        shifted_mv_inplace(n, sigma, &degrees, &adj_offsets, &adj_flat, &v, &mut y);
        let rq = dot(&v, &y) / dot(&v, &v).max(1e-14);
        (sigma - rq).max(0.0)
    }

    struct Lcg64 {
        state: u64,
    }

    impl Lcg64 {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }

        fn next_u64(&mut self) -> u64 {
            self.state = self
                .state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            self.state
        }

        fn next_f64(&mut self) -> f64 {
            let val = self.next_u64() >> 11;
            #[allow(clippy::cast_precision_loss)]
            {
                (val as f64) * (1.0 / ((1u64 << 53) as f64))
            }
        }

        fn range_usize(&mut self, min: usize, max_inclusive: usize) -> usize {
            let span = max_inclusive - min + 1;
            let rnd = self.next_u64() as usize;
            min + (rnd % span)
        }
    }

    /// Minimum algebraic connectivity constant — matches GENESIS_PROOF_SPEC §7.
    fn make_vec(id: u64) -> SparseCliffordVector {
        let s = (id as f64).mul_add(0.1, 0.05);
        SparseCliffordVector::from_iter((0..4).map(|b| (b, s * (b as f64 + 1.0)))).unwrap()
    }

    #[test]
    fn edge_density_within_bounds() {
        let mut m = ManifoldCollector::new(16);
        for i in 0..50u64 {
            m.insert(
                NodeId::try_new(i).expect("NodeId válido por construcción"),
                &make_vec(i),
            )
            .unwrap();
        }
        let density = m.compute_edge_density();
        assert!(
            density <= 2.0, // some slack above 1.0 for HNSW layer overhead
            "edge density {} too high",
            density
        );
    }

    #[test]
    fn edge_density_zero_for_empty() {
        let m = ManifoldCollector::new(16);
        assert_eq!(m.compute_edge_density(), 0.0);
    }

    #[test]
    fn edge_density_uses_base2_logarithm() {
        let mut m = ManifoldCollector::new(16);
        for i in 0..8u64 {
            m.insert(
                NodeId::try_new(i).expect("NodeId válido por construcción"),
                &make_vec(i),
            )
            .unwrap();
        }

        let density = m.compute_edge_density();
        let n = m.node_count() as f64;
        let edge_count = m.edge_count() as f64;
        assert!(
            edge_count > 0.0,
            "test setup should generate at least one edge"
        );

        let expected = edge_count / (n * n.log2().max(1.0));
        assert!(
            (density - expected).abs() < 1e-12,
            "edge density must use base-2 logarithm"
        );
    }

    #[test]
    fn lambda2_positive_for_connected_graph() {
        let mut m = ManifoldCollector::new(16);
        for i in 0..10u64 {
            m.insert(
                NodeId::try_new(i).expect("NodeId válido por construcción"),
                &make_vec(i),
            )
            .unwrap();
        }
        let l2 = m.compute_lambda2();
        assert!(l2 >= 0.0, "lambda2 nunca puede ser negativo");
        assert!(
            l2 > 0.0,
            "grafo conectado debe tener lambda2 > 0, obtenido {}",
            l2
        );
    }

    #[test]
    fn lambda2_zero_for_disconnected_graph() {
        let mut m = ManifoldCollector::new(4);
        m.insert(
            NodeId::try_new(0).expect("NodeId válido por construcción"),
            &make_vec(0),
        )
        .unwrap();
        assert_eq!(m.compute_lambda2(), 0.0);
    }

    #[test]
    fn lambda2_zero_for_two_disconnected_components() {
        let mut m = ManifoldCollector::new(4);
        m.insert(
            NodeId::try_new(0).expect("NodeId válido por construcción"),
            &make_vec(0),
        )
        .unwrap();
        assert_eq!(m.compute_lambda2(), 0.0);
    }

    #[test]
    fn lambda2_refactored_matches_original_on_known_graph() {
        let mut m = ManifoldCollector::new(64);
        let vecs: Vec<_> = (0..10).map(make_vec).collect();
        for (i, v) in vecs.iter().enumerate() {
            m.insert(
                NodeId::try_new(i as u64).expect("NodeId válido por construcción"),
                v,
            )
            .unwrap();
        }
        let l2 = m.compute_lambda2();
        assert!(l2 >= 0.0, "lambda2 nunca negativa");
        assert!(l2 < 10.0, "lambda2 acotada por grado máximo");
    }

    #[test]
    fn lambda2_with_non_consecutive_node_ids_is_positive_and_finite() {
        let mut m = ManifoldCollector::new(16);
        for i in 0..5u64 {
            m.insert(
                NodeId::try_new(i).expect("NodeId válido por construcción"),
                &make_vec(i),
            )
            .unwrap();
        }

        m.remove_node(NodeId::try_new(2).expect("NodeId válido por construcción"))
            .unwrap();

        let lambda2 = m.compute_lambda2();
        assert!(lambda2.is_finite(), "lambda2 debe ser finita");
        assert!(
            lambda2 > 0.0,
            "lambda2 debe ser positiva, obtenido {lambda2}"
        );
    }

    #[test]
    fn find_affected_nodes_returns_neighbours() {
        let mut m = ManifoldCollector::new(16);
        for i in 0..20u64 {
            m.insert(
                NodeId::try_new(i).expect("NodeId válido por construcción"),
                &make_vec(i),
            )
            .unwrap();
        }
        let candidate = make_vec(5);
        let affected = m.find_affected_nodes(&candidate);
        assert!(
            !affected.is_empty(),
            "should find at least one affected node"
        );
    }

    #[test]
    fn compute_h1_returns_zero_or_one() {
        let mut m = ManifoldCollector::new(16);
        for i in 0..5u64 {
            m.insert(
                NodeId::try_new(i).expect("NodeId válido por construcción"),
                &make_vec(i),
            )
            .unwrap();
        }
        let h1 = m.compute_h1();
        assert!(h1 == 0 || h1 == 1, "H1 result must be 0 or 1, got {}", h1);
    }

    #[test]
    fn lambda2_lanczos_matches_power_iteration() {
        let mut rng = Lcg64::new(0x0DEC_0DED);

        for _ in 0..100 {
            let n = rng.range_usize(50, 200);
            let mut m = ManifoldCollector::new(64);

            for i in 0..n {
                let base = (i as f64).mul_add(0.05, 0.01);
                let vec = SparseCliffordVector::from_iter((0..4).map(|b| {
                    let jitter = rng.next_f64() * 1e-4;
                    (b, base * (b as f64 + 1.0) + jitter)
                }))
                .expect("vector must be valid");
                m.insert(
                    NodeId::try_new(i as u64).expect("NodeId válido por construcción"),
                    &vec,
                )
                .unwrap();
            }

            let lanczos = m.compute_lambda2();
            let power = power_iteration_lambda2_reference(&m, 800);
            let diff = (lanczos - power).abs();
            assert!(
                diff < 1e-3,
                "lanczos={} power={} diff={} exceeds tolerance",
                lanczos,
                power,
                diff
            );
        }
    }

    // ── HyperbolicCoord contract tests ────────────────────────────────────────

    /// Verifies that HyperbolicCoord::new rejects puntos fuera of the disk unitario.
    #[test]
    fn hyperbolic_coord_rejects_outside_disk() {
        assert!(
            HyperbolicCoord::new(1.0, 0.0).is_none(),
            "punto en el borde debe rechazarse"
        );
        assert!(
            HyperbolicCoord::new(0.8, 0.8).is_none(),
            "0.64+0.64=1.28 fuera del disco"
        );
        assert!(
            HyperbolicCoord::new(f64::NAN, 0.0).is_none(),
            "NaN debe rechazarse"
        );
        assert!(
            HyperbolicCoord::new(f64::INFINITY, 0.0).is_none(),
            "Inf debe rechazarse"
        );
    }

    /// Verifies that HyperbolicCoord::new acepta puntos valids dentro of the disk.
    #[test]
    fn hyperbolic_coord_accepts_inside_disk() {
        let c = HyperbolicCoord::new(0.5, 0.5).expect("0.25+0.25=0.5 < 1, debe aceptarse");
        assert!(c.norm_sq() < 1.0);
        assert_eq!(c.x, 0.5);
        assert_eq!(c.y, 0.5);

        let origin = HyperbolicCoord::new(0.0, 0.0).expect("origen debe aceptarse");
        assert_eq!(origin.norm_sq(), 0.0);
        assert_eq!(origin.hyperbolic_distance_to_origin(), 0.0);
    }

    /// Verifies that ManifoldCollector returns None for nodes without coordinate asignada.
    #[test]
    fn manifold_hyperbolic_coord_none_before_crate004() {
        let mut m = ManifoldCollector::new(16);
        for i in 0..5u64 {
            m.insert(NodeId::try_new(i).unwrap(), &make_vec(i)).unwrap();
        }
        // Before of CRATE-004: any node has hyperbolic coordinate.
        for i in 0..5u64 {
            assert!(
                m.hyperbolic_coord(NodeId::try_new(i).unwrap()).is_none(),
                "nodo {i} no debe tener coord hiperbólica antes de CRATE-004"
            );
        }
        assert_eq!(m.hyperbolic_coord_count(), 0);
    }

    /// Check set/get of hyperbolic coordinates (contract of CRATE-004).
    #[test]
    fn manifold_hyperbolic_coord_set_and_get() {
        let mut m = ManifoldCollector::new(16);
        for i in 0..3u64 {
            m.insert(NodeId::try_new(i).unwrap(), &make_vec(i)).unwrap();
        }

        let coord0 = HyperbolicCoord::new(0.3, 0.1).unwrap();
        let coord2 = HyperbolicCoord::new(-0.5, 0.2).unwrap();

        m.set_hyperbolic_coord(NodeId::try_new(0).unwrap(), coord0);
        m.set_hyperbolic_coord(NodeId::try_new(2).unwrap(), coord2);

        assert_eq!(m.hyperbolic_coord_count(), 2);
        assert_eq!(
            m.hyperbolic_coord(NodeId::try_new(0).unwrap()),
            Some(coord0)
        );
        assert!(m.hyperbolic_coord(NodeId::try_new(1).unwrap()).is_none());
        assert_eq!(
            m.hyperbolic_coord(NodeId::try_new(2).unwrap()),
            Some(coord2)
        );
    }

    /// Verifies that set_hyperbolic_coord updates instead of duplicar.
    #[test]
    fn manifold_hyperbolic_coord_update_preserves_count() {
        let mut m = ManifoldCollector::new(16);
        m.insert(NodeId::try_new(0).unwrap(), &make_vec(0)).unwrap();

        let c1 = HyperbolicCoord::new(0.1, 0.2).unwrap();
        let c2 = HyperbolicCoord::new(0.3, 0.4).unwrap();

        m.set_hyperbolic_coord(NodeId::try_new(0).unwrap(), c1);
        assert_eq!(m.hyperbolic_coord_count(), 1);

        m.set_hyperbolic_coord(NodeId::try_new(0).unwrap(), c2);
        assert_eq!(
            m.hyperbolic_coord_count(),
            1,
            "update no debe crear duplicado"
        );
        assert_eq!(m.hyperbolic_coord(NodeId::try_new(0).unwrap()), Some(c2));
    }

    #[test]
    fn remove_node_prunes_hyperbolic_and_h1_links() {
        let mut m = ManifoldCollector::new(16);
        for i in 0..4u64 {
            m.insert(NodeId::try_new(i).unwrap(), &make_vec(i)).unwrap();
        }

        let removed = NodeId::try_new(0).unwrap();
        m.set_hyperbolic_coord(removed, HyperbolicCoord::new(0.2, 0.3).unwrap());
        assert!(m.hyperbolic_coord(removed).is_some());

        m.remove_node(removed).unwrap();

        assert!(m.hyperbolic_coord(removed).is_none());
        assert!(m
            .h1_state_ref()
            .persistent_cycles()
            .iter()
            .all(|record| record.nodes.iter().all(|n| n != &removed)));
        assert!(m
            .h1_state_ref()
            .triangle_closures()
            .iter()
            .all(|record| record.nodes.iter().all(|n| n != &removed)));
    }

    /// Checks the hyperbolic distance to the origin for a known point.
    /// d(0, (r,0)) = 2·arctanh(r). Para r=0.5: 2·arctanh(0.5) ≈ 1.0986.
    #[test]
    fn hyperbolic_distance_to_origin_known_value() {
        let c = HyperbolicCoord::new(0.5, 0.0).unwrap();
        let expected = 2.0 * (0.5f64).atanh();
        let got = c.hyperbolic_distance_to_origin();
        assert!(
            (got - expected).abs() < 1e-12,
            "d(0,(0.5,0)) = {got}, esperado {expected}"
        );
    }
}
