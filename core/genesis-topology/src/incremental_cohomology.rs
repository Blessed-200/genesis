//! TDA Incremental: mantenimiento exacto de H¹(M, F) = 0.
//!
//! Este módulo reemplaza el full-rebuild O(N·K²) por actualizaciones incrementales
//! con complejidad amortizada O(K·α(N) + K·L) por inserción, donde:
//! - K es el grado medio del grafo HNSW (acotado ≈ 32)
//! - α(N) es la función inversa de Ackermann (≤ 5 para N < 10^80)
//! - L es la longitud media de ciclos en el grafo (≈ O(1) para HNSW)
//!
//! El invariante H¹ = dim(ker ∂₁) − dim(im ∂₂) = 0 se mantiene exactamente.
//!
//! AX-ID: AXIOMA-007, AXIOMA-009, H_restricción (LEY_FUNDACIONAL §3.5)

use genesis_types::NodeId;
use smallvec::SmallVec;

// ── Union-Find para ∂₁ ───────────────────────────────────────────────────────

/// Union-Find con compresión de caminos y unión por rango para rank_d1 = N - c.
pub(crate) struct PersistentUnionFind {
    parent:     Vec<u32>,
    rank:       Vec<u8>,
    components: usize,
    num_nodes:  usize,
}

impl PersistentUnionFind {
    pub fn new() -> Self {
        Self { parent: Vec::new(), rank: Vec::new(), components: 0, num_nodes: 0 }
    }

    pub fn add_node(&mut self) {
        let idx = self.num_nodes;
        if idx >= self.parent.len() {
            self.parent.push(idx as u32);
            self.rank.push(0);
        } else {
            self.parent[idx] = idx as u32;
            self.rank[idx] = 0;
        }
        self.num_nodes += 1;
        self.components += 1;
    }

    pub fn find(&mut self, x: usize) -> usize {
        // Pass 1: walk to root.
        let mut root = x;
        while self.parent[root] as usize != root {
            root = self.parent[root] as usize;
        }
        // Pass 2: path compression — point every node on the path directly to root.
        // Guarantees amortised O(α(N)) per operation (inverse Ackermann).
        // The former single-pass "halving" only compressed partially and degraded amortisation.
        let mut node = x;
        while self.parent[node] as usize != root {
            let next = self.parent[node] as usize;
            self.parent[node] = root as u32;
            node = next;
        }
        root
    }

    pub fn union(&mut self, u: usize, v: usize) -> bool {
        let ru = self.find(u);
        let rv = self.find(v);
        if ru == rv { return false; }
        match self.rank[ru].cmp(&self.rank[rv]) {
            std::cmp::Ordering::Less    => self.parent[ru] = rv as u32,
            std::cmp::Ordering::Greater => self.parent[rv] = ru as u32,
            std::cmp::Ordering::Equal   => {
                self.parent[rv] = ru as u32;
                self.rank[ru] = self.rank[ru].saturating_add(1);
            }
        }
        self.components -= 1;
        true
    }

    #[inline]
    pub fn rank_d1(&self) -> usize {
        self.num_nodes.saturating_sub(self.components)
    }
}

// ── IncrementalD2: Base escalonada de im(∂₂) ─────────────────────────────────

enum Column {
    Sparse(SmallVec<[u32; 8]>),
    Dense(Box<[u64]>),
}

const DENSE_THRESHOLD: usize = 64;

pub(crate) struct IncrementalD2 {
    base_cols:   Vec<Column>,
    pivot_row:   Vec<u32>,   // edge_idx -> index in base_cols (u32::MAX = free)
    /// Clearance flags (BN-09): once a base column has been used as a pivot in a
    /// reduction that produced a zero column (triangle boundary = exact cycle),
    /// it is marked cleared and skipped in future XOR traversals.
    /// This mirrors the Ripser clearance algorithm for persistent homology.
    cleared:     Vec<bool>,
    num_edges:   usize,
    rank:        usize,
}

impl IncrementalD2 {
    pub fn new() -> Self {
        Self {
            base_cols: Vec::new(),
            pivot_row: Vec::new(),
            cleared:   Vec::new(),
            num_edges: 0,
            rank: 0,
        }
    }

    pub fn register_edge(&mut self, edge_idx: u32) {
        let idx = edge_idx as usize;
        if idx >= self.pivot_row.len() {
            self.pivot_row.resize(idx + 1, u32::MAX);
        }
        self.num_edges = self.num_edges.max(idx + 1);
    }

    pub fn add_triangle(&mut self, mut e1: u32, mut e2: u32, mut e3: u32) -> bool {
        if e1 > e2 { std::mem::swap(&mut e1, &mut e2); }
        if e2 > e3 { std::mem::swap(&mut e2, &mut e3); }
        if e1 > e2 { std::mem::swap(&mut e1, &mut e2); }

        let mut col = Column::Sparse({
            let mut sv: SmallVec<[u32; 8]> = SmallVec::new();
            sv.push(e1); sv.push(e2); sv.push(e3);
            sv
        });

        loop {
            let pivot = match &col {
                Column::Sparse(sv) => sv.first().copied(),
                Column::Dense(bm)  => first_set_bit(bm),
            };

            let p = match pivot {
                None    => return false,
                Some(p) => p as usize,
            };

            if p >= self.pivot_row.len() || self.pivot_row[p] == u32::MAX {
                if p >= self.pivot_row.len() { self.pivot_row.resize(p + 1, u32::MAX); }
                let col_idx = self.base_cols.len();
                self.pivot_row[p] = col_idx as u32;
                self.base_cols.push(col);
                // Extend cleared flags to match base_cols length
                if self.cleared.len() <= col_idx {
                    self.cleared.resize(col_idx + 1, false);
                }
                self.rank += 1;
                return true;
            }

            let base_idx = self.pivot_row[p] as usize;

            // BN-09 / FIX-4 Clearance: if this base column was already used in a reduction
            // that produced zero (it's been cleared), skip it.
            //
            // CRITICAL: do NOT mutate pivot_row[p] = u32::MAX here.
            // Setting pivot_row[p] = u32::MAX during a live reduction corrupts the
            // pivot index and breaks future reductions that need to find that pivot.
            // The correct Ripser clearance contract is: skip the column in XOR traversal,
            // but leave pivot_row[p] intact so subsequent insertions can use it.
            // Compaction of cleared columns is done during a checkpoint sweep (future
            // run_checkpoint() when base_cols grows large), not during per-triangle reduction.
            if base_idx < self.cleared.len() && self.cleared[base_idx] {
                // Column cleared — skip without touching pivot_row.
                continue;
            }

            let next_col = xor_columns_opt(col, &self.base_cols[base_idx], self.num_edges);

            // If result is zero, this triangle was a boundary (no new H¹).
            // Mark the base column as cleared — it produced a witness.
            let is_zero = match &next_col {
                Column::Sparse(sv) => sv.is_empty(),
                Column::Dense(bm)  => bm.iter().all(|&w| w == 0),
            };
            if is_zero {
                if base_idx < self.cleared.len() {
                    self.cleared[base_idx] = true;
                }
            }

            col = next_col;
        }
    }

    #[inline] pub fn rank(&self) -> usize { self.rank }
}

/// XOR of two boundary columns — optimised version used by add_triangle (BN-09).
///
/// Dispatches to an AVX-512 vectorised kernel for the dense×dense case when the
/// target CPU supports AVX-512F. Falls back to scalar on other targets.
/// All other cases (sparse×sparse, sparse×dense, dense×sparse) are identical
/// to the scalar path since they are already memory-bound, not compute-bound.
#[inline]
fn xor_columns_opt(a: Column, b: &Column, num_edges: usize) -> Column {
    match (a, b) {
        (Column::Dense(mut da), Column::Dense(db)) => {
            // AVX-512 path: XOR 512 bits (8 × u64) per instruction with prefetch.
            #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
            // SAFETY: AVX-512 XOR on aligned data guaranteed by caller contract.
            unsafe {
                use std::arch::x86_64::*;
                let len = da.len().min(db.len());
                let mut i = 0usize;
                while i + 8 <= len {
                    // Prefetch the next 512-bit chunk of source into L1.
                    if i + 16 < len {
                        _mm_prefetch(
                            db.as_ptr().add(i + 8) as *const i8,
                            _MM_HINT_T0,
                        );
                    }
                    let t = da.as_mut_ptr().add(i) as *mut __m512i;
                    let s = db.as_ptr().add(i) as *const __m512i;
                    let result = _mm512_xor_si512(_mm512_loadu_si512(t), _mm512_loadu_si512(s));
                    _mm512_storeu_si512(t, result);
                    i += 8;
                }
                // Scalar tail for remainder
                for j in i..len { da[j] ^= db[j]; }
            }
            // Scalar fallback (also used when AVX-512 is not available)
            #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
            {
                for (aw, &bw) in da.iter_mut().zip(db.iter()) { *aw ^= bw; }
            }
            if da.iter().all(|&w| w == 0) { Column::Sparse(SmallVec::new()) }
            else { Column::Dense(da) }
        }
        // All other combinations delegate to the existing scalar path.
        other => xor_columns(other.0, other.1, num_edges),
    }
}

fn xor_columns(a: Column, b: &Column, num_edges: usize) -> Column {
    match (a, b) {
        (Column::Sparse(sa), Column::Sparse(sb)) => {
            let mut result: SmallVec<[u32; 8]> = SmallVec::new();
            let (mut ia, mut ib) = (0, 0);
            while ia < sa.len() && ib < sb.len() {
                match sa[ia].cmp(&sb[ib]) {
                    std::cmp::Ordering::Less    => { result.push(sa[ia]); ia += 1; }
                    std::cmp::Ordering::Greater => { result.push(sb[ib]); ib += 1; }
                    std::cmp::Ordering::Equal   => { ia += 1; ib += 1; }
                }
            }
            result.extend_from_slice(&sa[ia..]);
            result.extend_from_slice(&sb[ib..]);
            if result.len() > DENSE_THRESHOLD {
                Column::Dense(sparse_to_dense(&result, num_edges))
            } else {
                Column::Sparse(result)
            }
        }
        (Column::Sparse(sa), Column::Dense(db)) => {
            let mut bm = db.to_vec().into_boxed_slice();
            for &idx in &sa {
                let word = idx as usize / 64;
                let bit  = idx as usize % 64;
                if word < bm.len() { bm[word] ^= 1u64 << bit; }
            }
            if bm.iter().all(|&w| w == 0) { Column::Sparse(SmallVec::new()) }
            else { Column::Dense(bm) }
        }
        (Column::Dense(mut da), Column::Sparse(sb)) => {
            for &idx in sb.iter() {
                let word = idx as usize / 64;
                let bit  = idx as usize % 64;
                if word < da.len() { da[word] ^= 1u64 << bit; }
            }
            if da.iter().all(|&w| w == 0) { Column::Sparse(SmallVec::new()) }
            else { Column::Dense(da) }
        }
        (Column::Dense(mut da), Column::Dense(db)) => {
            for (aw, &bw) in da.iter_mut().zip(db.iter()) { *aw ^= bw; }
            if da.iter().all(|&w| w == 0) { Column::Sparse(SmallVec::new()) }
            else { Column::Dense(da) }
        }
    }
}

fn sparse_to_dense(sv: &[u32], num_edges: usize) -> Box<[u64]> {
    let n_words = (num_edges + 63) / 64;
    let mut bm = vec![0u64; n_words.max(1)].into_boxed_slice();
    for &idx in sv {
        let word = idx as usize / 64;
        let bit  = idx as usize % 64;
        if word < bm.len() { bm[word] |= 1u64 << bit; }
    }
    bm
}

fn first_set_bit(bm: &[u64]) -> Option<u32> {
    for (i, &w) in bm.iter().enumerate() {
        if w != 0 {
            return Some((i * 64 + w.trailing_zeros() as usize) as u32);
        }
    }
    None
}

// ── Estado unificado de H¹ incremental ───────────────────────────────────────

type EdgeKey = u128;

#[inline]
fn edge_key(u: NodeId, v: NodeId) -> EdgeKey {
    let (a, b) = if u <= v { (u.get(), v.get()) } else { (v.get(), u.get()) };
    (a as u128) << 64 | b as u128
}

/// Estado incremental de H¹(M, F).
///
/// Mantiene exactamente el invariante dim(ker ∂₁) − dim(im ∂₂) = 0
/// con complejidad amortizada O(K·α(N) + K·L) por inserción.
///
/// AX-ID: AXIOMA-007, AXIOMA-009
/// Edge registry: sorted `Vec<(EdgeKey, u32)>` — binary search O(log E).
///
/// # Why not HashMap (AGENTS prohibition compliance)
///
/// `edge_map` was previously `HashMap<EdgeKey, u32>`. While HNSW search is
/// the canonical hot path, `add_edge` is called O(N×K) during manifold construction
/// (K ≤ M0 = 32 per node). With N=10⁶ that is ~32M calls — too frequent for
/// HashMap's per-call allocation and cache-unfriendly bucket traversal.
///
/// Sorted `Vec` + `binary_search` gives O(log E) lookup with cache-friendly
/// sequential layout (EdgeKey = u128, so entries are 24 bytes → ~2.7 entries/cache line).
/// `insert` is O(E) shift in the worst case, but HNSW insertions are mostly sequential
/// (IDs are assigned in order) so new keys land near the end → amortised O(1) shift.
pub struct IncrementalH1State {
    uf:                   PersistentUnionFind,
    d2:                   IncrementalD2,
    /// Sorted (EdgeKey, edge_id) pairs. Binary search for O(log E) lookup.
    edge_map:             Vec<(EdgeKey, u32)>,
    num_edges:            usize,
    ops_since_checkpoint: usize,
}

const CHECKPOINT_INTERVAL: usize = 1_000_000;

impl IncrementalH1State {
    /// Creates an empty H¹ state with no nodes or edges.
    pub fn new() -> Self {
        Self {
            uf:                   PersistentUnionFind::new(),
            d2:                   IncrementalD2::new(),
            edge_map:             Vec::new(),
            num_edges:            0,
            ops_since_checkpoint: 0,
        }
    }

    /// Añade un nodo al Union-Find.
    pub fn add_node(&mut self) {
        self.uf.add_node();
    }

    /// Registra una arista (u, v). Idempotente si ya existe.
    /// Retorna el ID de arista (existente o nuevo).
    pub fn add_edge(&mut self, u: NodeId, v: NodeId) -> u32 {
        let key = edge_key(u, v);
        match self.edge_map.binary_search_by_key(&key, |&(k, _)| k) {
            Ok(pos) => return self.edge_map[pos].1, // already present
            Err(ins) => {
                // Insert at sorted position — O(E) shift but sequential IDs
                // make this near-O(1) amortised in practice.
                let edge_id = self.num_edges as u32;
                self.edge_map.insert(ins, (key, edge_id));
                self.d2.register_edge(edge_id);
                self.num_edges += 1;
                self.uf.union(u.get() as usize, v.get() as usize);

                self.ops_since_checkpoint += 1;
                if self.ops_since_checkpoint >= CHECKPOINT_INTERVAL {
                    self.run_checkpoint();
                }

                edge_id
            }
        }
    }

    /// Registra un triángulo (a, b, c) si todas sus aristas existen.
    pub fn add_triangle(&mut self, a: NodeId, b: NodeId, c: NodeId) {
        if let (Some(e1), Some(e2), Some(e3)) = (
            self.lookup_edge(a, b),
            self.lookup_edge(a, c),
            self.lookup_edge(b, c),
        ) {
            if self.d2.add_triangle(e1, e2, e3) {
                self.ops_since_checkpoint += 1;
                if self.ops_since_checkpoint >= CHECKPOINT_INTERVAL {
                    self.run_checkpoint();
                }
            }
        }
    }

    fn lookup_edge(&self, u: NodeId, v: NodeId) -> Option<u32> {
        let key = edge_key(u, v);
        self.edge_map.binary_search_by_key(&key, |&(k, _)| k)
            .ok()
            .map(|pos| self.edge_map[pos].1)
    }

    fn run_checkpoint(&mut self) {
        // Placeholder para compactación de base o validación externa.
        self.ops_since_checkpoint = 0;
    }

    /// Retorna true si H¹ = 0 (invariante de cohomología satisfecho). O(1).
    ///
    /// AX-ID: AXIOMA-007, AXIOMA-009
    pub fn h1_is_zero(&self) -> bool {
        let rank_d1 = self.uf.rank_d1();
        let rank_d2 = self.d2.rank();
        let dim_ker_d1 = self.num_edges.saturating_sub(rank_d1);
        dim_ker_d1.saturating_sub(rank_d2) == 0
    }

    /// Dimensión de H¹ según el estado incremental. O(1).
    pub fn h1_dim(&self) -> usize {
        let rank_d1 = self.uf.rank_d1();
        let rank_d2 = self.d2.rank();
        self.num_edges
            .saturating_sub(rank_d1)
            .saturating_sub(rank_d2)
    }
}

impl Default for IncrementalH1State {
    fn default() -> Self { Self::new() }
}

// ── Tests ─────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    fn node(n: u64) -> NodeId { NodeId::try_new(n).expect("NodeId válido") }

    #[test]
    fn single_triangle_h1_zero() {
        let mut s = IncrementalH1State::new();
        let (a, b, c) = (node(0), node(1), node(2));
        for _ in 0..3 { s.add_node(); }
        s.add_edge(a, b);
        s.add_edge(b, c);
        s.add_edge(a, c);
        s.add_triangle(a, b, c);
        assert!(s.h1_is_zero(), "Triángulo relleno debe tener H¹ = 0");
    }

    #[test]
    fn cycle_without_fill_h1_nonzero() {
        let mut s = IncrementalH1State::new();
        let (a, b, c) = (node(0), node(1), node(2));
        for _ in 0..3 { s.add_node(); }
        s.add_edge(a, b);
        s.add_edge(b, c);
        s.add_edge(a, c);
        // Sin add_triangle → H¹ ≠ 0
        assert!(!s.h1_is_zero(), "Ciclo sin relleno debe tener H¹ ≠ 0");
        assert_eq!(s.h1_dim(), 1);
    }

    #[test]
    fn empty_graph_h1_zero() {
        let s = IncrementalH1State::new();
        assert!(s.h1_is_zero(), "Grafo vacío: H¹ = 0");
    }

    #[test]
    fn add_edge_idempotent() {
        let mut s = IncrementalH1State::new();
        let (a, b) = (node(0), node(1));
        s.add_node(); s.add_node();
        let id1 = s.add_edge(a, b);
        let id2 = s.add_edge(a, b);
        assert_eq!(id1, id2, "add_edge debe ser idempotente");
        assert_eq!(s.num_edges, 1);
    }

    #[test]
    fn two_triangles_sharing_edge_h1_zero() {
        let mut s = IncrementalH1State::new();
        let nodes: Vec<NodeId> = (0..4).map(|i| { s.add_node(); node(i) }).collect();
        let (a, b, c, d) = (nodes[0], nodes[1], nodes[2], nodes[3]);
        // Triángulo 1: a-b-c
        s.add_edge(a, b); s.add_edge(b, c); s.add_edge(a, c);
        s.add_triangle(a, b, c);
        // Triángulo 2: a-b-d
        s.add_edge(a, d); s.add_edge(b, d);
        s.add_triangle(a, b, d);
        assert!(s.h1_is_zero());
    }
}
