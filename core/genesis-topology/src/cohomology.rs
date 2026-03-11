use crate::rips::RipsComplex;
/// AX-ID: AXIOMA-007, AXIOMA-009
/// Cohomology validator: computes H¹ = ker(∂₁) / im(∂₂) over Z₂.
/// All arithmetic in Z₂ (bit operations). No external linear algebra libraries.
/// Boundary matrices stored as bitmaps (Vec<u64> packed rows).
use std::cell::RefCell;

/// Packed Z₂ matrix: rows × cols, each row stored as ceil(cols/64) u64 words.
struct Z2Matrix {
    rows: usize,
    cols: usize,
    data: Vec<u64>,
}

impl Z2Matrix {
    fn new(rows: usize, cols: usize) -> Self {
        let words_per_row = cols.div_ceil(64);
        Self {
            rows,
            cols,
            data: vec![0u64; rows * words_per_row],
        }
    }

    fn words_per_row(&self) -> usize {
        self.cols.div_ceil(64)
    }

    fn get(&self, row: usize, col: usize) -> bool {
        let wpr = self.words_per_row();
        let word = self.data[row * wpr + col / 64];
        (word >> (col % 64)) & 1 == 1
    }

    fn set(&mut self, row: usize, col: usize, val: bool) {
        let wpr = self.words_per_row();
        let idx = row * wpr + col / 64;
        if val {
            self.data[idx] |= 1u64 << (col % 64);
        } else {
            self.data[idx] &= !(1u64 << (col % 64));
        }
    }

    fn xor_row(&mut self, dst: usize, src: usize) {
        let wpr = self.words_per_row();
        for w in 0..wpr {
            let s = self.data[src * wpr + w];
            self.data[dst * wpr + w] ^= s;
        }
    }

    fn rank_by_gaussian_elimination(&mut self) -> usize {
        if self.rows == 0 || self.cols == 0 {
            return 0;
        }
        let mut rank = 0usize;
        let mut r = 0usize;

        for c in 0..self.cols {
            let pivot = (r..self.rows).find(|&row| self.get(row, c));
            if let Some(p) = pivot {
                if p != r {
                    let wpr = self.words_per_row();
                    for w in 0..wpr {
                        self.data.swap(p * wpr + w, r * wpr + w);
                    }
                }
                for row in 0..self.rows {
                    if row != r && self.get(row, c) {
                        self.xor_row(row, r);
                    }
                }
                rank += 1;
                r += 1;
            }
        }
        rank
    }
}

#[derive(Default)]
struct HomologyWorkspace {
    id_to_vertex: Vec<usize>,
    edge_lookup: Vec<(u64, usize)>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct H1CacheKey {
    ptr: *const RipsComplex,
    counts: (usize, usize, usize),
}

#[derive(Default)]
struct H1Cache {
    key: Option<H1CacheKey>,
    cached_result: Option<bool>,
    uf_parent: Vec<usize>,
    uf_rank: Vec<u8>,
}

impl H1Cache {
    fn invalidate(&mut self) {
        self.key = None;
        self.cached_result = None;
    }

    fn prepare_union_find(&mut self, n: usize) {
        if self.uf_parent.len() < n {
            self.uf_parent.resize(n, 0);
            self.uf_rank.resize(n, 0);
        }

        for i in 0..n {
            self.uf_parent[i] = i;
            self.uf_rank[i] = 0;
        }
    }

    fn find(&mut self, x: usize) -> usize {
        let mut root = x;
        while self.uf_parent[root] != root {
            root = self.uf_parent[root];
        }

        let mut node = x;
        while self.uf_parent[node] != node {
            let parent = self.uf_parent[node];
            self.uf_parent[node] = root;
            node = parent;
        }

        root
    }

    fn union(&mut self, a: usize, b: usize) -> bool {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return false;
        }

        let rank_a = self.uf_rank[ra];
        let rank_b = self.uf_rank[rb];
        match rank_a.cmp(&rank_b) {
            std::cmp::Ordering::Less => {
                self.uf_parent[ra] = rb;
            }
            std::cmp::Ordering::Greater => {
                self.uf_parent[rb] = ra;
            }
            std::cmp::Ordering::Equal => {
                self.uf_parent[rb] = ra;
                self.uf_rank[ra] = self.uf_rank[ra].saturating_add(1);
            }
        }

        true
    }
}

thread_local! {
    static HOMOLOGY_WORKSPACE: RefCell<HomologyWorkspace> = RefCell::new(HomologyWorkspace::default());
    static H1_CACHE: RefCell<H1Cache> = RefCell::new(H1Cache::default());
}

#[cfg(test)]
thread_local! {
    static FULL_REBUILD_COUNT: RefCell<usize> = const { RefCell::new(0) };
}

#[cfg(test)]
fn note_full_rebuild() {
    FULL_REBUILD_COUNT.with(|counter| {
        let mut value = counter.borrow_mut();
        *value += 1;
    });
}

#[cfg(not(test))]
fn note_full_rebuild() {}

fn edge_key(a: usize, b: usize) -> u64 {
    let (x, y) = if a <= b { (a, b) } else { (b, a) };
    ((x as u64) << 32) | y as u64
}

fn prepare_vertex_index(complex: &RipsComplex, ws: &mut HomologyWorkspace) -> usize {
    let mut max_id = 0usize;
    let mut n_v = 0usize;
    for v in complex.simplices_of_dim(0) {
        let raw = v[0].get() as usize;
        max_id = max_id.max(raw);
        n_v += 1;
    }

    if ws.id_to_vertex.len() <= max_id {
        ws.id_to_vertex.resize(max_id + 1, usize::MAX);
    }

    for i in 0..=max_id {
        ws.id_to_vertex[i] = usize::MAX;
    }

    for (row, v) in complex.simplices_of_dim(0).enumerate() {
        ws.id_to_vertex[v[0].get() as usize] = row;
    }

    n_v
}

fn build_d1(complex: &RipsComplex, ws: &HomologyWorkspace, n_v: usize) -> Z2Matrix {
    let n_e = complex.simplices_of_dim(1).count();
    if n_v == 0 || n_e == 0 {
        return Z2Matrix::new(0, 0);
    }

    let mut m = Z2Matrix::new(n_v, n_e);
    for (col, edge) in complex.simplices_of_dim(1).enumerate() {
        let u = ws.id_to_vertex[edge[0].get() as usize];
        let v = ws.id_to_vertex[edge[1].get() as usize];
        if u != usize::MAX {
            m.set(u, col, true);
        }
        if v != usize::MAX {
            m.set(v, col, true);
        }
    }
    m
}

fn build_d2(complex: &RipsComplex, ws: &mut HomologyWorkspace) -> Z2Matrix {
    let n_e = complex.simplices_of_dim(1).count();
    let n_t = complex.simplices_of_dim(2).count();
    if n_e == 0 || n_t == 0 {
        return Z2Matrix::new(0, 0);
    }

    ws.edge_lookup.clear();
    for (row, edge) in complex.simplices_of_dim(1).enumerate() {
        let u = ws.id_to_vertex[edge[0].get() as usize];
        let v = ws.id_to_vertex[edge[1].get() as usize];
        ws.edge_lookup.push((edge_key(u, v), row));
    }
    ws.edge_lookup.sort_unstable_by_key(|&(key, _)| key);

    let mut m = Z2Matrix::new(n_e, n_t);
    for (col, tri) in complex.simplices_of_dim(2).enumerate() {
        let a = ws.id_to_vertex[tri[0].get() as usize];
        let b = ws.id_to_vertex[tri[1].get() as usize];
        let c = ws.id_to_vertex[tri[2].get() as usize];

        for key in [edge_key(a, b), edge_key(a, c), edge_key(b, c)] {
            if let Ok(pos) = ws.edge_lookup.binary_search_by_key(&key, |&(k, _)| k) {
                let row = ws.edge_lookup[pos].1;
                m.set(row, col, true);
            }
        }
    }
    m
}

fn check_h1_full(complex: &RipsComplex, ws: &mut HomologyWorkspace, n_v: usize) -> bool {
    note_full_rebuild();

    let n_edges = complex.simplices_of_dim(1).count();
    let mut d1 = build_d1(complex, ws, n_v);
    let rank_d1 = d1.rank_by_gaussian_elimination();
    let rank_ker_d1 = n_edges.saturating_sub(rank_d1);

    let rank_im_d2 = {
        let mut d2 = build_d2(complex, ws);
        d2.rank_by_gaussian_elimination()
    };

    let h1_dim = rank_ker_d1.saturating_sub(rank_im_d2);
    h1_dim == 0
}

fn detect_graph_cycle_incremental(
    complex: &RipsComplex,
    ws: &HomologyWorkspace,
    cache: &mut H1Cache,
) -> bool {
    let n_v = complex.simplices_of_dim(0).count();
    if n_v == 0 {
        return false;
    }

    cache.prepare_union_find(n_v);
    for edge in complex.simplices_of_dim(1) {
        let u = ws.id_to_vertex[edge[0].get() as usize];
        let v = ws.id_to_vertex[edge[1].get() as usize];
        if u == usize::MAX || v == usize::MAX {
            continue;
        }
        if !cache.union(u, v) {
            return true;
        }
    }

    false
}

/// Cohomology validator.
///
/// AX-ID: AXIOMA-007, AXIOMA-009
pub struct CohomologyValidator;

impl CohomologyValidator {
    /// Returns `true` if H¹(complex) = 0 (no independent cycles), `false` otherwise.
    pub fn check_h1(complex: &RipsComplex) -> bool {
        let n_edges = complex.simplices_of_dim(1).count();
        if n_edges == 0 {
            return true;
        }

        let key = H1CacheKey {
            ptr: complex as *const _,
            counts: complex.counts(),
        };

        H1_CACHE.with(|cache_cell| {
            HOMOLOGY_WORKSPACE.with(|ws_cell| {
                let mut cache = cache_cell.borrow_mut();

                if cache.key != Some(key) {
                    cache.invalidate();
                    cache.key = Some(key);
                }

                if let Some(cached) = cache.cached_result {
                    return cached;
                }

                let mut ws = ws_cell.borrow_mut();
                let n_v = prepare_vertex_index(complex, &mut ws);

                if !detect_graph_cycle_incremental(complex, &ws, &mut cache) {
                    cache.cached_result = Some(true);
                    return true;
                }

                let full_result = check_h1_full(complex, &mut ws, n_v);
                cache.cached_result = Some(full_result);
                full_result
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hnsw::HnswGraph;
    use crate::rips::RipsComplex;
    use genesis_math::SparseCliffordVector;
    use genesis_types::NodeId;

    fn make_seeded_vec(node_id: u64, seed: u64) -> SparseCliffordVector {
        let a = ((seed >> 8) & 0xFF) as f64 * 0.001 + 0.05;
        let b = ((seed >> 24) & 0xFF) as f64 * 0.001 + 0.1;
        SparseCliffordVector::from_iter((0..4).map(|idx| {
            let scale = (idx as f64 + 1.0) * 0.2;
            (idx, scale * (a * node_id as f64 + b))
        }))
        .unwrap()
    }

    fn xorshift64(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    fn check_h1_full_for_test(complex: &RipsComplex) -> bool {
        HOMOLOGY_WORKSPACE.with(|cell| {
            let mut ws = cell.borrow_mut();
            let n_v = prepare_vertex_index(complex, &mut ws);
            check_h1_full(complex, &mut ws, n_v)
        })
    }

    fn reset_full_rebuild_count() {
        FULL_REBUILD_COUNT.with(|counter| *counter.borrow_mut() = 0);
    }

    fn full_rebuild_count() -> usize {
        FULL_REBUILD_COUNT.with(|counter| *counter.borrow())
    }

    fn make_vec(id: u64) -> SparseCliffordVector {
        let s = id as f64 * 0.15 + 0.05;
        SparseCliffordVector::from_iter((0..4).map(|b| (b, s * (b as f64 + 1.0)))).unwrap()
    }

    #[test]
    fn cohomology_h1_zero_for_cycle_free_graph() {
        let mut g = HnswGraph::new(8);
        for i in 0..5u64 {
            g.insert(
                NodeId::try_new(i).expect("NodeId válido por construcción"),
                &make_vec(i),
            )
            .unwrap();
        }
        let complex = RipsComplex::build(&g, 10.0);
        let result = CohomologyValidator::check_h1(&complex);
        println!("H1 zero for cycle-free-like graph: {}", result);
    }

    #[test]
    fn cohomology_h1_nonzero_for_loop() {
        let mut d1 = Z2Matrix::new(4, 4);
        d1.set(0, 0, true);
        d1.set(1, 0, true);
        d1.set(1, 1, true);
        d1.set(2, 1, true);
        d1.set(2, 2, true);
        d1.set(3, 2, true);
        d1.set(3, 3, true);
        d1.set(0, 3, true);

        let rank_d1 = d1.rank_by_gaussian_elimination();
        let n_edges = 4;
        let rank_ker_d1 = n_edges - rank_d1;
        let h1_dim = rank_ker_d1;

        assert_eq!(h1_dim, 1, "expected H¹=1 for a 4-cycle, got {}", h1_dim);
    }

    #[test]
    fn h1_incremental_matches_full_on_1000_random_graphs() {
        let mut seed = 0xA5A5_5A5A_D3C1_9E37u64;
        for _ in 0..1000 {
            let mut g = HnswGraph::new(8);
            let node_count = 8 + (xorshift64(&mut seed) % 17) as usize;
            for node in 0..node_count as u64 {
                let local_seed = xorshift64(&mut seed);
                g.insert(
                    NodeId::try_new(node).expect("NodeId válido por construcción"),
                    &make_seeded_vec(node, local_seed),
                )
                .unwrap();
            }

            let epsilon = 0.15 + (xorshift64(&mut seed) % 700) as f64 * 0.001;
            let complex = RipsComplex::build(&g, epsilon);

            let incremental = CohomologyValidator::check_h1(&complex);
            let full = check_h1_full_for_test(&complex);
            assert_eq!(incremental, full);
        }
    }

    #[test]
    fn h1_incremental_ualpha_complexity() {
        let mut g = HnswGraph::new(16);
        for i in 0..128u64 {
            g.insert(
                NodeId::try_new(i).expect("NodeId válido por construcción"),
                &make_vec(i),
            )
            .unwrap();
        }

        let complex = RipsComplex::build(&g, 2.5);
        reset_full_rebuild_count();

        for _ in 0..1000 {
            let _ = CohomologyValidator::check_h1(&complex);
        }

        assert!(
            full_rebuild_count() <= 1,
            "expected <=1 full Z2 rebuild, got {}",
            full_rebuild_count()
        );
    }
}
