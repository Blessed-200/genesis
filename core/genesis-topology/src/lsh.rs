use genesis_math::SparseCliffordVector;
/// AX-ID: AXIOMA-013
/// Locality Sensitive Hashing for G(1,3) vectors.
/// Projections are `SparseCliffordVectors`; inner product via `metric_scalar_product`.
/// Buckets: Vec<Vec<NodeId>> indexed by sorted (u32, Vec<NodeId>) pairs.
/// No `HashMap`. O(log N) amortized.
use genesis_types::NodeId;
use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

/// Number of projection hyperplanes per table.
const N_PROJECTIONS: usize = 8;

/// Number of independent hash tables.
const N_TABLES: usize = 4;

/// Total projection vectors needed.
const TOTAL_PROJECTIONS: usize = N_TABLES * N_PROJECTIONS;

/// Canonical grade-2 blade indices in G(1,3): e01, e02, e12, e03, e13, e23.
const BIVECTOR_BLADES: [usize; 6] = [0b0011, 0b0101, 0b0110, 0b1001, 0b1010, 0b1100];

/// Projection coefficients [`TOTAL_PROJECTIONS`][16] generated at compile time
/// with a fixed LCG seed → stored in .rodata.
///
/// LSH is explicitly restricted to the grade-2 (bivectorial) subspace:
/// only the 6 bivector blades receive non-zero coefficients.
///
/// AX-ID: AXIOMA-013
const PROJ_COEFFS: [[f64; 16]; TOTAL_PROJECTIONS] = {
    let mut out = [[0.0f64; 16]; TOTAL_PROJECTIONS];
    let mut seed: u64 = 0xDEAD_BEEF_CAFE_BABE_u64;
    let mut p = 0;
    while p < TOTAL_PROJECTIONS {
        let mut b = 0;
        while b < BIVECTOR_BLADES.len() {
            let blade = BIVECTOR_BLADES[b];
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let sign = if (seed >> 63) == 0 { 1.0 } else { -1.0 };
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            // seed >> 11 produce valor de 53 bits. IEEE 754 f64 mantissa = 52 bits implícito +
            // 1 implícito = 53 bits exactos. (seed >> 11) as f64 es sin pérdida por diseño
            // (técnica estándar de conversión PRNG→[0,1): Vigna 2015, §2).
            // 1u64 << 53 = 2^53, exactamente representable como f64.
            #[allow(clippy::cast_precision_loss)]
            let mag = ((seed >> 11) as f64) * (1.0 / (1u64 << 53) as f64);
            out[p][blade] = sign * (mag * 0.9 + 0.1);
            b += 1;
        }
        p += 1;
    }
    out
};

/// One LSH table: sorted (`bucket_id`, Vec<NodeId>) pairs, binary searched.
struct LshTable {
    buckets: Vec<(u32, Vec<NodeId>)>,
}

impl LshTable {
    fn new() -> Self {
        Self {
            buckets: Vec::new(),
        }
    }

    fn pos(&self, bucket: u32) -> Result<usize, usize> {
        self.buckets.binary_search_by_key(&bucket, |&(b, _)| b)
    }

    fn insert(&mut self, bucket: u32, id: NodeId) {
        match self.pos(bucket) {
            Ok(i) => {
                // Bucket ya existe: insertar `id` manteniendo orden por get().
                // binary_search: O(log bucket_size) en lugar de O(bucket_size).
                // Invariante del bucket: Vec<NodeId> siempre ordenado por id.get().
                let bucket_vec = &mut self.buckets[i].1;
                match bucket_vec.binary_search_by_key(&id.get(), |n| n.get()) {
                    Ok(_) => {} // ya presente — no duplicar
                    Err(pos) => bucket_vec.insert(pos, id),
                }
            }
            Err(i) => {
                // Bucket nuevo: insertar en posición ordenada del Vec de buckets.
                self.buckets.insert(i, (bucket, vec![id]));
            }
        }
    }

    fn get(&self, bucket: u32) -> &[NodeId] {
        match self.pos(bucket) {
            Ok(i) => &self.buckets[i].1,
            Err(_) => &[],
        }
    }
}

/// `CliffordHashTable` — LSH index for G(1,3) vectors.
///
/// Uses `N_TABLES` independent random-projection hash tables.
/// Each table maps a 16-blade vector to a N_PROJECTIONS-bit signature.
/// Insert: `O(N_TABLES * log bucket_size)`.
/// Candidates: `O(sum(bucket_sizes) * log N_TABLES)` with k-way merge + deduplicación.
///
/// # Projection vector cache (BN-04)
/// The 32 `SparseCliffordVector` projections derived from `PROJ_COEFFS` are
/// computed **once** in `new()` and stored in `proj_vecs`. This eliminates
/// 32 × `from_dense()` calls (including finite checks + metadata computation)
/// on every `insert()` — reducing ~6400 CPU instructions per insert to ~32
/// scalar product evaluations.
///
/// AX-ID: AXIOMA-013
pub struct CliffordHashTable {
    tables: [LshTable; N_TABLES],
    /// Precomputed projection vectors. Computed once in `new()`, reused on every hash.
    /// `proj_vecs[t * N_PROJECTIONS + p]` is the p-th projection vector for table t.
    proj_vecs: [SparseCliffordVector; TOTAL_PROJECTIONS],
}

impl CliffordHashTable {
    /// Create a new empty hash table with precomputed projection vectors.
    ///
    /// The `PROJ_COEFFS` compile-time constants are converted to
    /// `SparseCliffordVector` instances once here, eliminating per-insert recomputation.
    pub fn new() -> Self {
        let proj_vecs = core::array::from_fn(|i| {
            SparseCliffordVector::from_dense(&PROJ_COEFFS[i])
                .expect("BN-04: PROJ_COEFFS must produce valid SparseCliffordVectors")
        });
        Self {
            tables: [
                LshTable::new(),
                LshTable::new(),
                LshTable::new(),
                LshTable::new(),
            ],
            proj_vecs,
        }
    }

    /// Compute the N_PROJECTIONS-bit bucket hash for table `t`.
    /// Uses precomputed `proj_vecs` — zero allocation, 8 scalar products.
    #[inline]
    fn hash_vector_local(&self, vec: &SparseCliffordVector, t: usize) -> u32 {
        let start = t * N_PROJECTIONS;
        let mut bits: u32 = 0;
        for p in 0..N_PROJECTIONS {
            if vec.metric_scalar_product(&self.proj_vecs[start + p]) >= 0.0 {
                bits |= 1 << p;
            }
        }
        bits
    }

    /// Insert a node into the index.
    ///
    /// AX-ID: AXIOMA-013
    pub fn insert(&mut self, id: NodeId, vec: &SparseCliffordVector) {
        for t in 0..N_TABLES {
            let bucket = self.hash_vector_local(vec, t);
            self.tables[t].insert(bucket, id);
        }
    }

    /// Return candidate `NodeIds` for a query (union of all matching buckets).
    ///
    /// Complejidad: `O(sum(bucket_sizes) * log N_TABLES)`.
    /// Los buckets ya están ordenados por `NodeId::get()`, y se fusionan con k-way merge
    /// (min-heap) para eliminar duplicados sin `contains` lineal.
    ///
    /// AX-ID: AXIOMA-013
    pub fn candidates<'a>(
        &'a self,
        query: &SparseCliffordVector,
    ) -> impl Iterator<Item = NodeId> + 'a {
        #[derive(Copy, Clone, Eq, PartialEq)]
        struct MergeItem {
            id_key: u64,
            table_idx: usize,
            elem_idx: usize,
            id: NodeId,
        }

        impl Ord for MergeItem {
            fn cmp(&self, other: &Self) -> Ordering {
                self.id_key
                    .cmp(&other.id_key)
                    .then_with(|| self.table_idx.cmp(&other.table_idx))
                    .then_with(|| self.elem_idx.cmp(&other.elem_idx))
            }
        }

        impl PartialOrd for MergeItem {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }

        let bucket_slices: [&[NodeId]; N_TABLES] = core::array::from_fn(|t| {
            let bucket = self.hash_vector_local(query, t);
            self.tables[t].get(bucket)
        });

        let mut heap: BinaryHeap<Reverse<MergeItem>> = BinaryHeap::new();
        for (table_idx, bucket) in bucket_slices.iter().enumerate() {
            if let Some(&id) = bucket.first() {
                heap.push(Reverse(MergeItem {
                    id_key: id.get(),
                    table_idx,
                    elem_idx: 0,
                    id,
                }));
            }
        }

        let mut result: Vec<NodeId> = Vec::new();
        let mut last_emitted: Option<NodeId> = None;
        while let Some(Reverse(item)) = heap.pop() {
            if last_emitted != Some(item.id) {
                result.push(item.id);
                last_emitted = Some(item.id);
            }

            let next_idx = item.elem_idx + 1;
            if let Some(&next_id) = bucket_slices[item.table_idx].get(next_idx) {
                heap.push(Reverse(MergeItem {
                    id_key: next_id.get(),
                    table_idx: item.table_idx,
                    elem_idx: next_idx,
                    id: next_id,
                }));
            }
        }

        result.into_iter()
    }
}

impl Default for CliffordHashTable {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geodesic::geometric_distance;
    use genesis_math::SparseCliffordVector;

    fn make_vec(pairs: &[(usize, f64)]) -> SparseCliffordVector {
        SparseCliffordVector::from_iter(pairs.iter().copied()).unwrap()
    }

    #[test]
    fn lsh_insert_and_candidates() {
        let mut table = CliffordHashTable::new();
        for i in 0..50u64 {
            let v = make_vec(
                &(0..4)
                    .map(|b| (b, i as f64 * 0.02 + b as f64 * 0.1))
                    .collect::<Vec<_>>(),
            );
            table.insert(
                NodeId::try_new(i).expect("NodeId válido por construcción"),
                &v,
            );
        }
        let query = make_vec(&[(0, 0.5), (1, 0.2), (2, 0.4)]);
        let candidates: Vec<_> = table.candidates(&query).collect();
        println!("LSH candidates: {}", candidates.len());
    }

    #[test]
    fn lsh_same_vector_is_candidate_for_itself() {
        let mut table = CliffordHashTable::new();
        let v = make_vec(&[(0, 1.0), (1, -0.5), (2, 0.3), (3, 0.7)]);
        let id = NodeId::try_new(42).expect("NodeId válido por construcción");
        table.insert(id, &v);
        let candidates: Vec<_> = table.candidates(&v).collect();
        assert!(
            candidates.contains(&id),
            "self must be a candidate, got {:?}",
            candidates
        );
    }

    #[test]
    fn lsh_no_duplicates_in_candidates() {
        let mut table = CliffordHashTable::new();
        let v = make_vec(&[(0, 0.5), (1, 0.5), (2, 0.5), (3, 0.5)]);
        let id = NodeId::try_new(1).expect("NodeId válido por construcción");
        table.insert(id, &v);
        table.insert(id, &v);
        let candidates: Vec<_> = table.candidates(&v).collect();
        let count = candidates.iter().filter(|&&x| x == id).count();
        assert_eq!(count, 1, "expected exactly 1 occurrence");
    }

    #[test]
    fn lsh_no_duplicates_after_repeated_insert() {
        let mut table = CliffordHashTable::new();
        let id = genesis_types::NodeId::try_new(42).expect("NodeId válido por construcción");
        let v = genesis_math::SparseCliffordVector::from_iter(
            (0..4).map(|i| (i, (i as f64 + 1.0) * 0.1)),
        )
        .unwrap();
        // Insertar el mismo vector 10 veces → debe aparecer solo 1 vez en los resultados.
        for _ in 0..10 {
            table.insert(id, &v);
        }
        let candidates: Vec<_> = table.candidates(&v).collect();
        let count = candidates.iter().filter(|&&c| c == id).count();
        assert_eq!(
            count, 1,
            "NodeId debe aparecer exactamente una vez, apareció {}",
            count
        );
    }

    fn generic_proj_coeffs() -> [[f64; 16]; TOTAL_PROJECTIONS] {
        let mut out = [[0.0f64; 16]; TOTAL_PROJECTIONS];
        let mut seed: u64 = 0xDEAD_BEEF_CAFE_BABE_u64;
        let mut p = 0;
        while p < TOTAL_PROJECTIONS {
            let mut b = 0;
            while b < 16 {
                seed = seed
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let sign = if (seed >> 63) == 0 { 1.0 } else { -1.0 };
                seed = seed
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                #[allow(clippy::cast_precision_loss)]
                let mag = ((seed >> 11) as f64) * (1.0 / (1u64 << 53) as f64);
                out[p][b] = sign * (mag * 0.9 + 0.1);
                b += 1;
            }
            p += 1;
        }
        out
    }

    fn generic_project(
        vec: &SparseCliffordVector,
        p: usize,
        coeffs: &[[f64; 16]; TOTAL_PROJECTIONS],
    ) -> f64 {
        let proj = SparseCliffordVector::from_dense(&coeffs[p])
            .unwrap_or_else(|_| SparseCliffordVector::zero());
        vec.metric_scalar_product(&proj)
    }

    fn generic_hash(
        vec: &SparseCliffordVector,
        t: usize,
        coeffs: &[[f64; 16]; TOTAL_PROJECTIONS],
    ) -> u32 {
        let mut bits: u32 = 0;
        for p in 0..N_PROJECTIONS {
            let proj_idx = t * N_PROJECTIONS + p;
            if generic_project(vec, proj_idx, coeffs) >= 0.0 {
                bits |= 1 << p;
            }
        }
        bits
    }

    #[test]
    fn lsh_bivectorial_recall_vs_generic() {
        let generic_coeffs = generic_proj_coeffs();
        let mut table = CliffordHashTable::new();
        let mut nodes: Vec<(NodeId, SparseCliffordVector)> = Vec::new();

        for i in 0..512u64 {
            let ii = i as f64;
            let v = make_vec(&[
                (0b0011, (ii * 0.013).sin()),
                (0b0101, (ii * 0.017).cos()),
                (0b0110, (ii * 0.019).sin() * 0.75),
                (0b1001, (ii * 0.023).cos() * 0.65),
                (0b1010, (ii * 0.029).sin() * 0.55),
                (0b1100, (ii * 0.031).cos() * 0.45),
            ]);
            let id = NodeId::try_new(i).expect("NodeId válido por construcción");
            table.insert(id, &v);
            nodes.push((id, v));
        }

        let mut hits_bivector = 0usize;
        let mut hits_generic = 0usize;
        for q in 0..64u64 {
            let qq = q as f64;
            let query = make_vec(&[
                (0b0011, (qq * 0.041).sin()),
                (0b0101, (qq * 0.043).cos()),
                (0b0110, (qq * 0.047).sin() * 0.70),
                (0b1001, (qq * 0.053).cos() * 0.60),
                (0b1010, (qq * 0.059).sin() * 0.50),
                (0b1100, (qq * 0.061).cos() * 0.40),
            ]);

            let mut exact = nodes
                .iter()
                .map(|(id, v)| (*id, geometric_distance(&query, v)))
                .collect::<Vec<_>>();
            exact.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
            let top5 = exact.iter().take(5).map(|(id, _)| *id).collect::<Vec<_>>();

            let candidates_biv = table.candidates(&query).collect::<Vec<_>>();
            if top5.iter().any(|id| candidates_biv.contains(id)) {
                hits_bivector += 1;
            }

            let mut generic_seen: Vec<NodeId> = Vec::new();
            for t in 0..N_TABLES {
                let bucket = generic_hash(&query, t, &generic_coeffs);
                let mut bucket_ids = nodes
                    .iter()
                    .filter_map(|(id, v)| {
                        (generic_hash(v, t, &generic_coeffs) == bucket).then_some(*id)
                    })
                    .collect::<Vec<_>>();
                bucket_ids.sort_by_key(|id| id.get());
                for id in bucket_ids {
                    if generic_seen
                        .binary_search_by_key(&id.get(), |n| n.get())
                        .is_err()
                    {
                        let pos = generic_seen
                            .binary_search_by_key(&id.get(), |n| n.get())
                            .unwrap_or_else(|idx| idx);
                        generic_seen.insert(pos, id);
                    }
                }
            }
            if top5.iter().any(|id| generic_seen.contains(id)) {
                hits_generic += 1;
            }
        }

        assert!(
            hits_bivector >= hits_generic,
            "recall bivectorial ({hits_bivector}/64) must be >= generic ({hits_generic}/64)"
        );
    }
}
