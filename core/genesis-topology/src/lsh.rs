use genesis_math::SparseCliffordVector;
/// AX-ID: AXIOMA-013
/// Locality Sensitive Hashing for G(1,3) vectors.
/// Projections are `SparseCliffordVectors`; inner product via `metric_scalar_product`.
/// Buckets: Vec<Vec<NodeId>> indexed by sorted (u32, Vec<NodeId>) pairs.
/// No `HashMap`. O(log N) amortized.
use genesis_types::NodeId;
use std::sync::OnceLock;

/// Number of projection hyperplanes per table.
const N_PROJECTIONS: usize = 8;

/// Number of independent hash tables.
const N_TABLES: usize = 4;

/// Total projection vectors needed.
const TOTAL_PROJECTIONS: usize = N_TABLES * N_PROJECTIONS;

/// Exact reciprocal of `2^53` for PRNG-to-`f64` conversion.
const U53_RECIP: f64 = 1.0 / (1u64 << 53) as f64;

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
            let sign = seed_sign(seed);
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            // seed >> 11 produce value of 53 bits. IEEE 754 f64 mantissa = 52 bits implicit +
            // 1 implicit = 53 bits exacts. (seed >> 11) as f64 is lossless by design
            // (technique standard of conversion PRNG→[0,1): Vigna 2015, §2).
            // 1u64 << 53 = 2^53, exactly representable as f64.
            #[allow(clippy::cast_precision_loss)]
            let mag = ((seed >> 11) as f64) * U53_RECIP;
            // loop-invariant, hoisted
            // CRYSTAL: O3, O4 — inevitable
            out[p][blade] = sign * (mag * 0.9 + 0.1);
            b += 1;
        }
        p += 1;
    }
    out
};

#[inline]
const fn seed_sign(seed: u64) -> f64 {
    1.0 - 2.0 * ((seed >> 63) as f64)
}

/// One LSH table backed by parallel sorted arrays.
///
/// `bucket_ids` is the binary-search key array; `bucket_nodes` stores payloads
/// at matching indices.
struct LshTable {
    bucket_ids: Vec<u32>,
    bucket_nodes: Vec<Vec<NodeId>>,
}

impl LshTable {
    const fn new() -> Self {
        Self {
            bucket_ids: Vec::new(),
            bucket_nodes: Vec::new(),
        }
    }

    fn pos(&self, bucket: u32) -> Result<usize, usize> {
        self.bucket_ids.binary_search(&bucket)
    }

    fn insert(&mut self, bucket: u32, id: NodeId) {
        match self.pos(bucket) {
            Ok(i) => {
                // Bucket already exists: insert `id` while preserving order by get().
                // binary_search: O(log bucket_size) instead of O(bucket_size).
                // Bucket invariant: Vec<NodeId> is always sorted by id.get().
                let bucket_vec = &mut self.bucket_nodes[i];
                match bucket_vec.binary_search_by_key(&id.get(), |n| n.get()) {
                    Ok(_) => {} // already present — avoid duplicates
                    Err(pos) => bucket_vec.insert(pos, id),
                }
            }
            Err(i) => {
                // New bucket: insert at the sorted bucket position.
                self.bucket_ids.insert(i, bucket);
                self.bucket_nodes.insert(i, vec![id]);
            }
        }

        debug_assert!(self.bucket_ids.len() <= (1usize << N_PROJECTIONS));
    }

    fn get(&self, bucket: u32) -> &[NodeId] {
        match self.pos(bucket) {
            Ok(i) => &self.bucket_nodes[i],
            Err(_) => &[],
        }
    }
}

/// `CliffordHashTable` — LSH index for G(1,3) vectors.
///
/// Uses `N_TABLES` independent random-projection hash tables.
/// Each table maps a 16-blade vector to a N_PROJECTIONS-bit signature.
/// Insert: `O(N_TABLES * log bucket_size)`.
/// Candidates: `O(sum(bucket_sizes) * log N_TABLES)` with k-way merge + deduplication.
///
/// # Projection cache (BN-04)
/// Projection coefficients are generated at compile time (`PROJ_COEFFS`) and
/// packed once in `new()` into a fixed 6-lane bivector representation.
/// Hashing then uses a branchless fixed-width dot kernel with no allocation.
///
/// AX-ID: AXIOMA-013
pub struct CliffordHashTable {
    tables: [LshTable; N_TABLES],
}

/// Process-wide immutable cache for packed bivector projection coefficients.
///
/// HOT PATH: projection hashing reads from this array for every insertion and query.
/// The coefficients are initialized exactly once to avoid repeated packing work.
///
/// AX-ID: AXIOMA-013
static PACKED_BIVECTOR_PROJECTIONS: OnceLock<[[f64; 6]; TOTAL_PROJECTIONS]> = OnceLock::new();

#[inline]
fn packed_bivector_projections() -> &'static [[f64; 6]; TOTAL_PROJECTIONS] {
    PACKED_BIVECTOR_PROJECTIONS.get_or_init(|| {
        core::array::from_fn(|i| {
            [
                PROJ_COEFFS[i][BIVECTOR_BLADES[0]],
                PROJ_COEFFS[i][BIVECTOR_BLADES[1]],
                PROJ_COEFFS[i][BIVECTOR_BLADES[2]],
                PROJ_COEFFS[i][BIVECTOR_BLADES[3]],
                PROJ_COEFFS[i][BIVECTOR_BLADES[4]],
                PROJ_COEFFS[i][BIVECTOR_BLADES[5]],
            ]
        })
    })
}

/// Pack a multivector into a fixed 6-lane bivector array.
///
/// Lane order matches `BIVECTOR_BLADES` exactly.
///
/// AX-ID: AXIOMA-013
#[inline]
const fn pack_bivector_coeffs(vec: &SparseCliffordVector) -> [f64; 6] {
    [
        vec.coeffs[BIVECTOR_BLADES[0]],
        vec.coeffs[BIVECTOR_BLADES[1]],
        vec.coeffs[BIVECTOR_BLADES[2]],
        vec.coeffs[BIVECTOR_BLADES[3]],
        vec.coeffs[BIVECTOR_BLADES[4]],
        vec.coeffs[BIVECTOR_BLADES[5]],
    ]
}

/// Fixed-width bivector projection kernel.
///
/// AX-ID: AXIOMA-013
#[inline]
fn dot_bivector_lanes(lhs: &[f64; 6], rhs: &[f64; 6]) -> f64 {
    lhs[0].mul_add(rhs[0], lhs[1].mul_add(rhs[1], lhs[2].mul_add(rhs[2], 0.0)))
        + lhs[3].mul_add(rhs[3], lhs[4].mul_add(rhs[4], lhs[5] * rhs[5]))
}

impl CliffordHashTable {
    /// Create a new empty hash table.
    pub fn new() -> Self {
        let _ = packed_bivector_projections();
        Self {
            tables: [
                LshTable::new(),
                LshTable::new(),
                LshTable::new(),
                LshTable::new(),
            ],
        }
    }

    /// Compute the N_PROJECTIONS-bit bucket hash for table `t` from packed
    /// bivector lanes.
    ///
    /// AX-ID: AXIOMA-013
    #[inline]
    #[allow(clippy::unused_self)]
    fn hash_packed_bivector(&self, packed_bivector: &[f64; 6], t: usize) -> u32 {
        let start = t * N_PROJECTIONS;
        let mut bits: u32 = 0;
        let coeffs = packed_bivector_projections();
        for p in 0..N_PROJECTIONS {
            let dot = dot_bivector_lanes(packed_bivector, &coeffs[start + p]);
            bits |= ((dot >= 0.0) as u32) << p;
        }
        bits
    }

    /// Insert a node into the index.
    ///
    /// AX-ID: AXIOMA-013
    pub fn insert(&mut self, id: NodeId, vec: &SparseCliffordVector) {
        let packed = pack_bivector_coeffs(vec);
        for t in 0..N_TABLES {
            let bucket = self.hash_packed_bivector(&packed, t);
            self.tables[t].insert(bucket, id);
        }
    }

    /// Return candidate `NodeIds` for a query (union of all matching buckets).
    ///
    /// Complexity: `O(sum(bucket_sizes) * log N_TABLES)`.
    /// Buckets are already sorted by `NodeId::get()`, and are merged with a
    /// k-way merge to remove duplicates without linear `contains` scans.
    ///
    /// AX-ID: AXIOMA-013
    pub fn candidates<'a>(
        &'a self,
        query: &SparseCliffordVector,
    ) -> impl Iterator<Item = NodeId> + 'a {
        let packed = pack_bivector_coeffs(query);
        let bucket_slices: [&[NodeId]; N_TABLES] =
            core::array::from_fn(|t| self.tables[t].get(self.hash_packed_bivector(&packed, t)));
        let mut cursors = [0usize; N_TABLES];
        let mut last_emitted: Option<NodeId> = None;
        core::iter::from_fn(move || loop {
            let mut best: Option<(u64, usize, NodeId)> = None;
            for table_idx in 0..N_TABLES {
                let cursor = cursors[table_idx];
                let Some(&id) = bucket_slices[table_idx].get(cursor) else {
                    continue;
                };
                let candidate = (id.get(), table_idx, id);
                if best.as_ref().is_none_or(|current| candidate < *current) {
                    best = Some(candidate);
                }
            }
            let (_, table_idx, id) = best?;
            cursors[table_idx] += 1;
            if last_emitted == Some(id) {
                continue;
            }
            last_emitted = Some(id);
            return Some(id);
        })
    }
}

impl Default for CliffordHashTable {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use genesis_math::SparseCliffordVector;

    use super::*;
    use crate::geodesic::geometric_distance;

    fn make_vec(pairs: &[(usize, f64)]) -> SparseCliffordVector {
        SparseCliffordVector::from_iter(pairs.iter().copied()).unwrap()
    }

    #[test]
    fn lsh_insert_and_candidates() {
        let mut table = CliffordHashTable::new();
        for i in 0..50u64 {
            let v = make_vec(
                &(0..4)
                    .map(|b| (b, (i as f64).mul_add(0.02, b as f64 * 0.1)))
                    .collect::<Vec<_>>(),
            );
            table.insert(
                NodeId::try_new(i).expect("NodeId válido por construcción"),
                &v,
            );
        }
        let query = make_vec(&[(0, 0.5), (1, 0.2), (2, 0.4)]);
        println!("LSH candidates: {}", table.candidates(&query).count());
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
        // Insert the same vector 10 times → it must appear only 1 time in the results.
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
                let sign = seed_sign(seed);
                seed = seed
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                #[allow(clippy::cast_precision_loss)]
                let mag = ((seed >> 11) as f64) * U53_RECIP;
                // loop-invariant, hoisted
                // CRYSTAL: O57, O58 — inevitable
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
        let start = t * N_PROJECTIONS;
        // loop-invariant, hoisted
        // CRYSTAL: FO50 — inevitable
        for p in 0..N_PROJECTIONS {
            let proj_idx = start + p;
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
            exact.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
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
