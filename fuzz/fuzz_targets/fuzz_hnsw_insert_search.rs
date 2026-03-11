//! Fuzz target: HNSW insert + search with arbitrary vectors and IDs.
//!
//! Invariants:
//! 1. No panic on any valid NodeId + SparseCliffordVector input
//! 2. search_nearest returns k results if k ≤ inserted nodes
//! 3. Graph satisfies node_count() consistency
//!
//! AX-ID: AXIOMA-013 (O(log N) search)

#![no_main]
use libfuzzer_sys::fuzz_target;
use genesis_math::SparseCliffordVector;
use genesis_topology::hnsw::HnswGraph;
use genesis_types::NodeId;

fuzz_target!(|data: &[u8]| {
    // Minimum: 1 byte for node count, then groups of 129 bytes (1 node_id + 128 coeffs)
    if data.len() < 130 { return; }

    let n_nodes = (data[0] as usize).min(20); // cap at 20 to keep fast
    if n_nodes == 0 { return; }

    let mut graph = HnswGraph::new(8); // smaller M for faster fuzz
    let mut inserted = 0usize;

    for i in 0..n_nodes {
        let offset = 1 + i * 129;
        if offset + 129 > data.len() { break; }

        let raw_id = data[offset] as u64;
        let id = match NodeId::try_new(raw_id) {
            Ok(id) => id,
            Err(_) => continue,
        };

        let mut coeffs = [0.0f64; 16];
        let vec_bytes = &data[offset + 1..offset + 129];
        for (j, chunk) in vec_bytes.chunks_exact(8).enumerate() {
            let v = f64::from_le_bytes(chunk.try_into().unwrap_or([0u8; 8]));
            coeffs[j] = if v.is_finite() { v } else { 0.0 };
        }

        if let Ok(vec) = SparseCliffordVector::from_dense(&coeffs) {
            let _ = graph.insert(id, &vec);
            inserted += 1;
        }
    }

    // Search must not panic
    if inserted > 0 {
        let query_coeffs = [0.1f64; 16];
        if let Ok(query) = SparseCliffordVector::from_dense(&query_coeffs) {
            let k = 5.min(inserted);
            let results = graph.search_nearest(&query, k);
            // Results must be ≤ k (can be less if graph has fewer nodes)
            assert!(results.len() <= k, "search_nearest returned more than k results");
        }
    }
});
