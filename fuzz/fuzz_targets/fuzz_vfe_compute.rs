//! Fuzz target: VFEMinimizer::compute_vfe with arbitrary node IDs and observations.
//!
//! Invariants:
//! 1. compute_vfe never panics
//! 2. compute_vfe(id, None) ≥ 0.0 for registered nodes (Mahalanobis distance)
//! 3. Arbitrary NodeId never corrupts internal state
//!
//! AX-ID: AXIOMA-003, AXIOMA-008

#![no_main]
use libfuzzer_sys::fuzz_target;
use genesis_dynamics::free_energy::VFEMinimizer;
use genesis_types::NodeId;

fuzz_target!(|data: &[u8]| {
    if data.len() < 16 { return; }

    let mut vfe = VFEMinimizer::new();

    // Register up to 4 nodes with arbitrary IDs
    let n_nodes = (data[0] as usize % 4) + 1;
    for i in 0..n_nodes {
        let raw = data.get(i + 1).copied().unwrap_or(i as u8) as u64;
        if let Ok(id) = NodeId::try_new(raw) {
            let prior = [
                f64::from(data.get(8 + i * 4).copied().unwrap_or(0)) / 255.0,
                f64::from(data.get(9 + i * 4).copied().unwrap_or(0)) / 255.0,
                f64::from(data.get(10 + i * 4).copied().unwrap_or(0)) / 255.0,
                f64::from(data.get(11 + i * 4).copied().unwrap_or(0)) / 255.0,
            ];
            let _ = vfe.add_node(id, prior);
        }
    }

    // Query arbitrary NodeId — must not panic even for unregistered nodes
    let query_id_raw = u64::from_le_bytes(data[..8].try_into().unwrap_or([0u8; 8]));
    if let Ok(qid) = NodeId::try_new(query_id_raw.min(255)) {
        let result = vfe.compute_vfe(qid, None);
        // Must be finite (0.0 for unregistered nodes, ≥0 for registered)
        assert!(result.is_finite(), "compute_vfe must return finite value");
        assert!(result >= 0.0, "VFE must be non-negative (Mahalanobis distance squared)");
    }
});
