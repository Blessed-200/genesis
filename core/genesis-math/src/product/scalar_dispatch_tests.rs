use super::{geometric_product_dispatch_by_mask, geometric_product_scalar_sparse, TOTAL_BLADES};

#[inline]
fn next_unit(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    ((*state >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
}

#[test]
fn scalar_sparse_dispatch_routing_identity() {
    // NOTE: Current sparse dispatch path is scalar-only.
    // TODO(AVX-512 sparse): add geometric_product_x86_avx512_sparse and
    // reintroduce sparse dispatch-vs-kernel equivalence coverage.
    const SPARSE_MASKS: &[(u16, u16)] = &[
        (0x5555, 0xAAAA),
        (0x0F0F, 0x3333),
        (0x00FF, 0x0FF0),
        (0x1357, 0x2468),
    ];
    const CASES_PER_MASK: usize = 25_000;

    let mut state = 0xA76E_2F39_5D11_84C5u64;
    for &(mask_a, mask_b) in SPARSE_MASKS {
        assert!(mask_a.count_ones() < TOTAL_BLADES as u32);
        assert!(mask_b.count_ones() < TOTAL_BLADES as u32);

        for _ in 0..CASES_PER_MASK {
            let mut a = [0.0f64; TOTAL_BLADES];
            let mut b = [0.0f64; TOTAL_BLADES];

            for i in 0..TOTAL_BLADES {
                let x = next_unit(&mut state).mul_add(2.0, -1.0);
                if (mask_a & (1u16 << i)) != 0 {
                    a[i] = x;
                }

                let y = next_unit(&mut state).mul_add(2.0, -1.0);
                if (mask_b & (1u16 << i)) != 0 {
                    b[i] = y;
                }
            }

            let mut scalar = [0.0f64; TOTAL_BLADES];
            let mut dispatch = [0.0f64; TOTAL_BLADES];
            geometric_product_scalar_sparse(&a, mask_a, &b, mask_b, &mut scalar);
            geometric_product_dispatch_by_mask(&a, mask_a, &b, mask_b, &mut dispatch);

            for k in 0..TOTAL_BLADES {
                assert_eq!(
                    scalar[k].to_bits(),
                    dispatch[k].to_bits(),
                    "Sparse scalar routing mismatch at blade {k}: scalar={} dispatch={}",
                    scalar[k],
                    dispatch[k]
                );
            }
        }
    }
}
