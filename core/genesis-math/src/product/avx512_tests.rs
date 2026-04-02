use super::{geometric_product_scalar_dense, geometric_product_x86_avx512_dense, TOTAL_BLADES};

#[inline]
fn next_unit(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    ((*state >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
}

#[test]
fn avx512_dense_kernel_matches_scalar_for_one_hundred_thousand_cases() {
    if !std::arch::is_x86_feature_detected!("avx512f") {
        return;
    }

    const CASES: usize = 100_000;

    let mut state = 0x9B07_1D2A_F4E1_5C33u64;
    for _ in 0..CASES {
        let mut a = [0.0f64; TOTAL_BLADES];
        let mut b = [0.0f64; TOTAL_BLADES];

        for i in 0..TOTAL_BLADES {
            a[i] = next_unit(&mut state).mul_add(2.0, -1.0);
            b[i] = next_unit(&mut state).mul_add(2.0, -1.0);
        }

        let mut scalar = [0.0f64; TOTAL_BLADES];
        let mut avx512 = [0.0f64; TOTAL_BLADES];
        geometric_product_scalar_dense(&a, &b, &mut scalar);
        // SAFETY: This test runs only when AVX-512F is available at runtime.
        unsafe {
            geometric_product_x86_avx512_dense(&a, &b, &mut avx512);
        }

        for k in 0..TOTAL_BLADES {
            assert_eq!(
                scalar[k].to_bits(),
                avx512[k].to_bits(),
                "AVX-512 dense mismatch at blade {k}: scalar={} avx512={}",
                scalar[k],
                avx512[k]
            );
        }
    }
}
