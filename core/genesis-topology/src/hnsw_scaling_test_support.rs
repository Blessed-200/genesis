use genesis_math::SparseCliffordVector;

#[inline]
pub(super) fn make_scaling_vec(seed: u64) -> SparseCliffordVector {
    let mut coeffs = [0.0f64; 16];
    let mut rng = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    for c in &mut coeffs {
        rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let sample = (rng >> 32) as u32;
        *c = (sample as f64 / u32::MAX as f64).mul_add(2.0, -1.0);
    }
    SparseCliffordVector::from_dense(&coeffs)
        .unwrap_or_else(|_| panic!("make_scaling_vec failed for seed={seed}"))
}

#[inline]
pub(super) fn make_neighbor_vec(coeff: f64) -> SparseCliffordVector {
    SparseCliffordVector::from_iter((0..4).map(|i| (i, coeff * (i as f64 + 1.0) * 0.1)))
        .unwrap_or_else(|_| panic!("make_neighbor_vec failed for coeff={coeff}"))
}
