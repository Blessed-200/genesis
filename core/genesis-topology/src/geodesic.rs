//! Distancias geométricas en G(1,3).
//!
//! La función principal es `geometric_distance`, que satisface los cuatro
//! axiomas métricos. `fast_bivector_distance` queda obsoleta.
//!
//! AX-ID: AXIOMA-013, AXIOMA-014, LEY_FUNDACIONAL §3.6
use genesis_math::{
    bivector_norm_sq_of_product, fast_metric_distance, BivectorProduct, SparseCliffordVector,
};

/// Distancia geométrica global en G(1,3).
///
/// Es la métrica principal del sistema: satisface simetría, positividad,
/// identidad y desigualdad triangular. Se utiliza en HNSW, cálculo de λ₂
/// y todas las operaciones topológicas.
///
/// AX-ID: AXIOMA-013, AXIOMA-014, LEY_FUNDACIONAL §3.6
#[allow(clippy::inline_always)]
#[inline(always)]
pub fn geometric_distance(a: &SparseCliffordVector, b: &SparseCliffordVector) -> f64 {
    fast_metric_distance(a, b)
}

/// Interacción bivectorial: norma de la parte de grado 2 del producto a*b.
///
/// NO es una distancia métrica (no cumple la desigualdad triangular).
/// Mide la "separación angular" en el subespacio de bivectores, útil para
/// detectar alineamiento geométrico entre conceptos en genesis-evolution.
///
/// AX-ID: AXIOMA-001 (estructura G(1,3))
pub fn bivector_interaction(a: &SparseCliffordVector, b: &SparseCliffordVector) -> f64 {
    match bivector_norm_sq_of_product(a, b) {
        BivectorProduct::Computed(v) => v.abs().sqrt(),
        BivectorProduct::SubPlanck => f64::MAX,
    }
}

/// Alias de compatibilidad — DEPRECADO: usar `geometric_distance()`.
///
/// `geometric_distance` es una métrica verdadera que satisface la
/// desigualdad triangular. Esta función delega en ella.
///
/// AX-ID: AXIOMA-013, AXIOMA-014
#[deprecated(note = "usar geometric_distance(), que es una métrica verdadera")]
#[allow(clippy::inline_always)]
#[inline(always)]
pub fn fast_bivector_distance(a: &SparseCliffordVector, b: &SparseCliffordVector) -> f64 {
    geometric_distance(a, b)
}

/// Distance from a decompressed dense array to a sparse vector.
///
/// Used exclusively by the `hnsw-f16` path, where layer-0 vectors are stored
/// as f16 and decompressed to `[f64; 16]` before distance evaluation.
///
/// Computes the same grade-weighted metric as `geometric_distance` / `fast_metric_distance`:
/// `d(a_dense, b) = √(Σᵢ METRIC_WEIGHTS[i] · (a_dense[i] − b.coeffs[i])²)`
///
/// This guarantees **metric consistency**: the f16 compression path produces
/// the same neighbourhood structure as the full-precision path. The former
/// `fast_bivector_distance_from_dense` computed the bivector product norm —
/// a different function, producing different neighbourhoods.
///
/// AX-ID: AXIOMA-013, AXIOMA-014
#[allow(clippy::inline_always)]
#[inline(always)]
pub fn fast_bivector_distance_from_dense(a_dense: &[f64; 16], b: &SparseCliffordVector) -> f64 {
    genesis_math::fast_metric_distance_from_dense(a_dense, b)
}

#[cfg(test)]
mod tests {
    use genesis_math::SparseCliffordVector;

    use super::*;

    fn make_vec(pairs: &[(usize, f64)]) -> SparseCliffordVector {
        SparseCliffordVector::from_iter(pairs.iter().copied()).unwrap()
    }

    #[test]
    fn geometric_distance_is_symmetric() {
        let blades = [0b0001usize, 0b0010, 0b0100, 0b1000];
        let data: Vec<(usize, f64)> = blades
            .iter()
            .enumerate()
            .map(|(i, &b)| (b, (i as f64 + 1.0) * 0.3))
            .collect();
        let data2: Vec<(usize, f64)> = blades
            .iter()
            .enumerate()
            .map(|(i, &b)| (b, (i as f64 + 2.0) * 0.2))
            .collect();
        for shift in 0..100_u64 {
            let s = (shift as f64).mul_add(0.01, 0.1);
            let a = make_vec(&data.iter().map(|&(i, v)| (i, v * s)).collect::<Vec<_>>());
            let b = make_vec(
                &data2
                    .iter()
                    .map(|&(i, v)| (i, v * (s + 0.05)))
                    .collect::<Vec<_>>(),
            );
            let dab = geometric_distance(&a, &b);
            let dba = geometric_distance(&b, &a);
            assert!(
                (dab - dba).abs() < 1e-10,
                "asymmetry at shift {}: dab={} dba={}",
                shift,
                dab,
                dba
            );
        }
    }

    #[test]
    fn geometric_distance_zero_for_identical() {
        let v = make_vec(&[(0b0001, 1.0), (0b0010, 1.0)]);
        let d = geometric_distance(&v, &v);
        assert_eq!(d, 0.0, "d(v,v) debe ser exactamente 0.0");
        assert!(!d.is_nan());
    }

    #[test]
    fn geometric_distance_triangle_inequality() {
        let a = make_vec(&[(0b0001, 1.0), (0b0010, 0.5)]);
        let b = make_vec(&[(0b0001, 0.5), (0b0100, 1.0)]);
        let c = make_vec(&[(0b0010, 0.8), (0b1000, 0.3)]);
        let dab = geometric_distance(&a, &b);
        let dbc = geometric_distance(&b, &c);
        let dac = geometric_distance(&a, &c);
        assert!(
            dac <= dab + dbc + 1e-10,
            "triangle inequality violated: d(a,c)={} > d(a,b)+d(b,c)={}",
            dac,
            dab + dbc
        );
    }

    #[test]
    fn fast_bivector_distance_is_symmetric() {
        let blades = [0b0001usize, 0b0010, 0b0100, 0b1000];
        let data: Vec<(usize, f64)> = blades
            .iter()
            .enumerate()
            .map(|(i, &b)| (b, (i as f64 + 1.0) * 0.3))
            .collect();
        let data2: Vec<(usize, f64)> = blades
            .iter()
            .enumerate()
            .map(|(i, &b)| (b, (i as f64 + 2.0) * 0.2))
            .collect();
        #[allow(deprecated)]
        for shift in 0..100_u64 {
            let s = (shift as f64).mul_add(0.01, 0.1);
            let a = make_vec(&data.iter().map(|&(i, v)| (i, v * s)).collect::<Vec<_>>());
            let b = make_vec(
                &data2
                    .iter()
                    .map(|&(i, v)| (i, v * (s + 0.05)))
                    .collect::<Vec<_>>(),
            );
            let dab = fast_bivector_distance(&a, &b);
            let dba = fast_bivector_distance(&b, &a);
            assert!(
                (dab - dba).abs() < 1e-10,
                "asymmetry at shift {}: dab={} dba={}",
                shift,
                dab,
                dba
            );
        }
    }

    #[test]
    fn fast_bivector_distance_zero_for_identical() {
        let v = make_vec(&[(0b0001, 1.0), (0b0010, 1.0)]);
        #[allow(deprecated)]
        let d = fast_bivector_distance(&v, &v);
        assert_eq!(d, 0.0, "d(v,v) debe ser exactamente 0.0");
        assert!(!d.is_nan());
    }

    #[test]
    fn fast_bivector_distance_identical_vectors_is_zero() {
        let v = make_vec(&[(0b0001, 1.0), (0b0010, 1.0)]);
        #[allow(deprecated)]
        let d = fast_bivector_distance(&v, &v);
        assert!(d != f64::MAX, "d(v,v) no debe ser f64::MAX");
        assert!(!d.is_nan());
    }

    /// Verifica que fast_bivector_distance_from_dense produce la misma distancia
    /// que geometric_distance cuando el denso es extraído del sparse.
    ///
    /// Esto garantiza que el path f16 (hnsw-f16) produce los mismos vecinos
    /// que el path f32 estándar — consistencia métrica entre ambos paths.
    ///
    /// AX-ID: AXIOMA-014, LEY_FUNDACIONAL §3.1
    #[test]
    fn fast_bivector_distance_from_dense_matches_geometric_distance() {
        let a = make_vec(&[(0b0001, 1.0), (0b0010, 0.5), (0b0011, 0.3)]);
        let b = make_vec(&[(0b0001, 0.2), (0b0100, 0.8), (0b1111, 0.1)]);

        // Extraer a como denso (simula descompresión f16 → f64)
        let a_dense: [f64; 16] = a.coeffs;

        let d_sparse = geometric_distance(&a, &b);
        let d_dense = fast_bivector_distance_from_dense(&a_dense, &b);

        assert!(
            (d_sparse - d_dense).abs() < 1e-12,
            "fast_bivector_distance_from_dense debe coincidir con geometric_distance: \
             sparse={d_sparse:.12}, dense={d_dense:.12}"
        );
    }
}
