pub(super) use super::{
    geometric_product_dispatch_by_mask, geometric_product_scalar_sparse, TOTAL_BLADES,
};

#[cfg(all(
    target_arch = "x86_64",
    feature = "avx512",
    not(feature = "deterministic_strict")
))]
pub(super) use super::geometric_product_scalar_dense;

#[cfg(all(
    target_arch = "x86_64",
    feature = "avx512",
    not(feature = "deterministic_strict")
))]
pub(super) use super::geometric_product_x86_avx512_dense;

#[cfg(all(
    target_arch = "x86_64",
    feature = "avx512",
    not(feature = "deterministic_strict")
))]
#[path = "avx512_tests.rs"]
mod avx512_tests;

#[path = "scalar_dispatch_tests.rs"]
mod scalar_dispatch_tests;
