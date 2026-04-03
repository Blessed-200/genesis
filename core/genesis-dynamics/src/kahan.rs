//! Compensated floating-point accumulation primitives used by dynamics hot paths.
//!
//! This module provides a compact `KahanAccumulator` that keeps both the running
//! sum and the compensation term, reducing cancellation error in long reduction chains.
//! `merge` is provided for deterministic partial reduction composition in rayon paths.
//!
//! AX-ID: AXIOMA-006, H_dinámica (LEY_FUNDACIONAL §3.2)

/// Kahan compensated accumulator for `f64` reductions.
///
/// AX-ID: AXIOMA-006, H_dinámica (LEY_FUNDACIONAL §3.2)
#[derive(Clone, Copy, Debug, Default)]
pub struct KahanAccumulator {
    sum: f64,
    compensation: f64,
}

impl KahanAccumulator {
    /// Creates a zero-initialized accumulator.
    ///
    /// AX-ID: AXIOMA-006, H_dinámica (LEY_FUNDACIONAL §3.2)
    #[inline]
    pub const fn new() -> Self {
        Self {
            sum: 0.0,
            compensation: 0.0,
        }
    }

    /// Adds one value using Kahan compensation.
    ///
    /// AX-ID: AXIOMA-006, H_dinámica (LEY_FUNDACIONAL §3.2)
    #[inline]
    pub fn add(&mut self, value: f64) {
        let y = value - self.compensation;
        let t = self.sum + y;
        self.compensation = (t - self.sum) - y;
        self.sum = t;
    }

    /// Merges another partial accumulator preserving compensation quality.
    ///
    /// AX-ID: AXIOMA-006, H_dinámica (LEY_FUNDACIONAL §3.2)
    #[inline]
    pub fn merge(&mut self, other: Self) {
        self.add(other.sum);
        self.add(-other.compensation);
    }

    /// Returns the accumulated sum with compensation applied.
    ///
    /// AX-ID: AXIOMA-006, H_dinámica (LEY_FUNDACIONAL §3.2)
    #[inline]
    pub fn total(self) -> f64 {
        self.sum - self.compensation
    }

    /// Returns the accumulated sum with compensation applied.
    ///
    /// AX-ID: AXIOMA-006, H_dinámica (LEY_FUNDACIONAL §3.2)
    #[inline]
    pub fn sum(self) -> f64 {
        self.total()
    }

    /// Returns the absolute compensation magnitude for precision diagnostics.
    ///
    /// AX-ID: AXIOMA-006, H_dinámica (LEY_FUNDACIONAL §3.2)
    #[inline]
    pub const fn compensation_abs(self) -> f64 {
        self.compensation.abs()
    }
}

#[cfg(test)]
mod tests {
    use super::KahanAccumulator;

    #[test]
    fn merge_combines_partial_sums() {
        let mut left = KahanAccumulator::new();
        left.add(1.25);
        left.add(2.75);

        let mut right = KahanAccumulator::new();
        right.add(3.5);
        right.add(4.5);

        left.merge(right);
        assert!((left.total() - 12.0).abs() < 1e-12);
    }

    #[test]
    fn merge_preserves_followup_add_path() {
        let mut partial = KahanAccumulator::new();
        partial.add(1.0e16);
        partial.add(1.0);
        assert!(partial.compensation_abs() > 0.0);

        let mut merged = KahanAccumulator::new();
        merged.merge(partial);
        merged.add(1.0);

        let mut direct = KahanAccumulator::new();
        direct.add(1.0e16);
        direct.add(1.0);
        direct.add(1.0);

        assert!((merged.total() - direct.total()).abs() < 1e-12);
    }
}
