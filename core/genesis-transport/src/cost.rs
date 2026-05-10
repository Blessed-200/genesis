use crate::belief::BeliefDistribution;
use genesis_types::METRIC_WEIGHTS;

/// Lorentzian 16x16 transport cost matrix in row-major storage.
///
/// AX-ID: AXIOMA-001, AXIOMA-002, H_compresión
pub struct CostMatrix16 {
    values: [f64; 256],
}

impl CostMatrix16 {
    /// Builds a Lorentzian cost matrix from blade bitmask coordinates.
    ///
    /// AX-ID: AXIOMA-001, AXIOMA-002
    #[must_use]
    pub fn from_lorentzian_metric() -> Self {
        let mut values = [0.0; 256];
        for i in 0..16 {
            for j in 0..16 {
                let dt = ((j & 1) as f64) - ((i & 1) as f64);
                let dx = (((j >> 1) & 1) as f64) - (((i >> 1) & 1) as f64);
                let dy = (((j >> 2) & 1) as f64) - (((i >> 2) & 1) as f64);
                let dz = (((j >> 3) & 1) as f64) - (((i >> 3) & 1) as f64);
                let s_sq = dt * dt - dx * dx - dy * dy - dz * dz;
                values[i * 16 + j] = s_sq.abs() * METRIC_WEIGHTS[i] * METRIC_WEIGHTS[j];
            }
        }
        for i in 0..16 { values[i * 16 + i] = 0.0; }
        Self { values }
    }

    /// Returns c(i,j).
    #[inline]
    #[must_use]
    pub fn get(&self, i: usize, j: usize) -> f64 { self.values[i * 16 + j] }

    /// Lower bound of weighted outgoing transport cost.
    ///
    /// AX-ID: H_compresión
    #[must_use]
    pub fn min_cost_lower_bound(&self, source: &BeliefDistribution) -> f64 {
        source.weights.iter().enumerate().map(|(i, w)| {
            let mut min_cost = f64::INFINITY;
            for j in 0..16 {
                if j != i { min_cost = min_cost.min(self.values[i * 16 + j]); }
            }
            w * min_cost
        }).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::CostMatrix16;

    #[test]
    fn cost_matrix_diagonal_is_zero() {
        let c = CostMatrix16::from_lorentzian_metric();
        for i in 0..16 { assert!(c.get(i, i).abs() < 1e-12); }
    }

    #[test]
    fn cost_matrix_is_symmetric() {
        let c = CostMatrix16::from_lorentzian_metric();
        for i in 0..16 { for j in 0..16 { assert!((c.get(i, j) - c.get(j, i)).abs() < 1e-12); } }
    }
}
