//! Causal separation classification in G(1,3).
//!
//! AX-ID: AXIOMA-002, `H_estructura` (`LEY_FUNDACIONAL` §3.1)

pub const LIGHTLIKE_TOL: f64 = 1e-10;

/// Separation class between two nodes in G(1,3).
///
/// AX-ID: AXIOMA-002, `H_estructura` (`LEY_FUNDACIONAL` §3.1)
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CausalSeparation {
    /// Timelike interval with positive Minkowski separation.
    Timelike { separation_sq: f64 },
    /// Null interval within `LIGHTLIKE_TOL`.
    Lightlike,
    /// Spacelike interval with negative Minkowski separation.
    Spacelike { separation_sq: f64 },
}

impl CausalSeparation {
    /// Computes the separation class.
    #[must_use]
    pub fn compute(blade_a: &[f64; 16], blade_b: &[f64; 16]) -> Self {
        let delta_t = blade_b[1] - blade_a[1];
        let delta_x = blade_b[2] - blade_a[2];
        let delta_y = blade_b[4] - blade_a[4];
        let delta_z = blade_b[8] - blade_a[8];
        let s_sq = delta_t.mul_add(
            delta_t,
            -delta_x.mul_add(delta_x, delta_y.mul_add(delta_y, delta_z * delta_z)),
        );

        if s_sq.abs() < LIGHTLIKE_TOL {
            Self::Lightlike
        } else if s_sq > 0.0 {
            Self::Timelike {
                separation_sq: s_sq,
            }
        } else {
            Self::Spacelike {
                separation_sq: s_sq,
            }
        }
    }

    /// Returns true when relation is timelike or lightlike.
    #[must_use]
    pub const fn is_causal(self) -> bool {
        matches!(self, Self::Timelike { .. } | Self::Lightlike)
    }

    /// Returns true when `effect` is in the causal future of `cause`.
    #[must_use]
    pub fn is_forward_causal(cause: &[f64; 16], effect: &[f64; 16]) -> bool {
        let separation = Self::compute(cause, effect);
        let delta_t = effect[1] - cause[1];
        separation.is_causal() && delta_t > 0.0
    }
}
