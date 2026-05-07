//! Causal separation classification in G(1,3).
//!
//! AX-ID: AXIOMA-002, H_estructura (LEY_FUNDACIONAL §3.1)

pub const LIGHTLIKE_TOL: f64 = 1e-10;

/// Separation class between two nodes in G(1,3).
///
/// s²(A,B) = (Δe₀)² - (Δe₁)² - (Δe₂)² - (Δe₃)²,
/// where e₀ is blade index 1 and e₁/e₂/e₃ are indices 2/4/8.
///
/// AX-ID: AXIOMA-002, H_estructura (LEY_FUNDACIONAL §3.1)
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
    /// Computes the causal separation class for two G(1,3) dense blade vectors.
    ///
    /// AX-ID: AXIOMA-002, H_estructura (LEY_FUNDACIONAL §3.1)
    #[must_use]
    pub fn compute(blade_a: &[f64; 16], blade_b: &[f64; 16]) -> Self {
        let delta_t = blade_b[1] - blade_a[1];
        let delta_x = blade_b[2] - blade_a[2];
        let delta_y = blade_b[4] - blade_a[4];
        let delta_z = blade_b[8] - blade_a[8];
        let s_sq =
            delta_t * delta_t - delta_x * delta_x - delta_y * delta_y - delta_z * delta_z;

        if s_sq.abs() < LIGHTLIKE_TOL {
            Self::Lightlike
        } else if s_sq > 0.0 {
            Self::Timelike { separation_sq: s_sq }
        } else {
            Self::Spacelike { separation_sq: s_sq }
        }
    }

    /// Returns true when the relation is timelike or lightlike.
    ///
    /// AX-ID: AXIOMA-002, H_estructura (LEY_FUNDACIONAL §3.1)
    #[must_use]
    pub const fn is_causal(self) -> bool {
        matches!(self, Self::Timelike { .. } | Self::Lightlike)
    }
}
