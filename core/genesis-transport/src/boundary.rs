use crate::BeliefDistribution;
use genesis_types::{GenesisError, METRIC_WEIGHTS};

#[cfg(test)]
const ROUNDTRIP_TOL: f64 = 3.0;

/// Boundary projector Π and adjoint Π* between sensory events and beliefs.
///
/// AX-ID: AXIOMA-002, AXIOMA-003, H_estructura
pub struct SensoryProjector {
    /// Minimal detectable amplitude.
    pub detection_threshold: f64,
}

/// Sensory event in cognitive spacetime.
///
/// AX-ID: AXIOMA-002
#[derive(Clone, Debug)]
pub struct SensoryEvent {
    /// Coordinates `[t, x, y, z]` in G(1,3).
    pub spacetime_coords: [f64; 4],
    /// Event amplitude.
    pub amplitude: f64,
    /// Monotone event timestamp.
    pub system_timestamp: u64,
}

impl SensoryProjector {
    /// Creates a sensory projector.
    pub fn new(detection_threshold: f64) -> Self {
        Self {
            detection_threshold,
        }
    }

    /// Projects an event into a 16-blade belief distribution.
    pub fn project(&self, event: &SensoryEvent) -> Result<BeliefDistribution, GenesisError> {
        if event.amplitude < self.detection_threshold {
            return Err(GenesisError::InvalidInput(
                "event below detection threshold",
            ));
        }
        let [t, x, y, z] = event.spacetime_coords;
        let mut weights = [0.0; 16];
        weights[0] = event.amplitude.powi(2) * METRIC_WEIGHTS[0];
        weights[1] = (t * event.amplitude).powi(2) * METRIC_WEIGHTS[1];
        weights[2] = (x * event.amplitude).powi(2) * METRIC_WEIGHTS[2];
        weights[4] = (y * event.amplitude).powi(2) * METRIC_WEIGHTS[4];
        weights[8] = (z * event.amplitude).powi(2) * METRIC_WEIGHTS[8];
        BeliefDistribution::from_weights(weights)
    }

    /// Reconstructs an event from a belief (adjoint map).
    #[must_use]
    pub fn project_adjoint(&self, belief: &BeliefDistribution) -> SensoryEvent {
        let t = belief.weights[1].sqrt() / METRIC_WEIGHTS[1].sqrt();
        let x = belief.weights[2].sqrt() / METRIC_WEIGHTS[2].sqrt();
        let y = belief.weights[4].sqrt() / METRIC_WEIGHTS[4].sqrt();
        let z = belief.weights[8].sqrt() / METRIC_WEIGHTS[8].sqrt();
        SensoryEvent {
            spacetime_coords: [t, x, y, z],
            amplitude: belief.weighted_mean().sqrt(),
            system_timestamp: 0,
        }
    }

    /// Computes roundtrip error `||Π*(Π(event)) - event||`.
    pub fn roundtrip_error(&self, event: &SensoryEvent) -> Result<f64, GenesisError> {
        let belief = self.project(event)?;
        let reconstructed = self.project_adjoint(&belief);
        Ok(event
            .spacetime_coords
            .iter()
            .zip(reconstructed.spacetime_coords.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt())
    }
}

#[cfg(test)]
mod tests {
    use super::{SensoryEvent, SensoryProjector, ROUNDTRIP_TOL};

    #[test]
    fn sensory_projector_roundtrip_bounded() {
        let p = SensoryProjector::new(0.01);
        let event = SensoryEvent {
            spacetime_coords: [1.0, 0.2, 0.3, 0.4],
            amplitude: 1.2,
            system_timestamp: 7,
        };
        let err = p.roundtrip_error(&event).expect("roundtrip");
        assert!(err < ROUNDTRIP_TOL);
    }
}
