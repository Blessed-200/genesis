/// Default sliding window (in iterations) for the Fisher satiation gate.
///
/// `NonZeroUsize` guarantees at compile time that division by this value
/// in `FisherGate::delta_g()` is safe. Using it as loop count is
/// always meaningful (the window never has zero iterations).
///
/// The gate fires when the Fisher metric gradient stays below
/// `FISHER_SATIATION_EPSILON` for this many consecutive iterations.
///
/// Access the raw value with `.get()`.
///
/// AX-ID: AXIOMA-008 — Satiation and Self-Evaluation by Fisher Stability
// SAFETY: 50 is a non-zero literal. NonZeroUsize::new(50).unwrap() is the const-safe
// equivalent but Option::unwrap() is not yet stable as a const fn on our MSRV (Rust 1.75).
// When MSRV >= 1.83, replace with: NonZeroUsize::new(50).unwrap()
#[allow(clippy::useless_nonzero_new_unchecked)]
pub const FISHER_SATIATION_WINDOW: NonZeroUsize = unsafe { NonZeroUsize::new_unchecked(50) };