//! Typed scalar quantities for high-signal public APIs.
//!
//! These wrappers eliminate primitive `f64` ambiguity while preserving
//! zero-cost ABI semantics (`#[repr(transparent)]`) for SIMD hot paths.
//!
//! AX-ID: AXIOMA-006, AXIOMA-008, H_dinámica (LEY_FUNDACIONAL §3.2)

use core::ops::{Add, AddAssign, Div, Mul, Sub, SubAssign};

use bytemuck::{Pod, TransparentWrapper, Zeroable};

use crate::GenesisError;

macro_rules! impl_scalar_arithmetic {
    ($ty:ident) => {
        impl Add for $ty {
            type Output = Self;

            #[inline(always)]
            fn add(self, rhs: Self) -> Self::Output {
                Self::new(self.0 + rhs.0)
            }
        }

        impl Sub for $ty {
            type Output = Self;

            #[inline(always)]
            fn sub(self, rhs: Self) -> Self::Output {
                Self::new(self.0 - rhs.0)
            }
        }

        impl Mul<f64> for $ty {
            type Output = Self;

            #[inline(always)]
            fn mul(self, rhs: f64) -> Self::Output {
                Self::new(self.0 * rhs)
            }
        }

        impl Mul<$ty> for f64 {
            type Output = $ty;

            #[inline(always)]
            fn mul(self, rhs: $ty) -> Self::Output {
                $ty::new(self * rhs.0)
            }
        }

        impl Div<f64> for $ty {
            type Output = Self;

            #[inline(always)]
            fn div(self, rhs: f64) -> Self::Output {
                Self::new(self.0 / rhs)
            }
        }

        impl AddAssign for $ty {
            #[inline(always)]
            fn add_assign(&mut self, rhs: Self) {
                self.0 += rhs.0;
            }
        }

        impl SubAssign for $ty {
            #[inline(always)]
            fn sub_assign(&mut self, rhs: Self) {
                self.0 -= rhs.0;
            }
        }
    };
}

/// Clifford-grade phase angle (radians).
///
/// AX-ID: AXIOMA-006, H_dinámica (LEY_FUNDACIONAL §3.2)
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd, TransparentWrapper, Zeroable, Pod)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Phase(f64);

impl Phase {
    /// Constructs a phase wrapper from a raw scalar.
    #[inline(always)]
    pub const fn new(value: f64) -> Self {
        Self(value)
    }

    /// Constructs without any semantic validation.
    ///
    /// // SAFETY: Caller guarantees the scalar follows the surrounding domain invariants.
    #[inline(always)]
    pub const unsafe fn new_unchecked(value: f64) -> Self {
        Self(value)
    }

    /// Returns the scalar representation.
    #[inline(always)]
    pub const fn as_f64(self) -> f64 {
        self.0
    }
}

/// Oscillator amplitude scalar.
///
/// AX-ID: AXIOMA-006, AXIOMA-008, H_dinámica (LEY_FUNDACIONAL §3.2)
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd, TransparentWrapper, Zeroable, Pod)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Amplitude(f64);

impl Amplitude {
    /// Constructs an amplitude wrapper from a raw scalar.
    #[inline(always)]
    pub const fn new(value: f64) -> Self {
        Self(value)
    }

    /// Constructs without any semantic validation.
    ///
    /// // SAFETY: Caller guarantees the scalar follows the surrounding domain invariants.
    #[inline(always)]
    pub const unsafe fn new_unchecked(value: f64) -> Self {
        Self(value)
    }

    /// Returns the scalar representation.
    #[inline(always)]
    pub const fn as_f64(self) -> f64 {
        self.0
    }
}

/// Natural oscillator frequency scalar.
///
/// AX-ID: AXIOMA-006, H_dinámica (LEY_FUNDACIONAL §3.2)
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd, TransparentWrapper, Zeroable, Pod)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Frequency(f64);

impl Frequency {
    /// Constructs a frequency wrapper from a raw scalar.
    #[inline(always)]
    pub const fn new(value: f64) -> Self {
        Self(value)
    }

    /// Constructs without any semantic validation.
    ///
    /// // SAFETY: Caller guarantees the scalar follows the surrounding domain invariants.
    #[inline(always)]
    pub const unsafe fn new_unchecked(value: f64) -> Self {
        Self(value)
    }

    /// Returns the scalar representation.
    #[inline(always)]
    pub const fn as_f64(self) -> f64 {
        self.0
    }
}

/// Integration time step scalar.
///
/// AX-ID: AXIOMA-002, AXIOMA-006, H_dinámica (LEY_FUNDACIONAL §3.2)
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd, TransparentWrapper, Zeroable, Pod)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TimeStep(f64);

impl TimeStep {
    /// Constructs a time-step wrapper from a raw scalar.
    #[inline(always)]
    pub const fn new(value: f64) -> Self {
        Self(value)
    }

    /// Constructs without any semantic validation.
    ///
    /// // SAFETY: Caller guarantees the scalar follows the surrounding domain invariants.
    #[inline(always)]
    pub const unsafe fn new_unchecked(value: f64) -> Self {
        Self(value)
    }

    /// Returns the scalar representation.
    #[inline(always)]
    pub const fn as_f64(self) -> f64 {
        self.0
    }
}

/// Global synchrony order parameter in `[0, 1]`.
///
/// AX-ID: AXIOMA-006, H_dinámica (LEY_FUNDACIONAL §3.2)
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd, TransparentWrapper, Zeroable, Pod)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SyncOrder(f64);

impl SyncOrder {
    /// Fallible constructor enforcing the closed interval `[0, 1]`.
    ///
    /// # Errors
    /// Returns [`GenesisError::InvalidInput`] when `value` is outside `[0, 1]`
    /// or non-finite.
    #[inline(always)]
    pub fn try_new(value: f64) -> Result<Self, GenesisError> {
        if value.is_finite() && (0.0..=1.0).contains(&value) {
            Ok(Self(value))
        } else {
            Err(GenesisError::InvalidInput(
                "SyncOrder must be finite and within [0.0, 1.0]",
            ))
        }
    }

    /// Constructs without interval validation.
    ///
    /// // SAFETY: Caller guarantees `value` is finite and within `[0, 1]`.
    #[inline(always)]
    pub const unsafe fn new_unchecked(value: f64) -> Self {
        Self(value)
    }

    /// Returns the scalar representation.
    #[inline(always)]
    pub const fn as_f64(self) -> f64 {
        self.0
    }
}

/// Thermal energy scale (`kT`) used in stochastic Kuramoto integration.
///
/// AX-ID: AXIOMA-006, H_dinámica (LEY_FUNDACIONAL §3.2)
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd, TransparentWrapper, Zeroable, Pod)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Temperature(f64);

impl Temperature {
    /// Constructs a temperature wrapper from a raw scalar.
    #[inline(always)]
    pub const fn new(value: f64) -> Self {
        Self(value)
    }

    /// Constructs without any semantic validation.
    ///
    /// // SAFETY: Caller guarantees the scalar follows the surrounding domain invariants.
    #[inline(always)]
    pub const unsafe fn new_unchecked(value: f64) -> Self {
        Self(value)
    }

    /// Returns the scalar representation.
    #[inline(always)]
    pub const fn as_f64(self) -> f64 {
        self.0
    }
}

/// Learning-rate scalar for gauge adaptation.
///
/// AX-ID: AXIOMA-006, AXIOMA-007, H_estructura (LEY_FUNDACIONAL §3.1)
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd, TransparentWrapper, Zeroable, Pod)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LearningRate(f64);

impl LearningRate {
    /// Constructs a learning-rate wrapper from a raw scalar.
    #[inline(always)]
    pub const fn new(value: f64) -> Self {
        Self(value)
    }

    /// Constructs without any semantic validation.
    ///
    /// // SAFETY: Caller guarantees the scalar follows the surrounding domain invariants.
    #[inline(always)]
    pub const unsafe fn new_unchecked(value: f64) -> Self {
        Self(value)
    }

    /// Returns the scalar representation.
    #[inline(always)]
    pub const fn as_f64(self) -> f64 {
        self.0
    }
}

/// Small complex scalar used by oscillator kernels.
///
/// AX-ID: AXIOMA-006, H_dinámica (LEY_FUNDACIONAL §3.2)
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Zeroable, Pod)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ComplexPhasor {
    /// Real part.
    pub re: f64,
    /// Imaginary part.
    pub im: f64,
}

impl_scalar_arithmetic!(Phase);
impl_scalar_arithmetic!(Amplitude);
impl_scalar_arithmetic!(Frequency);
impl_scalar_arithmetic!(TimeStep);
