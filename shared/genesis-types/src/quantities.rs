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

            #[inline]
            fn add(self, rhs: Self) -> Self::Output {
                Self::new(self.0 + rhs.0)
            }
        }

        impl Sub for $ty {
            type Output = Self;

            #[inline]
            fn sub(self, rhs: Self) -> Self::Output {
                Self::new(self.0 - rhs.0)
            }
        }

        impl Mul<f64> for $ty {
            type Output = Self;

            #[inline]
            fn mul(self, rhs: f64) -> Self::Output {
                Self::new(self.0 * rhs)
            }
        }

        impl Mul<$ty> for f64 {
            type Output = $ty;

            #[inline]
            fn mul(self, rhs: $ty) -> Self::Output {
                $ty::new(self * rhs.0)
            }
        }

        impl Div<f64> for $ty {
            type Output = Self;

            #[inline]
            fn div(self, rhs: f64) -> Self::Output {
                Self::new(self.0 / rhs)
            }
        }

        impl AddAssign for $ty {
            #[inline]
            fn add_assign(&mut self, rhs: Self) {
                self.0 += rhs.0;
            }
        }

        impl SubAssign for $ty {
            #[inline]
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
    #[inline]
    pub const fn new(value: f64) -> Self {
        Self(value)
    }

    /// Constructs without any semantic validation.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that `value` respects the domain invariants
    /// of [`Phase`] (i.e. a valid finite radian angle meaningful in context).
    #[inline]
    pub const unsafe fn new_unchecked(value: f64) -> Self {
        Self(value)
    }

    /// Returns the scalar representation.
    #[inline]
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
    #[inline]
    pub const fn new(value: f64) -> Self {
        Self(value)
    }

    /// Constructs without any semantic validation.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that `value` respects the domain invariants
    /// of [`Amplitude`] (i.e. a non-negative finite scalar where applicable).
    #[inline]
    pub const unsafe fn new_unchecked(value: f64) -> Self {
        Self(value)
    }

    /// Returns the scalar representation.
    #[inline]
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
    #[inline]
    pub const fn new(value: f64) -> Self {
        Self(value)
    }

    /// Constructs without any semantic validation.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that `value` respects the domain invariants
    /// of [`Frequency`] (i.e. a finite scalar valid as a natural frequency).
    #[inline]
    pub const unsafe fn new_unchecked(value: f64) -> Self {
        Self(value)
    }

    /// Returns the scalar representation.
    #[inline]
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
    #[inline]
    pub const fn new(value: f64) -> Self {
        Self(value)
    }

    /// Constructs without any semantic validation.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that `value` respects the domain invariants
    /// of [`TimeStep`] (i.e. a finite positive scalar meaningful as an integration step).
    #[inline]
    pub const unsafe fn new_unchecked(value: f64) -> Self {
        Self(value)
    }

    /// Returns the scalar representation.
    #[inline]
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
    #[inline]
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
    /// # Safety
    ///
    /// The caller must guarantee that `value` is finite and in `[0.0, 1.0]`.
    #[inline]
    pub const unsafe fn new_unchecked(value: f64) -> Self {
        Self(value)
    }

    /// Returns the scalar representation.
    #[inline]
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
    #[inline]
    pub const fn new(value: f64) -> Self {
        Self(value)
    }

    /// Constructs without any semantic validation.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that `value` respects the domain invariants
    /// of [`Temperature`] (i.e. a finite scalar valid as a thermal energy scale).
    #[inline]
    pub const unsafe fn new_unchecked(value: f64) -> Self {
        Self(value)
    }

    /// Returns the scalar representation.
    #[inline]
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
    #[inline]
    pub const fn new(value: f64) -> Self {
        Self(value)
    }

    /// Constructs without any semantic validation.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that `value` respects the domain invariants
    /// of [`LearningRate`] (i.e. a finite scalar valid as a learning rate).
    #[inline]
    pub const unsafe fn new_unchecked(value: f64) -> Self {
        Self(value)
    }

    /// Returns the scalar representation.
    #[inline]
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
