//! Numbers of the simplex: exact rationals and floats behind one trait.

use std::fmt;
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign};

use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Signed, ToPrimitive, Zero};

// ==============================================================================
// Rational
// ==============================================================================

/// An exact rational number (a newtype, so that the representation can change
/// without changing the API).
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Rational(BigRational);

impl Rational {
    /// The rational `num / den`.
    ///
    /// # Panics
    ///
    /// If `den` is zero.
    #[must_use]
    pub fn new(num: i64, den: i64) -> Rational {
        Rational(BigRational::new(BigInt::from(num), BigInt::from(den)))
    }

    /// The integer `n`.
    #[must_use]
    pub fn from_int(n: i64) -> Rational {
        Rational::new(n, 1)
    }

    /// The decimal Python's `repr` prints for `x` (the shortest digits that
    /// round-trip), as an exact rational: `0.1` is `1/10`, not the binary
    /// expansion of the float. `None` if `x` is not finite.
    #[must_use]
    pub fn from_f64(x: f64) -> Option<Rational> {
        if !x.is_finite() {
            return None;
        }
        // Rust's `{:e}` prints the shortest round-trip digits: `-6.86e0`.
        let text = format!("{x:e}");
        let (mantissa, exponent) = text.split_once('e')?;
        let (digits, fraction) = match mantissa.split_once('.') {
            Some((integer, fraction)) => (format!("{integer}{fraction}"), fraction.len()),
            None => (mantissa.to_owned(), 0),
        };
        let numerator: BigInt = digits.parse().ok()?;
        let scale = exponent.parse::<i64>().ok()? - i64::try_from(fraction).ok()?;
        let power = BigInt::from(10).pow(u32::try_from(scale.unsigned_abs()).ok()?);
        Some(Rational(if scale >= 0 {
            BigRational::from_integer(numerator * power)
        } else {
            BigRational::new(numerator, power)
        }))
    }

    /// The nearest `f64`.
    #[must_use]
    pub fn to_f64(&self) -> f64 {
        self.0.to_f64().unwrap_or(f64::NAN)
    }
}

impl fmt::Display for Rational {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl Neg for Rational {
    type Output = Rational;
    fn neg(self) -> Rational {
        Rational(-self.0)
    }
}

macro_rules! rational_ops {
    ($($trait:ident $method:ident $assign_trait:ident $assign_method:ident),*) => {$(
        impl $trait<&Rational> for Rational {
            type Output = Rational;
            fn $method(self, rhs: &Rational) -> Rational {
                Rational($trait::$method(self.0, &rhs.0))
            }
        }
        impl $assign_trait<&Rational> for Rational {
            fn $assign_method(&mut self, rhs: &Rational) {
                $assign_trait::$assign_method(&mut self.0, &rhs.0);
            }
        }
    )*};
}

rational_ops!(Add add AddAssign add_assign, Sub sub SubAssign sub_assign);

impl Mul<&Rational> for Rational {
    type Output = Rational;
    fn mul(self, rhs: &Rational) -> Rational {
        Rational(self.0 * &rhs.0)
    }
}

impl Div<&Rational> for Rational {
    type Output = Rational;
    fn div(self, rhs: &Rational) -> Rational {
        Rational(self.0 / &rhs.0)
    }
}

// ==============================================================================
// Number
// ==============================================================================

// ~ Tolerances of the float stage. They only steer the float guess of the
// ~ optimal basis: the exact stage that follows decides every result, so none of
// ~ them can change a status or an optimum, only how many exact pivots are needed.
// ~ Values are those of HiGHS's defaults for the same roles.
/// Allowed bound violation, relative to `max(1, |bound|)` (primal feasibility).
const FEASIBILITY_TOL: f64 = 1e-7;
/// Smallest `|reduced cost|` counted as improving (dual feasibility).
const OPTIMALITY_TOL: f64 = 1e-7;
/// Smallest `|pivot|` accepted in the ratio test.
const PIVOT_TOL: f64 = 1e-9;
/// Smaller values are zero.
const ZERO_TOL: f64 = 1e-12;
/// Relative threshold of LU pivots: `|a| >= LU_THRESHOLD * max |column|`.
const LU_THRESHOLD: f64 = 0.1;

/// Arithmetic of the simplex, exact ([`Rational`]) or floating point (`f64`).
pub trait Number:
    Clone
    + PartialOrd
    + fmt::Debug
    + Send
    + Neg<Output = Self>
    + for<'a> Add<&'a Self, Output = Self>
    + for<'a> Sub<&'a Self, Output = Self>
    + for<'a> Mul<&'a Self, Output = Self>
    + for<'a> Div<&'a Self, Output = Self>
    + for<'a> AddAssign<&'a Self>
    + for<'a> SubAssign<&'a Self>
{
    /// Whether the arithmetic is exact (and slow enough to be worth caching).
    const EXACT: bool;
    /// Zero.
    fn zero() -> Self;
    /// One.
    fn one() -> Self;
    /// The nearest value (`f64`) or the same value (`Rational`).
    fn from_rational(q: &Rational) -> Self;
    /// Absolute value.
    #[must_use]
    fn abs(&self) -> Self;
    /// Counted as zero (`f64`: below [`ZERO_TOL`]; `Rational`: equal to zero).
    fn is_zero(&self) -> bool;
    /// Allowed bound violation, relative to `max(1, |bound|)`.
    fn feasibility_tol() -> Self;
    /// Smallest `|reduced cost|` counted as improving.
    fn optimality_tol() -> Self;
    /// Smallest `|pivot|` accepted in the ratio test.
    fn pivot_tol() -> Self;
    /// Relative threshold of LU pivots.
    fn lu_threshold() -> Self;
}

impl Number for f64 {
    const EXACT: bool = false;

    fn zero() -> f64 {
        0.0
    }
    fn one() -> f64 {
        1.0
    }
    fn from_rational(q: &Rational) -> f64 {
        q.to_f64()
    }
    fn abs(&self) -> f64 {
        f64::abs(*self)
    }
    fn is_zero(&self) -> bool {
        f64::abs(*self) <= ZERO_TOL
    }
    fn feasibility_tol() -> f64 {
        FEASIBILITY_TOL
    }
    fn optimality_tol() -> f64 {
        OPTIMALITY_TOL
    }
    fn pivot_tol() -> f64 {
        PIVOT_TOL
    }
    fn lu_threshold() -> f64 {
        LU_THRESHOLD
    }
}

impl Number for Rational {
    const EXACT: bool = true;

    fn zero() -> Rational {
        Rational(BigRational::zero())
    }
    fn one() -> Rational {
        Rational(BigRational::one())
    }
    fn from_rational(q: &Rational) -> Rational {
        q.clone()
    }
    fn abs(&self) -> Rational {
        Rational(self.0.abs())
    }
    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
    fn feasibility_tol() -> Rational {
        Self::zero()
    }
    fn optimality_tol() -> Rational {
        Self::zero()
    }
    fn pivot_tol() -> Rational {
        Self::zero()
    }
    fn lu_threshold() -> Rational {
        Self::zero()
    }
}

#[cfg(test)]
mod tests {
    use super::{Number, Rational};

    #[test]
    fn from_f64_is_the_printed_decimal() {
        assert_eq!(Rational::from_f64(0.1), Some(Rational::new(1, 10)));
        assert_eq!(Rational::from_f64(1e-5), Some(Rational::new(1, 100_000)));
        assert_eq!(Rational::from_f64(-0.0), Some(Rational::from_int(0)));
        assert_eq!(Rational::from_f64(6.86), Some(Rational::new(343, 50)));
        assert_eq!(Rational::from_f64(-1000.0), Some(Rational::from_int(-1000)));
        assert_eq!(
            Rational::from_f64(1.5e20),
            Some(Rational::from_int(15_000_000_000) * &Rational::from_int(10_000_000_000))
        );
        assert_eq!(Rational::from_f64(f64::NAN), None);
        assert_eq!(Rational::from_f64(f64::INFINITY), None);
        assert_eq!(Rational::from_f64(f64::NEG_INFINITY), None);
    }

    #[test]
    fn from_f64_round_trips() {
        for x in [
            0.1,
            2.0 / 3.0,
            1e-300,
            5e-324,
            123_456.789,
            -7.5e10,
            f64::MAX,
            f64::MIN_POSITIVE,
        ] {
            assert_eq!(Rational::from_f64(x).unwrap().to_f64(), x, "{x}");
        }
    }

    #[test]
    fn to_f64_rounds_to_nearest() {
        assert_eq!(Rational::new(1, 3).to_f64(), 1.0 / 3.0);
        assert_eq!(Rational::new(5, 7).to_f64(), 5.0 / 7.0);
        assert_eq!(Rational::new(-2, 3).to_f64(), -2.0 / 3.0);
    }

    #[test]
    fn arithmetic_is_exact() {
        let third = Rational::new(1, 3);
        let sum = third.clone() + &third + &third;
        assert_eq!(sum, Rational::from_int(1));
        assert_eq!(
            Rational::new(1, 2) * &Rational::new(2, 3) / &Rational::new(1, 3),
            Rational::from_int(1)
        );
        assert!(<Rational as Number>::is_zero(
            &(Rational::new(1, 5) - &Rational::new(1, 5))
        ));
    }
}
