//! Exact rational linear programming.
//!
//! A bounded-variable revised simplex that returns exact results: a floating
//! point simplex guesses the optimal basis, then the same algorithm, in exact
//! rational arithmetic, finishes from it (usually in no or a few pivots). On
//! top of it, [`fba`] solves flux balance analysis, plain and parsimonious.

pub mod fba;
pub mod lp;
pub mod lu;
pub mod number;
pub mod simplex;

#[cfg(feature = "python")]
mod python;
