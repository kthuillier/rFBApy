//! Python module `rFBApy.rational_lp_solver`: the class `Fba`.
//!
//! Floats are converted at this boundary only: a float is read as the decimal
//! Python prints for it (`0.1` is `1/10`), and results are rounded to the
//! nearest float.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::fba::{Bounds, Fba};
use crate::lp::Status;
use crate::number::Rational;

fn finite(x: f64, what: &str) -> PyResult<Rational> {
    Rational::from_f64(x).ok_or_else(|| PyValueError::new_err(format!("{what} must be finite, got {x}")))
}

/// A lower bound: `-inf` is no bound.
fn lower_bound(x: f64) -> PyResult<Option<Rational>> {
    if x == f64::NEG_INFINITY {
        Ok(None)
    } else {
        finite(x, "a lower bound").map(Some)
    }
}

/// An upper bound: `+inf` is no bound.
fn upper_bound(x: f64) -> PyResult<Option<Rational>> {
    if x == f64::INFINITY {
        Ok(None)
    } else {
        finite(x, "an upper bound").map(Some)
    }
}

fn bound_pair((lower, upper): (f64, f64)) -> PyResult<Bounds> {
    Ok((lower_bound(lower)?, upper_bound(upper)?))
}

/// Exact flux balance analysis and parsimonious FBA (see `fba::Fba`).
#[pyclass(module = "rFBApy.rational_lp_solver", name = "Fba")]
pub(crate) struct PyFba {
    fba: Fba,
    reactions: usize,
}

// pyo3 extracts the arguments as owned values.
#[allow(clippy::needless_pass_by_value)]
#[pymethods]
impl PyFba {
    /// `rows` metabolites; `stoichiometry` lists (metabolite, reaction,
    /// coefficient); `bounds` has a (lower, upper) pair per reaction, `-inf` and
    /// `inf` for none; `objective` is the reaction to maximize.
    #[new]
    fn new(
        rows: usize,
        stoichiometry: Vec<(usize, usize, f64)>,
        bounds: Vec<(f64, f64)>,
        objective: usize,
    ) -> PyResult<PyFba> {
        let reactions = bounds.len();
        if objective >= reactions {
            return Err(PyValueError::new_err("the objective is not a reaction"));
        }
        let mut triples = Vec::with_capacity(stoichiometry.len());
        for (row, reaction, coefficient) in stoichiometry {
            if row >= rows || reaction >= reactions {
                return Err(PyValueError::new_err("stoichiometry index out of range"));
            }
            triples.push((row, reaction, finite(coefficient, "a coefficient")?));
        }
        let bounds = bounds.into_iter().map(bound_pair).collect::<PyResult<Vec<Bounds>>>()?;
        Ok(PyFba {
            fba: Fba::new(rows, &triples, bounds, objective),
            reactions,
        })
    }

    /// Solves with the (reaction, lower, upper) bounds replacing the reaction's
    /// own for this call only. Returns `(optimum, fluxes)`: `(None, [])` if
    /// infeasible, `(inf, fluxes)` if unbounded.
    fn solve(&mut self, bounds: Vec<(usize, f64, f64)>, pfba: bool) -> PyResult<(Option<f64>, Vec<f64>)> {
        let mut converted = Vec::with_capacity(bounds.len());
        for (reaction, lower, upper) in bounds {
            if reaction >= self.reactions {
                return Err(PyValueError::new_err("bound on a reaction that does not exist"));
            }
            let (lower, upper) = bound_pair((lower, upper))?;
            converted.push((reaction, lower, upper));
        }
        let (status, fluxes) = self.fba.solve(&converted, pfba);
        let fluxes = fluxes.iter().map(Rational::to_f64).collect();
        Ok(match status {
            Status::Optimal(optimum) => (Some(optimum.to_f64()), fluxes),
            Status::Infeasible => (None, fluxes),
            Status::Unbounded => (Some(f64::INFINITY), fluxes),
        })
    }
}

#[pymodule]
fn rational_lp_solver(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFba>()
}
