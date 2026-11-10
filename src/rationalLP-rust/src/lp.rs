//! An exact linear program: `min c·x` subject to bounds on the variables and on
//! the rows `Σ_j a_ij x_j`.
//!
//! Each solve has two stages (Dhiflaoui et al. 2003; Applegate et al. 2007). A
//! floating point simplex, fast but approximate, runs first and only its final
//! basis is kept. The same simplex then runs in exact arithmetic from that
//! basis, usually in no or a few pivots, and its result is the answer. Float
//! tolerances therefore never change a status or an optimum.
//!
//! The basis of a solve starts the next one, so changing bounds or the objective
//! between solves is a warm start.

use crate::number::{Number, Rational};
use crate::simplex::{Basis, Outcome, Simplex};

/// The result of a solve.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Status {
    /// The minimum of the objective.
    Optimal(Rational),
    /// No point satisfies the bounds.
    Infeasible,
    /// The objective decreases forever.
    Unbounded,
}

/// Iterations of the float stage, per variable. A speed knob only (the exact
/// stage finishes the job): float simplex codes need a small multiple of the
/// number of variables.
const FLOAT_ITERATIONS_PER_VARIABLE: usize = 10;

/// An exact linear program. Rows start without terms and with bounds `[0, 0]`.
#[derive(Clone, Debug)]
pub struct Lp {
    rows: usize,
    exact: Simplex<Rational>,
    float: Simplex<f64>,
    /// Tests switch the float stage off to check that it never changes a result.
    #[cfg(test)]
    pub(crate) float_stage: bool,
}

/// Sorts by row, sums repeated rows and drops zeros.
fn normalize(terms: &[(usize, Rational)]) -> Vec<(usize, Rational)> {
    let mut sorted = terms.to_vec();
    sorted.sort_by_key(|(row, _)| *row);
    let mut merged: Vec<(usize, Rational)> = Vec::with_capacity(sorted.len());
    for (row, coefficient) in sorted {
        match merged.last_mut() {
            Some((last, sum)) if *last == row => *sum += &coefficient,
            _ => merged.push((row, coefficient)),
        }
    }
    merged.retain(|(_, coefficient)| !coefficient.is_zero());
    merged
}

fn to_float(bound: Option<&Rational>) -> Option<f64> {
    bound.map(f64::from_rational)
}

impl Lp {
    /// A program with `rows` rows and no column.
    #[must_use]
    pub fn new(rows: usize) -> Lp {
        Lp {
            rows,
            exact: Simplex::new(rows),
            float: Simplex::new(rows),
            #[cfg(test)]
            float_stage: true,
        }
    }

    /// Adds a column with the given terms (row, coefficient) and bounds (`None`:
    /// unbounded on that side), and returns its index.
    ///
    /// # Panics
    ///
    /// If a term has a row `>= rows`.
    pub fn add_column(
        &mut self,
        terms: &[(usize, Rational)],
        lower: Option<Rational>,
        upper: Option<Rational>,
    ) -> usize {
        let terms = normalize(terms);
        assert!(terms.iter().all(|(row, _)| *row < self.rows), "row out of range");
        let float_terms = terms.iter().map(|(row, c)| (*row, f64::from_rational(c))).collect();
        self.float
            .add_column(float_terms, to_float(lower.as_ref()), to_float(upper.as_ref()));
        self.exact.add_column(terms, lower, upper) - self.rows
    }

    /// Sets the bounds of a row.
    ///
    /// # Panics
    ///
    /// If the row does not exist.
    pub fn set_row_bounds(&mut self, row: usize, lower: Option<Rational>, upper: Option<Rational>) {
        assert!(row < self.rows, "row out of range");
        self.set_var_bounds(row, lower, upper);
    }

    /// Sets the bounds of a column.
    pub fn set_bounds(&mut self, column: usize, lower: Option<Rational>, upper: Option<Rational>) {
        self.set_var_bounds(self.rows + column, lower, upper);
    }

    fn set_var_bounds(&mut self, var: usize, lower: Option<Rational>, upper: Option<Rational>) {
        if self.exact.lower[var] == lower && self.exact.upper[var] == upper {
            return;
        }
        self.float
            .set_bounds(var, to_float(lower.as_ref()), to_float(upper.as_ref()));
        self.exact.set_bounds(var, lower, upper);
    }

    /// Sets the objective to minimize: the terms (column, coefficient); the other
    /// columns get 0.
    pub fn set_objective(&mut self, terms: &[(usize, Rational)]) {
        let mut costs = vec![Rational::from_int(0); self.exact.cost.len()];
        for (column, c) in normalize(terms) {
            costs[self.rows + column] = c;
        }
        self.float.set_costs(costs.iter().map(f64::from_rational).collect());
        self.exact.set_costs(costs);
    }

    /// Solves from the basis of the previous solve.
    ///
    /// # Panics
    ///
    /// Never on a consistent program (the exact stage has no iteration limit).
    pub fn solve(&mut self) -> Status {
        let crossed = |(l, u): (&Option<Rational>, &Option<Rational>)| matches!((l, u), (Some(l), Some(u)) if l > u);
        if self.exact.lower.iter().zip(&self.exact.upper).any(crossed) {
            return Status::Infeasible;
        }
        #[cfg(test)]
        let float_stage = self.float_stage;
        #[cfg(not(test))]
        let float_stage = true;
        if float_stage {
            self.float.set_basis(self.exact.basis());
            self.float
                .run(Some(FLOAT_ITERATIONS_PER_VARIABLE * self.exact.cols.len()));
            self.exact.set_basis(self.float.basis());
        }
        match self.exact.run(None) {
            Outcome::Optimal => Status::Optimal(self.exact.objective()),
            Outcome::Infeasible => Status::Infeasible,
            Outcome::Unbounded => Status::Unbounded,
            Outcome::IterationLimit => unreachable!("the exact stage has no iteration limit"),
        }
    }

    /// The basis the next solve starts from.
    #[must_use]
    pub fn basis(&self) -> &Basis {
        self.exact.basis()
    }

    /// Starts the next solve from `basis` (taken from a program with the same
    /// rows and columns).
    pub fn set_basis(&mut self, basis: &Basis) {
        self.exact.set_basis(basis);
    }

    /// The value of a column after the last solve (meaningless after an
    /// infeasible one).
    #[must_use]
    pub fn value(&self, column: usize) -> &Rational {
        self.exact.value(self.rows + column)
    }

    /// Pivots made so far by the float stage and by the exact stage.
    #[must_use]
    pub fn pivots(&self) -> (usize, usize) {
        (self.float.pivots(), self.exact.pivots())
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::{Lp, Status};
    use crate::number::{Number, Rational};

    fn q(n: i64) -> Rational {
        Rational::from_int(n)
    }

    // A bound, for the `Option` parameters.
    #[allow(clippy::unnecessary_wraps)]
    fn some(n: i64) -> Option<Rational> {
        Some(q(n))
    }

    /// `min −x` s.t. `7x − y = 0`, `0 ≤ y ≤ 5`, `x` free.
    fn seven() -> Lp {
        let mut lp = Lp::new(1);
        lp.add_column(&[(0, q(7))], None, None);
        lp.add_column(&[(0, q(-1))], some(0), some(5));
        lp.set_objective(&[(0, q(-1))]);
        lp
    }

    #[test]
    fn exact_optimum_of_a_non_integer_vertex() {
        let mut lp = seven();
        assert_eq!(lp.solve(), Status::Optimal(Rational::new(-5, 7)));
        assert_eq!(*lp.value(0), Rational::new(5, 7));
        assert_eq!(*lp.value(1), q(5));
        assert!(lp.exact.certify());
    }

    #[test]
    fn infeasible_by_bounds_and_by_rows() {
        let mut lp = seven();
        lp.set_bounds(1, some(6), some(5));
        assert_eq!(lp.solve(), Status::Infeasible);
        // x <= 0 and the row x >= 1.
        let mut lp = Lp::new(1);
        lp.add_column(&[(0, q(1))], None, some(0));
        lp.set_row_bounds(0, some(1), None);
        lp.set_objective(&[]);
        assert_eq!(lp.solve(), Status::Infeasible);
    }

    #[test]
    fn unbounded() {
        let mut lp = Lp::new(1);
        lp.add_column(&[(0, q(1))], None, None);
        lp.add_column(&[(0, q(-1))], None, None);
        lp.set_objective(&[(0, q(-1))]);
        assert_eq!(lp.solve(), Status::Unbounded);
    }

    #[test]
    fn duplicate_and_zero_terms() {
        let mut lp = Lp::new(1);
        lp.add_column(&[(0, q(3)), (0, q(4)), (0, q(0))], None, None);
        lp.add_column(&[(0, q(-1))], some(0), some(5));
        lp.set_objective(&[(0, q(-1)), (0, q(0))]);
        assert_eq!(lp.solve(), Status::Optimal(Rational::new(-5, 7)));
    }

    #[test]
    fn warm_start_matches_a_fresh_program() {
        let mut lp = seven();
        lp.solve();
        for ub in [1, 3, 5, 0] {
            lp.set_bounds(1, some(0), some(ub));
            let status = lp.solve();
            let mut fresh = seven();
            fresh.set_bounds(1, some(0), some(ub));
            assert_eq!(status, fresh.solve());
            assert_eq!(status, Status::Optimal(Rational::new(-ub, 7)));
            assert!(lp.exact.certify());
        }
        // Infeasible, then back.
        lp.set_bounds(1, some(2), some(1));
        assert_eq!(lp.solve(), Status::Infeasible);
        lp.set_bounds(1, some(0), some(5));
        assert_eq!(lp.solve(), Status::Optimal(Rational::new(-5, 7)));
    }

    /// The exact stage reuses its reduced costs while the basis and the costs are
    /// unchanged: new bounds and new costs must still give the right answers.
    #[test]
    fn reduced_costs_follow_bounds_and_costs() {
        for float_stage in [true, false] {
            // min -x1 - 2 x2  s.t.  x1 + x2 <= 4, -10 <= x1 - x2 <= 10, 0 <= x <= 3.
            let mut lp = Lp::new(2);
            lp.float_stage = float_stage;
            lp.add_column(&[(0, q(1)), (1, q(1))], some(0), some(3));
            lp.add_column(&[(0, q(1)), (1, q(-1))], some(0), some(3));
            lp.set_row_bounds(0, None, some(4));
            lp.set_row_bounds(1, some(-10), some(10));
            lp.set_objective(&[(0, q(-1)), (1, q(-2))]);
            assert_eq!(lp.solve(), Status::Optimal(q(-7)));
            for step in 0..6 {
                match step {
                    0 => lp.set_bounds(1, some(0), some(1)),
                    1 => lp.set_row_bounds(0, None, some(2)),
                    2 => lp.set_objective(&[(0, q(2)), (1, q(-1))]),
                    3 => lp.set_bounds(0, some(1), some(3)),
                    4 => lp.set_objective(&[(0, q(-1)), (1, q(-2))]),
                    _ => lp.set_bounds(0, some(0), some(3)),
                }
                let status = lp.solve();
                let mut fresh = Lp::new(2);
                fresh.add_column(
                    &[(0, q(1)), (1, q(1))],
                    lp.exact.lower[2].clone(),
                    lp.exact.upper[2].clone(),
                );
                fresh.add_column(
                    &[(0, q(1)), (1, q(-1))],
                    lp.exact.lower[3].clone(),
                    lp.exact.upper[3].clone(),
                );
                fresh.set_row_bounds(0, lp.exact.lower[0].clone(), lp.exact.upper[0].clone());
                fresh.set_row_bounds(1, some(-10), some(10));
                let costs: Vec<_> = (0..2).map(|j| (j, lp.exact.cost[2 + j].clone())).collect();
                fresh.set_objective(&costs);
                assert_eq!(status, fresh.solve(), "step {step}");
                assert!(lp.exact.certify());
            }
        }
    }

    #[test]
    fn a_free_column_that_gains_a_bound_is_replaced() {
        // x is free and nonbasic at 0; x >= 2 must be seen by phase 1.
        let mut lp = Lp::new(1);
        lp.add_column(&[(0, q(1))], None, None);
        lp.set_row_bounds(0, None, None);
        lp.set_objective(&[(0, q(1))]);
        lp.set_bounds(0, some(-1), some(4));
        assert_eq!(lp.solve(), Status::Optimal(q(-1)));
        lp.set_bounds(0, some(2), some(4));
        assert_eq!(lp.solve(), Status::Optimal(q(2)));
    }

    /// Beale's example, on which Dantzig's rule alone cycles.
    #[test]
    fn beale_terminates() {
        for float_stage in [true, false] {
            let mut lp = Lp::new(2);
            lp.float_stage = float_stage;
            let columns = [
                (Rational::new(1, 4), Rational::new(1, 2), Rational::new(-3, 4), None),
                (q(-8), q(-12), q(20), None),
                (q(-1), Rational::new(-1, 2), Rational::new(-1, 2), some(1)),
                (q(9), q(3), q(6), None),
            ];
            let mut objective = Vec::new();
            for (j, (a, b, c, ub)) in columns.into_iter().enumerate() {
                lp.add_column(&[(0, a), (1, b)], some(0), ub);
                objective.push((j, c));
            }
            lp.set_row_bounds(0, None, some(0));
            lp.set_row_bounds(1, None, some(0));
            lp.set_objective(&objective);
            assert_eq!(lp.solve(), Status::Optimal(Rational::new(-5, 4)));
            assert!(lp.exact.certify());
        }
    }

    // ==========================================================================
    // Random programs against vertex enumeration
    // ==========================================================================

    /// Solves `a x = b` by Gauss-Jordan elimination.
    fn gauss(mut a: Vec<Vec<Rational>>, mut b: Vec<Rational>) -> Option<Vec<Rational>> {
        let n = b.len();
        for c in 0..n {
            let p = (c..n).find(|&r| !a[r][c].is_zero())?;
            a.swap(c, p);
            b.swap(c, p);
            let pivot = a[c][c].clone();
            for v in &mut a[c] {
                *v = v.clone() / &pivot;
            }
            b[c] = b[c].clone() / &pivot;
            for r in (0..n).filter(|&r| r != c) {
                let f = a[r][c].clone();
                if f.is_zero() {
                    continue;
                }
                let (pivot_row, pivot_b) = (a[c].clone(), b[c].clone());
                for (v, u) in a[r].iter_mut().zip(&pivot_row) {
                    *v -= &(f.clone() * u);
                }
                b[r] -= &(f * &pivot_b);
            }
        }
        Some(b)
    }

    #[derive(Clone, Debug)]
    struct Random {
        rows: usize,
        /// Dense columns of the structural variables.
        columns: Vec<Vec<i64>>,
        /// Bounds of the slacks, then of the columns.
        bounds: Vec<(i64, i64)>,
        costs: Vec<i64>,
    }

    impl Random {
        /// The minimum over all vertices, `None` if no vertex is feasible.
        fn brute_force(&self) -> Option<Rational> {
            let (m, n) = (self.rows, self.columns.len());
            let total = m + n;
            let column = |j: usize| -> Vec<Rational> {
                if j < m {
                    (0..m).map(|i| q(if i == j { -1 } else { 0 })).collect()
                } else {
                    self.columns[j - m].iter().map(|&a| q(a)).collect()
                }
            };
            let mut best: Option<Rational> = None;
            for mask in 0u32..(1 << total) {
                if mask.count_ones() as usize != m {
                    continue;
                }
                let basic: Vec<usize> = (0..total).filter(|j| mask >> j & 1 == 1).collect();
                let nonbasic: Vec<usize> = (0..total).filter(|j| mask >> j & 1 == 0).collect();
                for sides in 0u32..(1 << nonbasic.len()) {
                    let mut x = vec![q(0); total];
                    for (i, &j) in nonbasic.iter().enumerate() {
                        let (lo, hi) = self.bounds[j];
                        x[j] = q(if sides >> i & 1 == 0 { lo } else { hi });
                    }
                    let mut rhs = vec![q(0); m];
                    for &j in &nonbasic {
                        for (r, a) in column(j).iter().enumerate() {
                            rhs[r] -= &(a.clone() * &x[j]);
                        }
                    }
                    let columns: Vec<Vec<Rational>> = basic.iter().map(|&j| column(j)).collect();
                    let matrix: Vec<Vec<Rational>> =
                        (0..m).map(|r| columns.iter().map(|c| c[r].clone()).collect()).collect();
                    let Some(values) = gauss(matrix, rhs) else { continue };
                    for (&j, v) in basic.iter().zip(values) {
                        x[j] = v;
                    }
                    let feasible = (0..total).all(|j| x[j] >= q(self.bounds[j].0) && x[j] <= q(self.bounds[j].1));
                    if feasible {
                        let value = (0..n).fold(q(0), |acc, j| acc + &(q(self.costs[j]) * &x[m + j]));
                        if best.as_ref().is_none_or(|b| value < *b) {
                            best = Some(value);
                        }
                    }
                }
            }
            best
        }

        fn lp(&self, float_stage: bool) -> Lp {
            let mut lp = Lp::new(self.rows);
            lp.float_stage = float_stage;
            for (i, (lo, hi)) in self.bounds[..self.rows].iter().enumerate() {
                lp.set_row_bounds(i, some(*lo), some(*hi));
            }
            for (j, column) in self.columns.iter().enumerate() {
                let terms: Vec<(usize, Rational)> = column.iter().enumerate().map(|(i, &a)| (i, q(a))).collect();
                let (lo, hi) = self.bounds[self.rows + j];
                lp.add_column(&terms, some(lo), some(hi));
            }
            let objective: Vec<(usize, Rational)> = self.costs.iter().enumerate().map(|(j, &c)| (j, q(c))).collect();
            lp.set_objective(&objective);
            lp
        }
    }

    fn random_program() -> impl Strategy<Value = Random> {
        (1usize..=3, 1usize..=4).prop_flat_map(|(rows, n)| {
            let bound = (-5i64..=5, prop_oneof![Just(0i64), 0i64..=6]).prop_map(|(lo, width)| (lo, lo + width));
            (
                Just(rows),
                prop::collection::vec(prop::collection::vec(-3i64..=3, rows), n),
                prop::collection::vec(bound, rows + n),
                prop::collection::vec(-3i64..=3, n),
            )
                .prop_map(|(rows, columns, bounds, costs)| Random {
                    rows,
                    columns,
                    bounds,
                    costs,
                })
        })
    }

    proptest! {
        #[test]
        fn matches_vertex_enumeration(program in random_program()) {
            let expected = program.brute_force();
            for float_stage in [true, false] {
                let mut lp = program.lp(float_stage);
                match (&expected, lp.solve()) {
                    (None, Status::Infeasible) => {}
                    (Some(best), Status::Optimal(value)) => {
                        prop_assert_eq!(best, &value);
                        prop_assert!(lp.exact.certify());
                    }
                    (expected, status) => prop_assert!(false, "{expected:?} vs {status:?}"),
                }
            }
        }

        #[test]
        fn warm_starts_match_vertex_enumeration(
            first in random_program(),
            new_costs in prop::collection::vec(-3i64..=3, 4),
            shift in -2i64..=2,
        ) {
            let mut lp = first.lp(true);
            lp.solve();
            let mut second = Random { rows: first.rows, columns: first.columns.clone(), bounds: first.bounds.clone(), costs: first.costs.clone() };
            for (c, new) in second.costs.iter_mut().zip(&new_costs) {
                *c = *new;
            }
            for bound in &mut second.bounds[first.rows..] {
                *bound = (bound.0 + shift, bound.1 + shift);
            }
            lp.set_objective(&second.costs.iter().enumerate().map(|(j, &c)| (j, q(c))).collect::<Vec<_>>());
            for (j, (lo, hi)) in second.bounds[second.rows..].iter().enumerate() {
                lp.set_bounds(j, some(*lo), some(*hi));
            }
            match (second.brute_force(), lp.solve()) {
                (None, Status::Infeasible) => {}
                (Some(best), Status::Optimal(value)) => {
                    prop_assert_eq!(best, value);
                    prop_assert!(lp.exact.certify());
                }
                (expected, status) => prop_assert!(false, "{expected:?} vs {status:?}"),
            }
        }
    }
}
