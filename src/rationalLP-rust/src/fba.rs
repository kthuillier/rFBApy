//! Flux balance analysis and parsimonious FBA, exact.
//!
//! Each reaction `r` is split into two non-negative columns, `v_r = v⁺_r − v⁻_r`,
//! so that the sum of absolute fluxes of pFBA is a plain linear objective
//! `Σ (v⁺_r + v⁻_r)` (at an optimum one of the two is zero), with no extra row.
//! The optimum of the first step is fixed for the second one by bounds on the
//! objective columns, using its exact value: rounding it to a float could put it
//! above the true optimum and make the second step infeasible.

use crate::lp::{Lp, Status};
use crate::number::Rational;

/// Optional lower and upper bound (`None`: infinite).
pub type Bounds = (Option<Rational>, Option<Rational>);

/// An FBA problem `S v = 0`, `lb ≤ v ≤ ub`, maximizing one reaction.
#[derive(Clone, Debug)]
pub struct Fba {
    lp: Lp,
    reactions: usize,
    objective: usize,
    /// Bounds of each reaction. Column `r` of the LP is `v⁺_r`, column `n + r` is `v⁻_r`.
    bounds: Vec<Bounds>,
}

fn positive_part(x: &Rational) -> Rational {
    let zero = Rational::from_int(0);
    if *x > zero { x.clone() } else { zero }
}

/// The bounds of `v⁺` and of `v⁻` that make `v⁺ − v⁻` range over `[lb, ub]`
/// (an empty range stays empty).
fn split((lb, ub): &Bounds) -> (Bounds, Bounds) {
    let zero = Rational::from_int(0);
    let plus = (
        Some(lb.as_ref().map_or(zero.clone(), positive_part)),
        ub.as_ref().map(positive_part),
    );
    let minus = (
        Some(ub.as_ref().map_or(zero, |u| positive_part(&-u.clone()))),
        lb.as_ref().map(|l| positive_part(&-l.clone())),
    );
    (plus, minus)
}

impl Fba {
    /// A problem with `rows` metabolites. `stoichiometry` lists (metabolite,
    /// reaction, coefficient), summing repeated pairs; `bounds` has one pair per
    /// reaction.
    ///
    /// # Panics
    ///
    /// If an index is out of range.
    #[must_use]
    pub fn new(rows: usize, stoichiometry: &[(usize, usize, Rational)], bounds: Vec<Bounds>, objective: usize) -> Fba {
        let n = bounds.len();
        assert!(objective < n, "objective out of range");
        let mut terms: Vec<Vec<(usize, Rational)>> = vec![Vec::new(); n];
        for (row, reaction, coefficient) in stoichiometry {
            assert!(*row < rows && *reaction < n, "index out of range");
            terms[*reaction].push((*row, coefficient.clone()));
        }
        let mut lp = Lp::new(rows);
        let splits: Vec<(Bounds, Bounds)> = bounds.iter().map(split).collect();
        for (column, (plus, _)) in terms.iter().zip(&splits) {
            lp.add_column(column, plus.0.clone(), plus.1.clone());
        }
        for (column, (_, minus)) in terms.iter().zip(&splits) {
            let negated: Vec<(usize, Rational)> = column.iter().map(|(row, c)| (*row, -c.clone())).collect();
            lp.add_column(&negated, minus.0.clone(), minus.1.clone());
        }
        Fba {
            lp,
            reactions: n,
            objective,
            bounds,
        }
    }

    /// Maximizes the objective reaction, then, with `pfba`, minimizes the total
    /// absolute flux at that optimum. `overrides` (reaction, lower, upper)
    /// replace bounds for this solve only.
    ///
    /// Returns the status (`Optimal` holds the maximum of the objective reaction)
    /// and the flux of every reaction: empty if infeasible, the last point
    /// reached if unbounded (then without pFBA).
    ///
    /// # Panics
    ///
    /// If an override names a reaction that does not exist.
    pub fn solve(
        &mut self,
        overrides: &[(usize, Option<Rational>, Option<Rational>)],
        pfba: bool,
    ) -> (Status, Vec<Rational>) {
        let n = self.reactions;
        let mut effective: Vec<(&Option<Rational>, &Option<Rational>)> =
            self.bounds.iter().map(|(l, u)| (l, u)).collect();
        for (reaction, lb, ub) in overrides {
            effective[*reaction] = (lb, ub);
        }
        for (r, (lb, ub)) in effective.into_iter().enumerate() {
            let (plus, minus) = split(&(lb.clone(), ub.clone()));
            self.lp.set_bounds(r, plus.0, plus.1);
            self.lp.set_bounds(n + r, minus.0, minus.1);
        }
        let one = Rational::from_int(1);
        self.lp
            .set_objective(&[(self.objective, -one.clone()), (n + self.objective, one.clone())]);
        let optimum = match self.lp.solve() {
            Status::Infeasible => return (Status::Infeasible, Vec::new()),
            Status::Unbounded => return (Status::Unbounded, self.fluxes()),
            Status::Optimal(minimum) => -minimum,
        };
        let first = self.fluxes();
        if !pfba {
            return (Status::Optimal(optimum), first);
        }
        // Fix the objective reaction at its exact optimum.
        let zero = Rational::from_int(0);
        let (plus, minus) = if optimum >= zero {
            (
                (Some(optimum.clone()), Some(optimum.clone())),
                (Some(zero.clone()), Some(zero)),
            )
        } else {
            (
                (Some(zero.clone()), Some(zero)),
                (Some(-optimum.clone()), Some(-optimum.clone())),
            )
        };
        self.lp.set_bounds(self.objective, plus.0, plus.1);
        self.lp.set_bounds(n + self.objective, minus.0, minus.1);
        let total: Vec<(usize, Rational)> = (0..2 * n).map(|column| (column, one.clone())).collect();
        self.lp.set_objective(&total);
        let second = self.lp.solve();
        debug_assert!(
            matches!(second, Status::Optimal(_)),
            "the second step is feasible and bounded below"
        );
        let fluxes = if matches!(second, Status::Optimal(_)) {
            self.fluxes()
        } else {
            first
        };
        (Status::Optimal(optimum), fluxes)
    }

    /// `v⁺ − v⁻` of every reaction at the last point.
    fn fluxes(&self) -> Vec<Rational> {
        (0..self.reactions)
            .map(|r| self.lp.value(r).clone() - self.lp.value(self.reactions + r))
            .collect()
    }

    /// Pivots made so far by the float stage and by the exact stage.
    #[must_use]
    pub fn pivots(&self) -> (usize, usize) {
        self.lp.pivots()
    }
}

#[cfg(test)]
mod tests {
    use super::{Bounds, Fba, split};
    use crate::lp::Status;
    use crate::number::Rational;

    fn q(n: i64) -> Rational {
        Rational::from_int(n)
    }

    fn b(lb: Option<i64>, ub: Option<i64>) -> Bounds {
        (lb.map(q), ub.map(q))
    }

    #[test]
    fn split_bounds() {
        let cases = [
            // (lb, ub) -> (v+ lb, v+ ub, v- lb, v- ub)
            ((Some(0), Some(10)), (0, Some(10), 0, Some(0))),
            ((Some(-5), Some(3)), (0, Some(3), 0, Some(5))),
            ((Some(2), Some(4)), (2, Some(4), 0, Some(0))),
            ((Some(-4), Some(-1)), (0, Some(0), 1, Some(4))),
            ((None, None), (0, None, 0, None)),
            ((None, Some(-2)), (0, Some(0), 2, None)),
            ((Some(3), None), (3, None, 0, Some(0))),
        ];
        for ((lb, ub), (pl, pu, ml, mu)) in cases {
            let (plus, minus) = split(&b(lb, ub));
            assert_eq!(plus, (Some(q(pl)), pu.map(q)), "{lb:?} {ub:?}");
            assert_eq!(minus, (Some(q(ml)), mu.map(q)), "{lb:?} {ub:?}");
        }
        // An empty range stays empty.
        for (lb, ub) in [(2, 1), (-1, -2), (1, -2), (3, 0)] {
            let ((pl, pu), (ml, mu)) = split(&b(Some(lb), Some(ub)));
            assert!(pl > pu || ml > mu, "{lb} {ub}");
        }
    }

    /// `s1: -> A (r0, uptake <= 10)`, `r1: A -> B` (reversible), `r2: B ->` (objective).
    fn chain() -> Fba {
        let stoichiometry = [(0, 0, q(1)), (0, 1, q(-1)), (1, 1, q(1)), (1, 2, q(-1))];
        Fba::new(
            2,
            &stoichiometry,
            vec![b(Some(0), Some(10)), b(Some(-100), Some(100)), b(Some(0), None)],
            2,
        )
    }

    #[test]
    fn chain_fba_and_pfba() {
        let mut fba = chain();
        for pfba in [false, true] {
            let (status, fluxes) = fba.solve(&[], pfba);
            assert_eq!(status, Status::Optimal(q(10)));
            assert_eq!(fluxes, vec![q(10), q(10), q(10)]);
        }
    }

    #[test]
    fn overrides_are_temporary() {
        let mut fba = chain();
        let (status, _) = fba.solve(&[(0, Some(q(0)), Some(q(4)))], true);
        assert_eq!(status, Status::Optimal(q(4)));
        let (status, _) = fba.solve(&[(1, Some(q(0)), Some(q(0)))], true);
        assert_eq!(status, Status::Optimal(q(0)));
        let (status, fluxes) = fba.solve(&[], true);
        assert_eq!(status, Status::Optimal(q(10)));
        assert_eq!(fluxes[2], q(10));
    }

    #[test]
    fn infeasible_and_unbounded() {
        let mut fba = chain();
        // The objective reaction cannot be satisfied by an empty range.
        let (status, fluxes) = fba.solve(&[(1, Some(q(2)), Some(q(1)))], true);
        assert_eq!((status, fluxes), (Status::Infeasible, Vec::new()));
        // Lower bound above what the network can carry.
        let (status, _) = fba.solve(&[(2, Some(q(11)), None)], false);
        assert_eq!(status, Status::Infeasible);
        // The chain is capped by `r1 <= 100` ...
        let (status, _) = fba.solve(&[(0, Some(q(0)), None)], true);
        assert_eq!(status, Status::Optimal(q(100)));
        // ... until that is lifted too.
        let (status, _) = fba.solve(&[(0, Some(q(0)), None), (1, Some(q(0)), None)], true);
        assert_eq!(status, Status::Unbounded);
    }

    /// A negative optimum: the objective reaction is forced backwards, and pFBA
    /// still fixes it exactly.
    #[test]
    fn negative_optimum() {
        // r0: <-> A (-5..=5), r1: A -> (objective, -3..=-1): r0 = r1.
        let stoichiometry = [(0, 0, q(1)), (0, 1, q(-1))];
        let mut fba = Fba::new(1, &stoichiometry, vec![b(Some(-5), Some(5)), b(Some(-3), Some(-1))], 1);
        let (status, fluxes) = fba.solve(&[], true);
        assert_eq!(status, Status::Optimal(q(-1)));
        assert_eq!(fluxes, vec![q(-1), q(-1)]);
    }

    /// pFBA minimizes the total absolute flux: the short path wins over the long one.
    #[test]
    fn pfba_prefers_the_short_path() {
        // r0: -> A (<= 1); r1: A -> B; r2: B -> C; r3: C ->  (objective);
        // r4: A -> C (direct).
        let stoichiometry = [
            (0, 0, q(1)),
            (0, 1, q(-1)),
            (0, 4, q(-1)),
            (1, 1, q(1)),
            (1, 2, q(-1)),
            (2, 2, q(1)),
            (2, 3, q(-1)),
            (2, 4, q(1)),
        ];
        let bounds = vec![
            b(Some(0), Some(1)),
            b(Some(0), None),
            b(Some(0), None),
            b(Some(0), None),
            b(Some(0), None),
        ];
        let mut fba = Fba::new(3, &stoichiometry, bounds, 3);
        let (status, fluxes) = fba.solve(&[], true);
        assert_eq!(status, Status::Optimal(q(1)));
        assert_eq!(fluxes, vec![q(1), q(0), q(0), q(1), q(1)]);
    }

    /// The pitfall of float optima: `max x` with `7x = y <= 5` is `5/7`, whose
    /// float is above it.
    #[test]
    fn pfba_with_a_non_dyadic_optimum() {
        let stoichiometry = [(0, 0, q(7)), (0, 1, q(-1))];
        let mut fba = Fba::new(1, &stoichiometry, vec![b(None, None), b(Some(0), Some(5))], 0);
        let (status, fluxes) = fba.solve(&[], true);
        assert_eq!(status, Status::Optimal(Rational::new(5, 7)));
        assert_eq!(fluxes, vec![Rational::new(5, 7), q(5)]);
    }
}
