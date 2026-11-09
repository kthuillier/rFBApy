//! A bounded-variable revised primal simplex, generic over the number type.
//!
//! The problem is `min c·x` subject to `Σ_j a_ij x_j = s_i` for each row `i`,
//! with bounds on every `x_j` and every slack `s_i`. Variables `0..m` are the
//! slacks (variable `i` is the slack of row `i`, with column `−e_i`), variables
//! `m..` are the structural columns. The right-hand side of `[A −I] x = 0` is
//! zero, so with basic variables `x_B`: `x_B = −B⁻¹ N x_N`.
//!
//! Only the matrix and a sparse LU factorization of the basis are stored (see
//! [`crate::lu`]). Phase 1 minimizes the sum of infeasibilities and starts from
//! any basis, so one algorithm handles cold starts, bound changes and
//! objective changes. Dantzig pricing, with Bland's rule (Bland 1977) after a
//! run of degenerate steps longer than the number of rows, which makes the exact
//! version terminate. Shorter runs are normal: an FBA starts from a basis of
//! slacks fixed at zero, and needs up to one degenerate pivot per row to bring
//! structural columns in, where Bland's rule would crawl (53 000 pivots instead
//! of 2 500 on iML1515).

use crate::lu::Factor;
use crate::number::Number;

/// Updates of the factorization before it is recomputed. A speed knob only: the
/// results do not depend on it.
const REFACTOR: usize = 100;
/// Where a variable is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Place {
    /// In the basis, at this position.
    Basic(usize),
    /// Nonbasic, at its lower bound.
    AtLower,
    /// Nonbasic, at its upper bound.
    AtUpper,
    /// Nonbasic without bounds, at 0.
    Free,
}

/// A basis: the variable at each of the `m` positions, and the place of every variable.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Basis {
    /// The basic variable of each position.
    pub head: Vec<usize>,
    /// The place of each variable.
    pub place: Vec<Place>,
}

/// How a run ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// An optimal solution was found.
    Optimal,
    /// No point satisfies the bounds.
    Infeasible,
    /// The objective decreases forever.
    Unbounded,
    /// The limit on iterations was reached (or, in floating point, a step failed).
    IterationLimit,
}

/// How a basic variable stands with respect to its bounds.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Violation {
    Below,
    Above,
    Within,
}

/// The step of an iteration.
enum Step<T> {
    /// The entering variable goes to its other bound.
    Flip(T),
    /// The basic variable at `position` leaves, at its upper bound or its lower one.
    Pivot { theta: T, position: usize, to_upper: bool },
    /// Nothing limits the step.
    Unlimited,
}

/// A bounded-variable simplex over `T`.
#[derive(Clone, Debug)]
pub struct Simplex<T: Number> {
    m: usize,
    /// The columns of `[−I A]`, by row.
    pub(crate) cols: Vec<Vec<(usize, T)>>,
    pub(crate) lower: Vec<Option<T>>,
    pub(crate) upper: Vec<Option<T>>,
    pub(crate) cost: Vec<T>,
    pub(crate) basis: Basis,
    /// The factorization of the basis, if it is up to date with `basis.head`.
    factor: Option<Factor<T>>,
    pub(crate) x: Vec<T>,
    pivots: usize,
}

impl<T: Number> Simplex<T> {
    /// A problem with `m` rows and no column: every slack is basic with bounds `[0, 0]`.
    #[must_use]
    pub fn new(m: usize) -> Simplex<T> {
        Simplex {
            m,
            cols: (0..m).map(|i| vec![(i, -T::one())]).collect(),
            lower: vec![Some(T::zero()); m],
            upper: vec![Some(T::zero()); m],
            cost: vec![T::zero(); m],
            basis: Basis {
                head: (0..m).collect(),
                place: (0..m).map(Place::Basic).collect(),
            },
            factor: None,
            x: vec![T::zero(); m],
            pivots: 0,
        }
    }

    /// Adds a structural column (sorted by row, no duplicate, no zero) and
    /// returns its variable.
    pub fn add_column(&mut self, terms: Vec<(usize, T)>, lower: Option<T>, upper: Option<T>) -> usize {
        self.cols.push(terms);
        self.lower.push(lower);
        self.upper.push(upper);
        self.cost.push(T::zero());
        self.basis.place.push(Place::Free);
        self.x.push(T::zero());
        self.cols.len() - 1
    }

    /// Sets the bounds of a variable.
    pub fn set_bounds(&mut self, var: usize, lower: Option<T>, upper: Option<T>) {
        self.lower[var] = lower;
        self.upper[var] = upper;
    }

    /// Sets the cost of every variable.
    ///
    /// # Panics
    ///
    /// If there is not one cost per variable.
    pub fn set_costs(&mut self, costs: Vec<T>) {
        assert_eq!(costs.len(), self.cost.len(), "one cost per variable");
        self.cost = costs;
    }

    /// The basis.
    #[must_use]
    pub fn basis(&self) -> &Basis {
        &self.basis
    }

    /// Starts the next run from `basis`.
    pub fn set_basis(&mut self, basis: &Basis) {
        if basis.head != self.basis.head {
            self.factor = None;
        }
        self.basis = basis.clone();
    }

    /// The value of a variable at the end of the last run.
    #[must_use]
    pub fn value(&self, var: usize) -> &T {
        &self.x[var]
    }

    /// The objective at the end of the last run.
    #[must_use]
    pub fn objective(&self) -> T {
        let mut total = T::zero();
        for (c, x) in self.cost.iter().zip(&self.x) {
            if !c.is_zero() {
                total += &(c.clone() * x);
            }
        }
        total
    }

    /// Pivots made so far.
    #[must_use]
    pub fn pivots(&self) -> usize {
        self.pivots
    }

    // ==========================================================================
    // Run
    // ==========================================================================

    /// Minimizes the objective from the current basis: phase 1 while a basic
    /// variable violates a bound, then phase 2. At most `limit` iterations.
    ///
    /// # Panics
    ///
    /// If a bound pair is crossed (`lower > upper`), which callers rule out.
    pub fn run(&mut self, limit: Option<usize>) -> Outcome {
        self.place_nonbasic();
        if self.factor.is_none() {
            self.refactor();
        }
        self.compute_x();
        let mut degenerate = 0;
        let mut iterations = 0;
        loop {
            if limit.is_some_and(|l| iterations >= l) {
                return Outcome::IterationLimit;
            }
            iterations += 1;
            if self.factor.as_ref().expect("factor").num_updates() >= REFACTOR {
                self.refactor();
                self.compute_x();
            }
            let infeasibility = self.infeasibility_costs();
            let phase_one = infeasibility.is_some();
            let basic_costs =
                infeasibility.unwrap_or_else(|| self.basis.head.iter().map(|&j| self.cost[j].clone()).collect());
            let y = self.factor.as_ref().expect("factor").btran(&basic_costs);
            let bland = degenerate > self.m;
            let Some((j, up)) = self.entering(&y, phase_one, bland) else {
                return if phase_one {
                    Outcome::Infeasible
                } else {
                    Outcome::Optimal
                };
            };
            let mut column = vec![T::zero(); self.m];
            for (r, a) in &self.cols[j] {
                column[*r] = a.clone();
            }
            let d = self.factor.as_ref().expect("factor").ftran(&column);
            let step = self.ratio_test(j, up, &d, phase_one, bland);
            let moved = match step {
                Step::Unlimited if phase_one => return Outcome::IterationLimit,
                Step::Unlimited => return Outcome::Unbounded,
                Step::Flip(theta) => {
                    self.flip(j, up, &theta, &d);
                    theta
                }
                Step::Pivot {
                    theta,
                    position,
                    to_upper,
                } => {
                    self.pivot(j, up, &theta, &d, position, to_upper);
                    theta
                }
            };
            if moved.is_zero() {
                degenerate += 1;
            } else {
                degenerate = 0;
            }
        }
    }

    // ==========================================================================
    // Basis and values
    // ==========================================================================

    /// Puts every nonbasic variable on a bound it has: it stays on its side if
    /// that bound exists, else goes to the other side, else is free. A free
    /// variable that has a bound goes to it (the lower one if it is nearer to
    /// zero), so no nonbasic variable sits outside its bounds, where phase 1
    /// would not see it.
    fn place_nonbasic(&mut self) {
        for j in 0..self.cols.len() {
            let (lower, upper) = (&self.lower[j], &self.upper[j]);
            self.basis.place[j] = match self.basis.place[j] {
                Place::Basic(p) => Place::Basic(p),
                Place::AtLower if lower.is_some() => Place::AtLower,
                Place::AtUpper if upper.is_some() => Place::AtUpper,
                _ => match (lower, upper) {
                    (Some(l), Some(u)) if l.abs() <= u.abs() => Place::AtLower,
                    (Some(_), None) => Place::AtLower,
                    (_, Some(_)) => Place::AtUpper,
                    (None, None) => Place::Free,
                },
            };
        }
    }

    /// The value of a nonbasic variable.
    fn nonbasic_value(&self, j: usize) -> Option<T> {
        match self.basis.place[j] {
            Place::Basic(_) => None,
            Place::AtLower => Some(self.lower[j].clone().expect("lower bound")),
            Place::AtUpper => Some(self.upper[j].clone().expect("upper bound")),
            Place::Free => Some(T::zero()),
        }
    }

    /// Factors the basis, replacing the variables of a singular one by slacks.
    fn refactor(&mut self) {
        loop {
            let columns: Vec<&[(usize, T)]> = self.basis.head.iter().map(|&j| self.cols[j].as_slice()).collect();
            match Factor::new(self.m, &columns) {
                Ok(factor) => {
                    self.factor = Some(factor);
                    return;
                }
                Err((positions, rows)) => {
                    // An unpivoted row has its slack nonbasic: bring it in.
                    for (p, r) in positions.into_iter().zip(rows) {
                        let leaving = self.basis.head[p];
                        self.basis.head[p] = r;
                        self.basis.place[r] = Place::Basic(p);
                        self.basis.place[leaving] = Place::Free;
                    }
                    self.place_nonbasic();
                }
            }
        }
    }

    /// Recomputes every value from the nonbasic ones: `x_B = −B⁻¹ N x_N`.
    fn compute_x(&mut self) {
        let mut rhs = vec![T::zero(); self.m];
        for j in 0..self.cols.len() {
            let Some(value) = self.nonbasic_value(j) else { continue };
            if !value.is_zero() {
                for (r, a) in &self.cols[j] {
                    rhs[*r] -= &(a.clone() * &value);
                }
            }
            self.x[j] = value;
        }
        let basic = self.factor.as_ref().expect("factor").ftran(&rhs);
        for (p, value) in basic.into_iter().enumerate() {
            self.x[self.basis.head[p]] = value;
        }
    }

    // ==========================================================================
    // Iteration
    // ==========================================================================

    /// The allowed violation of a bound.
    fn slack(bound: &T) -> T {
        let size = bound.abs();
        let scale = if size > T::one() { size } else { T::one() };
        T::feasibility_tol() * &scale
    }

    fn violation(&self, j: usize) -> Violation {
        let x = &self.x[j];
        if let Some(lower) = &self.lower[j]
            && *x < lower.clone() - &Self::slack(lower)
        {
            return Violation::Below;
        }
        if let Some(upper) = &self.upper[j]
            && *x > upper.clone() + &Self::slack(upper)
        {
            return Violation::Above;
        }
        Violation::Within
    }

    /// Phase 1 costs of the basic variables (−1 below the lower bound, +1 above
    /// the upper one), or `None` if the basic variables satisfy their bounds.
    fn infeasibility_costs(&self) -> Option<Vec<T>> {
        let mut any = false;
        let costs = self
            .basis
            .head
            .iter()
            .map(|&j| match self.violation(j) {
                Violation::Below => {
                    any = true;
                    -T::one()
                }
                Violation::Above => {
                    any = true;
                    T::one()
                }
                Violation::Within => T::zero(),
            })
            .collect();
        any.then_some(costs)
    }

    /// The entering variable and whether it increases: the largest reduced cost
    /// (Dantzig), or the first eligible variable under Bland's rule.
    fn entering(&self, y: &[T], phase_one: bool, bland: bool) -> Option<(usize, bool)> {
        let tolerance = T::optimality_tol();
        let negative_tolerance = -tolerance.clone();
        let mut best: Option<(T, usize, bool)> = None;
        for j in 0..self.cols.len() {
            let place = self.basis.place[j];
            if matches!(place, Place::Basic(_)) {
                continue;
            }
            if let (Some(l), Some(u)) = (&self.lower[j], &self.upper[j])
                && l == u
            {
                continue;
            }
            let mut d = if phase_one { T::zero() } else { self.cost[j].clone() };
            for (r, a) in &self.cols[j] {
                if !y[*r].is_zero() {
                    d -= &(a.clone() * &y[*r]);
                }
            }
            let can_increase = matches!(place, Place::AtLower | Place::Free);
            let can_decrease = matches!(place, Place::AtUpper | Place::Free);
            let up = if d < negative_tolerance && can_increase {
                true
            } else if d > tolerance && can_decrease {
                false
            } else {
                continue;
            };
            if bland {
                return Some((j, up));
            }
            let size = d.abs();
            if best.as_ref().is_none_or(|(b, _, _)| size > *b) {
                best = Some((size, j, up));
            }
        }
        best.map(|(_, j, up)| (j, up))
    }

    /// The step along the entering variable `j` (`d = B⁻¹ a_j`): the basic
    /// variables move by `−σθd`. A feasible basic variable stops at the bound it
    /// moves toward; in phase 1 an infeasible one stops where it becomes feasible,
    /// and has no limit moving away.
    fn ratio_test(&self, j: usize, up: bool, d: &[T], phase_one: bool, bland: bool) -> Step<T> {
        let pivot_tolerance = T::pivot_tol();
        let mut best: Option<(T, usize, bool)> = None;
        for (p, dp) in d.iter().enumerate() {
            if dp.abs() <= pivot_tolerance {
                continue;
            }
            let k = self.basis.head[p];
            let increasing = (*dp < T::zero()) == up;
            let state = if phase_one {
                self.violation(k)
            } else {
                Violation::Within
            };
            let target = match (state, increasing) {
                (Violation::Below, true) | (Violation::Within, false) => self.lower[k].as_ref().map(|b| (b, false)),
                (Violation::Above, false) | (Violation::Within, true) => self.upper[k].as_ref().map(|b| (b, true)),
                (Violation::Below, false) | (Violation::Above, true) => None,
            };
            let Some((bound, to_upper)) = target else { continue };
            let distance = if increasing {
                bound.clone() - &self.x[k]
            } else {
                self.x[k].clone() - bound
            };
            let distance = if distance < T::zero() { T::zero() } else { distance };
            let theta = distance / &dp.abs();
            let better = match &best {
                None => true,
                Some((t, q, _)) => {
                    theta < *t
                        || (theta == *t
                            && if bland {
                                k < self.basis.head[*q]
                            } else {
                                dp.abs() > d[*q].abs()
                            })
                }
            };
            if better {
                best = Some((theta, p, to_upper));
            }
        }
        let range = match (&self.lower[j], &self.upper[j]) {
            (Some(l), Some(u)) => Some(u.clone() - l),
            _ => None,
        };
        match (best, range) {
            (None, None) => Step::Unlimited,
            (None, Some(r)) => Step::Flip(r),
            (Some((theta, _, _)), Some(r)) if r <= theta => Step::Flip(r),
            (Some((theta, position, to_upper)), _) => Step::Pivot {
                theta,
                position,
                to_upper,
            },
        }
    }

    /// Moves the basic variables by `−σθd`.
    fn move_basics(&mut self, up: bool, theta: &T, d: &[T]) {
        for (p, dp) in d.iter().enumerate() {
            if dp.is_zero() {
                continue;
            }
            let delta = theta.clone() * dp;
            let k = self.basis.head[p];
            if up {
                self.x[k] -= &delta;
            } else {
                self.x[k] += &delta;
            }
        }
    }

    /// The entering variable goes to its other bound.
    fn flip(&mut self, j: usize, up: bool, theta: &T, d: &[T]) {
        self.move_basics(up, theta, d);
        if up {
            self.x[j] = self.upper[j].clone().expect("upper bound");
            self.basis.place[j] = Place::AtUpper;
        } else {
            self.x[j] = self.lower[j].clone().expect("lower bound");
            self.basis.place[j] = Place::AtLower;
        }
    }

    /// The entering variable `j` replaces the basic variable at `position`,
    /// which leaves at a bound.
    fn pivot(&mut self, j: usize, up: bool, theta: &T, d: &[T], position: usize, to_upper: bool) {
        self.move_basics(up, theta, d);
        if up {
            self.x[j] += theta;
        } else {
            self.x[j] -= theta;
        }
        let leaving = self.basis.head[position];
        let fixed = self.lower[leaving] == self.upper[leaving];
        if to_upper {
            self.x[leaving] = self.upper[leaving].clone().expect("upper bound");
            self.basis.place[leaving] = if fixed { Place::AtLower } else { Place::AtUpper };
        } else {
            self.x[leaving] = self.lower[leaving].clone().expect("lower bound");
            self.basis.place[leaving] = Place::AtLower;
        }
        self.basis.head[position] = j;
        self.basis.place[j] = Place::Basic(position);
        self.factor.as_mut().expect("factor").update(position, d);
        self.pivots += 1;
    }
}

#[cfg(test)]
impl Simplex<crate::number::Rational> {
    /// Checks the solution of an optimal run from scratch: the rows hold, every
    /// variable is within its bounds, nonbasic variables sit on their bounds,
    /// and every reduced cost has the sign of its place.
    pub(crate) fn certify(&self) -> bool {
        use crate::number::Rational;
        let zero = Rational::from_int(0);
        let mut residual = vec![zero.clone(); self.m];
        for (col, x) in self.cols.iter().zip(&self.x) {
            for (r, a) in col {
                residual[*r] += &(a.clone() * x);
            }
        }
        let rows = residual.iter().all(|r| *r == zero);
        let bounds = (0..self.cols.len()).all(|j| {
            self.lower[j].as_ref().is_none_or(|l| self.x[j] >= *l)
                && self.upper[j].as_ref().is_none_or(|u| self.x[j] <= *u)
        });
        let nonbasic = (0..self.cols.len()).all(|j| self.nonbasic_value(j).is_none_or(|v| v == self.x[j]));
        let columns: Vec<&[(usize, Rational)]> = self.basis.head.iter().map(|&j| self.cols[j].as_slice()).collect();
        let Ok(factor) = Factor::new(self.m, &columns) else {
            return false;
        };
        let basic_costs: Vec<Rational> = self.basis.head.iter().map(|&j| self.cost[j].clone()).collect();
        let y = factor.btran(&basic_costs);
        let signs = (0..self.cols.len()).all(|j| {
            let mut d = self.cost[j].clone();
            for (r, a) in &self.cols[j] {
                d -= &(a.clone() * &y[*r]);
            }
            let fixed = self.lower[j].is_some() && self.lower[j] == self.upper[j];
            match self.basis.place[j] {
                Place::Basic(_) => d == zero,
                _ if fixed => true,
                Place::AtLower => d >= zero,
                Place::AtUpper => d <= zero,
                Place::Free => d == zero,
            }
        });
        rows && bounds && nonbasic && signs
    }
}
