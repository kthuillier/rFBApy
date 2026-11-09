//! Sparse LU factorization of a simplex basis, with product-form updates.
//!
//! `B = L·U` is found by right-looking elimination with Markowitz pivoting
//! (Markowitz 1957; Suhl & Suhl 1990): singleton columns and rows first, which
//! cost no fill-in, then the entry of least `(row count − 1)(column count − 1)`.
//! After a basis change, the factorization is kept and an eta matrix is
//! appended (product form of the inverse).

use crate::number::Number;

/// Columns searched for a Markowitz pivot (the search width trades time against
/// fill-in, never results).
const MARKOWITZ_COLUMNS: usize = 4;

/// Positions and rows left without a pivot: the basis is singular.
pub type Singular = (Vec<usize>, Vec<usize>);

/// A sparse vector: (index, value) pairs.
type Sparse<T> = Vec<(usize, T)>;

/// The factorization of an `m × m` basis. Rows are the constraints, positions
/// are the places of the basic variables.
#[derive(Clone, Debug)]
pub struct Factor<T> {
    m: usize,
    /// Step `k` pivots row `pivot_row[k]` and position `pivot_pos[k]`.
    pivot_row: Vec<usize>,
    pivot_pos: Vec<usize>,
    /// Multipliers of step `k`: (row, l).
    lower: Vec<Sparse<T>>,
    /// Pivot of step `k`.
    diag: Vec<T>,
    /// Row of step `k` right of the pivot: (position, u), positions of later steps.
    upper: Vec<Sparse<T>>,
    /// Since the factorization: (position `p`, column η of the eta matrix, `η_p` included).
    etas: Vec<(usize, Sparse<T>)>,
}

/// The active submatrix during the factorization.
struct Active<T> {
    /// Entries (position, value) of each row.
    rows: Vec<Sparse<T>>,
    /// Rows having an entry in each position.
    columns: Vec<Vec<usize>>,
    /// Rows and positions not pivoted yet.
    row_left: Vec<bool>,
    pos_left: Vec<bool>,
    /// Candidates (possibly stale, checked when popped) with one entry.
    column_singletons: Vec<usize>,
    row_singletons: Vec<usize>,
}

impl<T: Number> Active<T> {
    fn new(m: usize, columns: &[&[(usize, T)]]) -> Active<T> {
        let mut active = Active {
            rows: vec![Vec::new(); m],
            columns: vec![Vec::new(); m],
            row_left: vec![true; m],
            pos_left: vec![true; m],
            column_singletons: Vec::new(),
            row_singletons: Vec::new(),
        };
        for (p, column) in columns.iter().enumerate() {
            for (r, value) in *column {
                if !value.is_zero() {
                    active.rows[*r].push((p, value.clone()));
                    active.columns[p].push(*r);
                }
            }
        }
        for p in 0..m {
            if active.columns[p].len() == 1 {
                active.column_singletons.push(p);
            }
            if active.rows[p].len() == 1 {
                active.row_singletons.push(p);
            }
        }
        active
    }

    fn value(&self, r: usize, p: usize) -> &T {
        &self.rows[r]
            .iter()
            .find(|(q, _)| *q == p)
            .expect("entry of the active matrix")
            .1
    }

    /// The next pivot `(row, position)`: a singleton column, a singleton row,
    /// else the best Markowitz entry. `None`: no entry is left.
    fn pivot(&mut self) -> Option<(usize, usize)> {
        while let Some(p) = self.column_singletons.pop() {
            if self.pos_left[p] && self.columns[p].len() == 1 {
                return Some((self.columns[p][0], p));
            }
        }
        while let Some(r) = self.row_singletons.pop() {
            if self.row_left[r] && self.rows[r].len() == 1 {
                return Some((r, self.rows[r][0].0));
            }
        }
        self.markowitz()
    }

    fn markowitz(&self) -> Option<(usize, usize)> {
        // The few active columns with the fewest entries, by insertion.
        let mut candidates: Vec<(usize, usize)> = Vec::with_capacity(MARKOWITZ_COLUMNS + 1);
        for p in (0..self.columns.len()).filter(|&p| self.pos_left[p] && !self.columns[p].is_empty()) {
            let count = self.columns[p].len();
            let at = candidates.partition_point(|&(c, _)| c <= count);
            if at < MARKOWITZ_COLUMNS {
                candidates.insert(at, (count, p));
                candidates.truncate(MARKOWITZ_COLUMNS);
            }
        }
        let threshold = T::lu_threshold();
        let mut best: Option<(usize, usize, usize)> = None;
        for &(count, p) in &candidates {
            let largest = self.columns[p]
                .iter()
                .map(|&r| self.value(r, p).abs())
                .fold(T::zero(), |a, b| if b > a { b } else { a });
            let floor = threshold.clone() * &largest;
            for &r in &self.columns[p] {
                if self.value(r, p).abs() < floor {
                    continue;
                }
                let cost = (self.rows[r].len() - 1) * (count - 1);
                if best.is_none_or(|(c, _, _)| cost < c) {
                    best = Some((cost, r, p));
                }
            }
        }
        best.map(|(_, r, p)| (r, p))
    }

    /// Eliminates position `p` with pivot row `r`: returns the multipliers of
    /// the other rows and the pivot row without its pivot.
    fn eliminate(&mut self, r: usize, p: usize, scratch: &mut [usize]) -> (Sparse<T>, Sparse<T>, T) {
        let pivot_row = std::mem::take(&mut self.rows[r]);
        self.row_left[r] = false;
        self.pos_left[p] = false;
        for (q, _) in &pivot_row {
            self.columns[*q].retain(|&x| x != r);
        }
        let pivot = pivot_row.iter().find(|(q, _)| *q == p).expect("pivot entry").1.clone();
        let upper: Sparse<T> = pivot_row.into_iter().filter(|(q, _)| *q != p).collect();
        let mut lower: Sparse<T> = Vec::new();
        for r2 in std::mem::take(&mut self.columns[p]) {
            let mut row = std::mem::take(&mut self.rows[r2]);
            for (i, (q, _)) in row.iter().enumerate() {
                scratch[*q] = i;
            }
            let l = row[scratch[p]].1.clone() / &pivot;
            for (q, u) in &upper {
                let delta = l.clone() * u;
                if scratch[*q] == usize::MAX {
                    scratch[*q] = row.len();
                    row.push((*q, -delta));
                    self.columns[*q].push(r2);
                } else {
                    row[scratch[*q]].1 -= &delta;
                }
            }
            let mut kept = Vec::with_capacity(row.len());
            for (q, value) in row {
                scratch[q] = usize::MAX;
                if q == p {
                    continue;
                }
                if value.is_zero() {
                    self.columns[q].retain(|&x| x != r2);
                    if self.columns[q].len() == 1 {
                        self.column_singletons.push(q);
                    }
                } else {
                    kept.push((q, value));
                }
            }
            if kept.len() == 1 {
                self.row_singletons.push(r2);
            }
            self.rows[r2] = kept;
            lower.push((r2, l));
        }
        for (q, _) in &upper {
            if self.columns[*q].len() == 1 {
                self.column_singletons.push(*q);
            }
        }
        (lower, upper, pivot)
    }
}

impl<T: Number> Factor<T> {
    /// Factors the basis whose column at each position is `columns[position]`
    /// (sparse, by row).
    ///
    /// # Errors
    ///
    /// If the basis is singular: the positions and the rows left without a pivot.
    ///
    /// # Panics
    ///
    /// If `columns` does not hold `m` columns, or has a row `>= m`.
    pub fn new(m: usize, columns: &[&[(usize, T)]]) -> Result<Factor<T>, Singular> {
        assert_eq!(columns.len(), m, "a basis has m columns");
        let mut active = Active::new(m, columns);
        let mut scratch = vec![usize::MAX; m];
        let mut factor = Factor {
            m,
            pivot_row: Vec::with_capacity(m),
            pivot_pos: Vec::with_capacity(m),
            lower: Vec::with_capacity(m),
            diag: Vec::with_capacity(m),
            upper: Vec::with_capacity(m),
            etas: Vec::new(),
        };
        while factor.pivot_row.len() < m {
            let Some((r, p)) = active.pivot() else {
                let positions = (0..m).filter(|&p| active.pos_left[p]).collect();
                let rows = (0..m).filter(|&r| active.row_left[r]).collect();
                return Err((positions, rows));
            };
            let (lower, upper, pivot) = active.eliminate(r, p, &mut scratch);
            factor.pivot_row.push(r);
            factor.pivot_pos.push(p);
            factor.lower.push(lower);
            factor.diag.push(pivot);
            factor.upper.push(upper);
        }
        Ok(factor)
    }

    /// Solves `B x = a`: `a` is dense by row, the result dense by position.
    #[must_use]
    pub fn ftran(&self, a: &[T]) -> Vec<T> {
        let mut w = a.to_vec();
        for (pivot_row, lower) in self.pivot_row.iter().zip(&self.lower) {
            if w[*pivot_row].is_zero() {
                continue;
            }
            let t = w[*pivot_row].clone();
            for (r, l) in lower {
                w[*r] -= &(l.clone() * &t);
            }
        }
        let mut x = vec![T::zero(); self.m];
        for k in (0..self.m).rev() {
            let mut value = w[self.pivot_row[k]].clone();
            for (q, u) in &self.upper[k] {
                if !x[*q].is_zero() {
                    value -= &(u.clone() * &x[*q]);
                }
            }
            x[self.pivot_pos[k]] = value / &self.diag[k];
        }
        for (p, eta) in &self.etas {
            if x[*p].is_zero() {
                continue;
            }
            let t = x[*p].clone();
            for (i, e) in eta {
                if i == p {
                    x[*p] = e.clone() * &t;
                } else {
                    x[*i] += &(e.clone() * &t);
                }
            }
        }
        x
    }

    /// Solves `Bᵀ y = c`: `c` is dense by position, the result dense by row.
    #[must_use]
    pub fn btran(&self, c: &[T]) -> Vec<T> {
        let mut c = c.to_vec();
        for (p, eta) in self.etas.iter().rev() {
            let mut sum = T::zero();
            for (i, e) in eta {
                if !c[*i].is_zero() {
                    sum += &(e.clone() * &c[*i]);
                }
            }
            c[*p] = sum;
        }
        let mut w = vec![T::zero(); self.m];
        for k in 0..self.m {
            let t = c[self.pivot_pos[k]].clone() / &self.diag[k];
            if !t.is_zero() {
                for (q, u) in &self.upper[k] {
                    c[*q] -= &(u.clone() * &t);
                }
            }
            w[self.pivot_row[k]] = t;
        }
        for k in (0..self.m).rev() {
            for (r, l) in &self.lower[k] {
                if !w[*r].is_zero() {
                    let delta = l.clone() * &w[*r];
                    w[self.pivot_row[k]] -= &delta;
                }
            }
        }
        w
    }

    /// Records that position `p` now holds the column whose `ftran` is `d`
    /// (`d[p]` must not be zero).
    pub fn update(&mut self, p: usize, d: &[T]) {
        let mut eta: Sparse<T> = Vec::new();
        for (i, value) in d.iter().enumerate() {
            if i == p {
                eta.push((p, T::one() / &d[p]));
            } else if !value.is_zero() {
                eta.push((i, -(value.clone() / &d[p])));
            }
        }
        self.etas.push((p, eta));
    }

    /// Number of updates since the factorization.
    #[must_use]
    pub fn num_updates(&self) -> usize {
        self.etas.len()
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::Factor;
    use crate::number::Rational;

    type Matrix = Vec<Vec<i64>>;

    fn q(n: i64) -> Rational {
        Rational::from_int(n)
    }

    fn columns(matrix: &Matrix) -> Vec<Vec<(usize, Rational)>> {
        let m = matrix.len();
        (0..m)
            .map(|p| {
                (0..m)
                    .filter(|&r| matrix[r][p] != 0)
                    .map(|r| (r, q(matrix[r][p])))
                    .collect()
            })
            .collect()
    }

    fn factor(matrix: &Matrix) -> Option<Factor<Rational>> {
        let cols = columns(matrix);
        let refs: Vec<&[(usize, Rational)]> = cols.iter().map(Vec::as_slice).collect();
        Factor::new(matrix.len(), &refs).ok()
    }

    fn product(matrix: &Matrix, x: &[Rational]) -> Vec<Rational> {
        matrix
            .iter()
            .map(|row| row.iter().zip(x).fold(q(0), |acc, (a, x)| acc + &(q(*a) * x)))
            .collect()
    }

    fn transposed_product(matrix: &Matrix, y: &[Rational]) -> Vec<Rational> {
        (0..matrix.len())
            .map(|p| matrix.iter().zip(y).fold(q(0), |acc, (row, y)| acc + &(q(row[p]) * y)))
            .collect()
    }

    fn square() -> impl Strategy<Value = Matrix> {
        (1usize..=7).prop_flat_map(|m| prop::collection::vec(prop::collection::vec(-3i64..=3, m), m))
    }

    proptest! {
        #[test]
        fn solves_are_exact_also_after_updates(
            matrix in square(),
            rhs in prop::collection::vec(-5i64..=5, 7),
            new_columns in prop::collection::vec(prop::collection::vec(-3i64..=3, 7), 4),
            places in prop::collection::vec(0usize..7, 4),
        ) {
            let mut matrix = matrix;
            let m = matrix.len();
            let Some(mut f) = factor(&matrix) else { return Ok(()); };
            let a: Vec<Rational> = rhs.iter().take(m).map(|&v| q(v)).collect();
            for (new, place) in new_columns.iter().zip(&places) {
                prop_assert_eq!(product(&matrix, &f.ftran(&a)), a.clone());
                prop_assert_eq!(transposed_product(&matrix, &f.btran(&a)), a.clone());
                let column: Vec<Rational> = new.iter().take(m).map(|&v| q(v)).collect();
                let d = f.ftran(&column);
                let p = place % m;
                if d[p] == q(0) {
                    break;
                }
                f.update(p, &d);
                for (r, row) in matrix.iter_mut().enumerate() {
                    row[p] = new[r];
                }
            }
        }

        #[test]
        fn singular_matrices_are_reported(matrix in square()) {
            let cols = columns(&matrix);
            let refs: Vec<&[(usize, Rational)]> = cols.iter().map(Vec::as_slice).collect();
            let m = matrix.len();
            match Factor::new(m, &refs) {
                Ok(f) => {
                    // Non-singular: solving B x = B e gives e.
                    let e: Vec<Rational> = (0..m).map(|i| q(i64::try_from(i).unwrap() + 1)).collect();
                    prop_assert_eq!(f.ftran(&product(&matrix, &e)), e);
                }
                Err((positions, rows)) => {
                    prop_assert_eq!(positions.len(), rows.len());
                    prop_assert!(!positions.is_empty());
                }
            }
        }
    }

    #[test]
    fn empty_basis_is_fine() {
        let f = Factor::<Rational>::new(0, &[]).unwrap();
        assert_eq!(f.ftran(&[]), []);
        assert_eq!(f.btran(&[]), []);
    }

    #[test]
    fn float_factor_solves() {
        let cols: Vec<Vec<(usize, f64)>> = vec![vec![(0, 2.0), (1, 1.0)], vec![(0, 1.0), (1, 3.0)]];
        let refs: Vec<&[(usize, f64)]> = cols.iter().map(Vec::as_slice).collect();
        let f = Factor::new(2, &refs).unwrap();
        let x = f.ftran(&[5.0, 10.0]);
        assert!((x[0] - 1.0).abs() < 1e-12 && (x[1] - 3.0).abs() < 1e-12);
    }
}
