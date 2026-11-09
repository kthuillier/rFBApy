//! FBA and pFBA on the toy network of rFBApy (`example/data/metabolic_network.sbml`),
//! against GLPK's optima and exact checks of the fluxes.

// The reference values are copied digit for digit from GLPK's output.
#![allow(clippy::unreadable_literal)]

use rational_lp_solver::fba::Fba;
use rational_lp_solver::lp::Status;
use rational_lp_solver::number::Rational;

/// GLPK's optimum of the objective reaction (a float): `387/110`.
const GLPK_OPTIMUM: f64 = 3.518181818181818;
/// GLPK's minimal total absolute flux at that optimum (a float): `2481/22`.
const GLPK_TOTAL_FLUX: f64 = 112.77272727272727;
/// GLPK's primal and dual feasibility tolerance (`tol_bnd`, `tol_dj`): the accuracy GLPK claims.
const GLPK_FEASIBILITY_TOL: f64 = 1e-7;

const METABOLITES: usize = 11;
const OBJECTIVE: usize = 8;
const KNOCKED_OUT: usize = 6;
/// (metabolite, reaction, coefficient) of the 20 reactions
/// R2a R3 R4 Tc2 R2b Tc1 R6 R7 Growth Rres Td Te Tf Th To2 R8a R8b R5b R1 R5a.
const STOICHIOMETRY: [(usize, usize, f64); 49] = [
    (2, 0, -1.0),
    (9, 0, 2.0),
    (3, 0, 1.0),
    (1, 0, 2.0),
    (2, 1, -1.0),
    (6, 1, 1.0),
    (3, 2, -1.0),
    (7, 2, 1.0),
    (0, 3, 1.0),
    (3, 4, -1.0),
    (9, 4, -2.0),
    (1, 4, -2.0),
    (2, 4, 1.0),
    (0, 5, 1.0),
    (3, 6, -1.0),
    (4, 6, 3.0),
    (1, 6, 2.0),
    (3, 7, -1.0),
    (9, 7, -4.0),
    (5, 7, 3.0),
    (8, 8, -1.0),
    (3, 8, -1.0),
    (1, 8, -10.0),
    (6, 8, -1.0),
    (9, 9, -1.0),
    (10, 9, -1.0),
    (1, 9, 1.0),
    (4, 10, -1.0),
    (5, 11, -1.0),
    (6, 12, 1.0),
    (8, 13, 1.0),
    (10, 14, 1.0),
    (9, 15, -2.0),
    (1, 15, -1.0),
    (7, 15, -1.0),
    (8, 15, 1.0),
    (8, 16, -1.0),
    (9, 16, 2.0),
    (1, 16, 1.0),
    (7, 16, 1.0),
    (7, 17, -1.0),
    (3, 17, 0.8),
    (9, 17, 2.0),
    (0, 18, -1.0),
    (1, 18, -1.0),
    (2, 18, 1.0),
    (7, 19, -1.0),
    (3, 19, 0.8),
    (9, 19, 2.0),
];
/// Upper bounds (the lower bounds are all 0).
const UPPER: [f64; 20] = [
    9999.0, 9999.0, 9999.0, 10.5, 9999.0, 10.5, 9999.0, 9999.0, 9999.0, 9999.0, 12.0, 12.0, 5.0, 5.0, 15.0, 9999.0,
    9999.0, 9999.0, 9999.0, 9999.0,
];

fn rational(x: f64) -> Rational {
    Rational::from_f64(x).unwrap()
}

fn toy() -> Fba {
    let stoichiometry: Vec<(usize, usize, Rational)> =
        STOICHIOMETRY.iter().map(|&(m, r, c)| (m, r, rational(c))).collect();
    let bounds = UPPER
        .iter()
        .map(|&u| (Some(rational(0.0)), Some(rational(u))))
        .collect();
    Fba::new(METABOLITES, &stoichiometry, bounds, OBJECTIVE)
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= GLPK_FEASIBILITY_TOL * b.abs().max(1.0)
}

/// Every metabolite is balanced and every flux within its bounds, exactly.
fn assert_feasible(fluxes: &[Rational]) {
    let mut balance = vec![rational(0.0); METABOLITES];
    for &(m, r, c) in &STOICHIOMETRY {
        balance[m] += &(rational(c) * &fluxes[r]);
    }
    assert!(balance.iter().all(|b| *b == rational(0.0)), "{balance:?}");
    for (flux, &u) in fluxes.iter().zip(&UPPER) {
        assert!(
            *flux >= rational(0.0) && *flux <= rational(u),
            "{flux} outside [0, {u}]"
        );
    }
}

fn total_flux(fluxes: &[Rational]) -> Rational {
    fluxes.iter().fold(rational(0.0), |acc, v| {
        acc + &if *v < rational(0.0) { -v.clone() } else { v.clone() }
    })
}

#[test]
fn fba_optimum_is_glpks() {
    let (status, fluxes) = toy().solve(&[], false);
    assert_eq!(status, Status::Optimal(Rational::new(387, 110)));
    let Status::Optimal(optimum) = status else {
        unreachable!()
    };
    assert!(close(optimum.to_f64(), GLPK_OPTIMUM));
    assert_feasible(&fluxes);
    assert_eq!(fluxes[OBJECTIVE], optimum);
}

#[test]
fn pfba_is_exact_and_parsimonious() {
    let (status, fluxes) = toy().solve(&[], true);
    assert_eq!(status, Status::Optimal(Rational::new(387, 110)));
    assert_feasible(&fluxes);
    assert_eq!(fluxes[OBJECTIVE], Rational::new(387, 110));
    let total = total_flux(&fluxes);
    assert_eq!(total, Rational::new(2481, 22));
    assert!(close(total.to_f64(), GLPK_TOTAL_FLUX));
}

#[test]
fn overrides_are_temporary() {
    let mut fba = toy();
    let (original, _) = fba.solve(&[], true);
    let off = (KNOCKED_OUT, Some(rational(0.0)), Some(rational(0.0)));
    let (status, fluxes) = fba.solve(&[off], true);
    assert_feasible_with_knockout(&fluxes);
    let (Status::Optimal(knocked), Status::Optimal(full)) = (status, original.clone()) else {
        panic!("optimal")
    };
    assert!(knocked < full);
    assert_eq!(fba.solve(&[], true).0, original);
}

fn assert_feasible_with_knockout(fluxes: &[Rational]) {
    assert_eq!(fluxes[KNOCKED_OUT], rational(0.0));
    assert_feasible(fluxes);
}

#[test]
fn crossed_bounds_are_infeasible() {
    let mut fba = toy();
    let crossed = (OBJECTIVE, Some(rational(2.0)), Some(rational(1.0)));
    assert_eq!(fba.solve(&[crossed], true), (Status::Infeasible, Vec::new()));
    // A lower bound the network cannot reach.
    let too_high = (OBJECTIVE, Some(rational(4.0)), None);
    assert_eq!(fba.solve(&[too_high], false).0, Status::Infeasible);
    assert!(matches!(fba.solve(&[], true).0, Status::Optimal(_)));
}
