% ==============================================================================
% Fixpoints of a Boolean network coupled to a metabolic network
% ==============================================================================
% Answer set: a state x left unchanged by its synchronous update, with fluxes
% f(R) at steady state, within bounds and the medium, in which obj carries flux.
% Run by solver.py: merrinasp propagator, --heuristic=Domain, --project when
% enumerating. The only linear variables are the fluxes f(R).
%
% Input facts
%   rule(N,C)        clause C is a disjunct of the rule of node N
%   clause(C)        C is a conjunctive clause
%   clause(C,L,V)    C reads literal L with sign V (1, -1); L is a node or
%                    thr(A,OP,B) for [A > B] (gt) / [A < B] (lt), on a flux if
%                    A has bounds/3, else on a concentration
%   cst(N,V)         node N is constant at V (1, -1)
%   reaction(R,M,S)  stoichiometric coefficient S of M in R
%   bounds(R,LB,UB)  flux bounds ("inf", "-inf" allowed), one per reaction
%   ext(M,R)         external metabolite M is exchanged by R
%   medium(A,V)      concentration V of A in the medium
%   obj(R)           R must carry a positive flux
% Numbers are integers or strings ("0.21", "1e-6"), always written the same way.
% @sign, @cmp and @clash compare them exactly (rFBApy/asp.py, GroundingContext).

#defined rule/2. #defined clause/1. #defined clause/3. #defined cst/2.
#defined reaction/3. #defined bounds/3. #defined ext/2. #defined medium/2.
#defined obj/1.

% Strictness margin of the linear constraints, well above the LP tolerance
#const eps = "1e-4".


% ==============================================================================
% Boolean network
% ==============================================================================
x(N) :- cst(N,1).

% Undefined node: 1, or 0 if it is also a threshold variable
x(N) :- clause(_,N,_), not rule(N,_), not cst(N,_), not thr(N,_,_),
        #false: N = thr(_,_,_).

{ x(N) } :- rule(N,_).

eval(C) :- clause(C); x(L): clause(C,L,1); not x(L): clause(C,L,-1).

% Fixpoint
:- rule(N,C), eval(C), not x(N).
:- rule(N,_), x(N); not eval(C): rule(N,C).

#project x(N) : rule(N,_).


% ==============================================================================
% Threshold predicates
% ==============================================================================
thr(A,OP,B) :- clause(_,thr(A,OP,B),_).

% --- Metabolite of the medium: decided at grounding ---------------------------
x(thr(A,OP,B)) :- thr(A,OP,B), medium(A,V), @cmp(OP,V,B) = 1.

% --- Missing metabolite: positive iff produced, else 0 -------------------------
miss(A) :- clause(_,thr(A,_,_),_), not medium(A,_), not bounds(A,_,_).

% One @sign call per distinct coefficient
coef(S) :- reaction(_,_,S).
sgn(S,@sign(S)) :- coef(S).

thr(R,gt,0) :- miss(A), reaction(R,A,S), sgn(S,1).
thr(R,lt,0) :- miss(A), reaction(R,A,S), sgn(S,-1).

prod(A) :- miss(A), reaction(R,A,S), sgn(S,1), x(thr(R,gt,0)).
prod(A) :- miss(A), reaction(R,A,S), sgn(S,-1), x(thr(R,lt,0)).

x(thr(A,gt,B)) :- miss(A), thr(A,gt,B), @sign(B) = -1.
x(thr(A,gt,B)) :- prod(A), thr(A,gt,B), @sign(B) = 0.
x(thr(A,lt,B)) :- miss(A), thr(A,lt,B), @sign(B) = 1, not prod(A).
{ x(thr(A,OP,B)) } :- prod(A), thr(A,OP,B), @sign(B) = 1.

% --- Flux: chosen, decided by the linear constraints -------------------------
{ x(thr(R,OP,B)) } :- thr(R,OP,B), bounds(R,_,_).

% Decide flux predicates first: the rest follows by propagation
#heuristic x(thr(R,OP,B)) : thr(R,OP,B), bounds(R,_,_). [1,level]

&sum{f(R)} >= B + eps :- bounds(R,_,_), x(thr(R,gt,B)).
&sum{f(R)} <= B       :- bounds(R,_,_), thr(R,gt,B), not x(thr(R,gt,B)).
&sum{f(R)} <= B - eps :- bounds(R,_,_), x(thr(R,lt,B)).
&sum{f(R)} >= B       :- bounds(R,_,_), thr(R,lt,B), not x(thr(R,lt,B)).

% --- Consistency of predicates on the same variable (exact) ------------------
:- x(thr(A,O1,B1)), x(thr(A,O2,B2)), (O1,B1) < (O2,B2),
   @clash(O1,B1,1,O2,B2,1) = 1.

:- x(thr(A,O1,B1)), thr(A,O2,B2), not x(thr(A,O2,B2)),
   @clash(O1,B1,1,O2,B2,-1) = 1.

:- thr(A,O1,B1), thr(A,O2,B2), (O1,B1) < (O2,B2),
   not x(thr(A,O1,B1)), not x(thr(A,O2,B2)), @clash(O1,B1,-1,O2,B2,-1) = 1.


% ==============================================================================
% Metabolic network
% ==============================================================================
&dom{LB..UB} = f(R) :- bounds(R,LB,UB).

% Steady state of internal metabolites
&sum{S*f(R): reaction(R,M,S)} = 0 :- reaction(_,M,_), not ext(M,_).

% No uptake of an external metabolite the medium does not feed
fed(M) :- medium(M,V), @sign(V) = 1.

&sum{f(R)} <= 0 :- ext(M,R), reaction(R,M,S), sgn(S,-1), not fed(M).

&sum{f(R)} >= 0 :- ext(M,R), reaction(R,M,S), sgn(S,1), not fed(M).

% Regulation: a reaction whose node is 0 carries no flux
&sum{f(R)} = 0 :- rule(R,_), bounds(R,_,_), not x(R).

&sum{f(R)} = 0 :- cst(R,-1), bounds(R,_,_).

&sum{f(R)} >= eps :- obj(R).
