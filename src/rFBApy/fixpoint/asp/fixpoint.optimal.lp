% ==============================================================================
% Optimality cuts (loaded with fixpoint.lp when `optimal`)
% ==============================================================================
% A cut is an optimal flux vector u of some state, with its growth G: wherever
% u is feasible, the optimum is at least G. Cuts may come from other instances
% of the same network (cache.py); solver.py only writes those within the bounds
% and the medium of this instance.
%
% Input facts
%   cut(K,G)       cut K: growth at least G
%   needs(K,R,D)   u of cut K uses R in direction D (1, -1)
%   hint(T,V)      try x(T) true (1) or false (-1) first
%   found(K)       K is a fixpoint already returned
%   found(K,N)     node N is at 1 in fixpoint K

#defined cut/2. #defined needs/3. #defined hint/2.
#defined found/1. #defined found/2.

% Directions closed by the state or the medium, for the reactions of the cuts.
% Every restriction fixpoint.lp puts on the fluxes must be listed here.
blocked(R,D) :- needs(_,R,D), rule(R,_), not x(R).

blocked(R,D) :- needs(_,R,D), cst(R,-1).

blocked(R,1)  :- needs(_,R,1),  ext(M,R), reaction(R,M,S), sgn(S,-1), not fed(M).

blocked(R,-1) :- needs(_,R,-1), ext(M,R), reaction(R,M,S), sgn(S,1), not fed(M).

cut_off(K) :- needs(K,R,D), blocked(R,D).

&sum{f(R)} >= G :- cut(K,G), obj(R), not cut_off(K).

% A new solver per cut: exclude the states already returned
:- found(K); x(N): found(K,N); not x(N): rule(N,_), not found(K,N).

#heuristic x(T) : hint(T,1).  [1,true]
#heuristic x(T) : hint(T,-1). [1,false]
