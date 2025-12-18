"""
Functions for Kuhn Poker observability and strategy conversion
"""

import numpy as np
from typing import List, Set, Dict


def build_observability_map() -> Dict[str, Dict[str, List[int]]]:
    """
    Build the observability functions o_1 and o_2 to returns a dict mapping leaf nodes to observable opp sequences.
    """

    observability = {
        'p1': {},
        'p2': {}
    }

    from .kuhn import P1_SEQUENCES, P2_SEQUENCES

    def p1_idx(seq_name: str) -> int:
        """Get index of P1 sequence by name"""
        try:
            return P1_SEQUENCES.index(seq_name)
        except ValueError:
            raise ValueError(f"P1 sequence not found: {seq_name}")

    def p2_idx(seq_name: str) -> int:
        """Get index of P2 sequence by name"""
        try:
            return P2_SEQUENCES.index(seq_name)
        except ValueError:
            raise ValueError(f"P2 sequence not found: {seq_name}")

    cards = ["K", "Q", "J"]
    deals = [(c1, c2) for c1 in cards for c2 in cards if c1 != c2]

    for c1, c2 in deals:
        remaining_for_p1 = [c for c in cards if c != c1]
        remaining_for_p2 = [c for c in cards if c != c2]

        # P1 bets, P2 calls -> showdown 
        leaf = f"B_{c1} ca_{c2}"
        observability["p1"][leaf] = [p2_idx(f"ca_{c2}")]
        observability["p2"][leaf] = [p1_idx(f"B_{c1}")]

        # P1 bets, P2 folds -> no showdown
        leaf = f"B_{c1} f_{c2}"
        observability["p1"][leaf] = [p2_idx(f"f_{c}") for c in remaining_for_p1]
        observability["p2"][leaf] = [p1_idx(f"B_{c}") for c in remaining_for_p2]

        # P1 checks, P2 checks -> showdown
        leaf = f"Ch_{c1} ch_{c2}"
        observability["p1"][leaf] = [p2_idx(f"ch_{c2}")]
        observability["p2"][leaf] = [p1_idx(f"Ch_{c1}")]

        # P1 checks, P2 bets, P1 calls -> showdown
        leaf = f"Ch_{c1} b_{c2} Ca_{c1}"
        observability["p1"][leaf] = [p2_idx(f"b_{c2}")]
        observability["p2"][leaf] = [p1_idx(f"Ch_{c1} Ca_{c1}")]

        # P1 checks, P2 bets, P1 folds -> no showdown
        leaf = f"Ch_{c1} b_{c2} F_{c1}"
        observability["p1"][leaf] = [p2_idx(f"b_{c}") for c in remaining_for_p1]
        observability["p2"][leaf] = [p1_idx(f"Ch_{c} F_{c}") for c in remaining_for_p2]

    return observability


def get_observable_sequences(leaf_node: str, player: int = 2) -> List[int]:
    """
    Get the observable sequences for a player given a leaf node
    """
    obs_map = build_observability_map()
    player_key = f'p{player}'
    try:
        return obs_map[player_key][leaf_node]
    except KeyError:
        raise ValueError(f"Leaf node {leaf_node} not found in observability map")


def compute_expected_value(x: np.ndarray, y: np.ndarray, A: np.ndarray) -> float:
    """
    Compute expected value of payoff to player 1
    """
    return np.dot(x, np.dot(A, y))


def sample_from_dirichlet(alpha: np.ndarray, size: int = 1) -> np.ndarray:
    return np.random.dirichlet(alpha, size=size)


def l2_distance(y1: np.ndarray, y2: np.ndarray) -> float:
    return np.linalg.norm(y1 - y2, ord=2)


def ensure_positive(y: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    return np.maximum(y, epsilon)


def get_uniform_strategy(F: np.ndarray, f: np.ndarray) -> np.ndarray:
    """
    Create a uniform feasible strategy that satisfies F y = f (or E x = e).
    For Kuhn Poker specifically, this supports both players:
    - Root sequence has weight 1
    - At each information set, distribute uniformly over available actions
    """
    n = F.shape[1] 
    y = np.zeros(n)

    if n == 13 and f.shape[0] == 7 and np.isclose(f[0], 1.0):
        if np.isclose(F[2, 2], -1.0): 
            y[0] = 1.0
            y[1] = y[2] = 0.5    #king
            y[3] = y[4] = 0.25  

            y[5] = y[6] = 0.5    #queen
            y[7] = y[8] = 0.25   

            y[9] = y[10] = 0.5   # jack
            y[11] = y[12] = 0.25 
        else:  
            y[0] = 1.0
            #queen
            y[1] = y[2] = 0.5   
            y[3] = y[4] = 0.5  
            #jack 
            y[5] = y[6] = 0.5    
            y[7] = y[8] = 0.5   
            #king
            y[9] = y[10] = 0.5   
            y[11] = y[12] = 0.5 
    else:
        try:
            import gurobipy as gp
            from gurobipy import GRB

            model = gp.Model("UniformStrategy")
            model.setParam('OutputFlag', 0)

            y_vars = model.addMVar(n, lb=0, name="y")

            model.addConstr(F @ y_vars == f, name="constraints")

            uniform = np.ones(n) / n
            model.setObjective((y_vars - uniform) @ (y_vars - uniform), GRB.MINIMIZE)

            model.optimize()

            if model.status == GRB.OPTIMAL:
                y = y_vars.X.copy()
            else:
                raise RuntimeError("Failed to find feasible uniform strategy")

        except ImportError:
            from scipy.optimize import minimize, LinearConstraint

            uniform = np.ones(n) / n

            def objective(y_vec):
                return np.sum((y_vec - uniform) ** 2)

            def grad(y_vec):
                return 2 * (y_vec - uniform)

            constraint = LinearConstraint(F, f, f)
            bounds = [(0, None) for _ in range(n)]

            y0 = np.ones(n) * 0.1
            y0[0] = f[0]

            result = minimize(
                objective,
                y0,
                method='SLSQP',
                jac=grad,
                constraints=constraint,
                bounds=bounds
            )

            if result.success:
                y = result.x
            else:
                raise RuntimeError(f"Failed to find feasible uniform strategy: {result.message}")

    return y
