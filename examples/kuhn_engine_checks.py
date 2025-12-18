"""
Checking for errors in Kuhn engine, payoffs, and observation model
Test: uv run python examples/kuhn_engine_checks.py
"""

import numpy as np
from scipy.optimize import linprog

from src.core import get_kuhn_game, get_uniform_strategy
from src.agents import NashAgent, DirichletOpponent
from src.engine.runner import _simulate_observation_worker


TOL = 1e-9


def best_response_p1(game, y: np.ndarray) -> np.ndarray:
    """Solve max_x x^T A y  s.t. E x = e, x >= 0."""
    res = linprog(
        -game.A @ y,
        A_eq=game.E,
        b_eq=game.e,
        bounds=[(0, None)] * game.num_p1_sequences,
        method="highs",
    )
    if not res.success:
        raise RuntimeError(f"P1 best-response LP failed: {res.message}")
    return res.x


def best_response_p2(game, x: np.ndarray) -> np.ndarray:
    """Solve min_y x^T A y  s.t. F y = f, y >= 0."""
    res = linprog(
        game.A.T @ x,
        A_eq=game.F,
        b_eq=game.f,
        bounds=[(0, None)] * game.num_p2_sequences,
        method="highs",
    )
    if not res.success:
        raise RuntimeError(f"P2 best-response LP failed: {res.message}")
    return res.x


def check_constraints(game):
    print("Checking basic dimensions and constraints...")
    assert game.E.shape == (7, 13)
    assert game.F.shape == (7, 13)
    assert game.A.shape == (13, 13)
    assert game.e.shape == (7,)
    assert game.f.shape == (7,)

    # Use a known-feasible P1 strategy (Nash) for constraint check
    x_uniform = NashAgent(game, player=1, use_gurobi=True).get_strategy()
    y_uniform = get_uniform_strategy(game.F, game.f)

    assert np.allclose(game.E @ x_uniform, game.e, atol=TOL)
    assert np.allclose(game.F @ y_uniform, game.f, atol=TOL)
    print("  Constraints satisfied for uniform feasible points.")


def check_nash_consistency(game):
    print("Checking Nash vs best responses consistency...")
    x_nash = NashAgent(game, player=1, use_gurobi=True).get_strategy()
    y_nash = NashAgent(game, player=2, use_gurobi=True).get_strategy()

    assert np.allclose(game.E @ x_nash, game.e, atol=TOL)
    assert np.allclose(game.F @ y_nash, game.f, atol=TOL)

    x_br_to_y = best_response_p1(game, y_nash)
    y_br_to_x = best_response_p2(game, x_nash)

    val = float(x_nash @ game.A @ y_nash)
    lower = float(x_br_to_y @ game.A @ y_nash)  
    upper = float(x_nash @ game.A @ y_br_to_x) 
    gap = upper - lower

    print(f"  Nash value: {val:.6f}, lower: {lower:.6f}, upper: {upper:.6f}, gap: {gap:.3e}")
    assert gap <= 1e-7, "Nash saddle gap too large"
    assert abs(val - lower) <= 1e-7
    assert abs(val - upper) <= 1e-7


def check_random_opponent(game, seed: int = 0):
    print("Checking random opponent best responses...")
    np.random.seed(seed)
    opponent = DirichletOpponent(
        alpha=np.ones(12) * 2.0,
        num_sequences=game.num_p2_sequences,
        random_state=seed,
        game=game,
    )
    y = opponent.get_strategy()
    x_br = best_response_p1(game, y)
    assert np.allclose(game.E @ x_br, game.e, atol=TOL)
    assert np.all(x_br >= 0)

    y_br = best_response_p2(game, x_br)
    assert np.allclose(game.F @ y_br, game.f, atol=TOL)
    assert np.all(y_br >= 0)

    val = float(x_br @ game.A @ y)
    worst_case = float(x_br @ game.A @ y_br)
    print(f"  BR vs random opponent: value={val:.6f}, worst-case vs BR(P2)={worst_case:.6f}")
    assert worst_case <= val + 1e-8


def check_observation_model(game, trials: int = 100):
    print("Checking observation model outputs...")
    x = NashAgent(game, player=1, use_gurobi=True).get_strategy()
    y = DirichletOpponent(
        alpha=np.ones(12) * 2.0,
        num_sequences=game.num_p2_sequences,
        random_state=123,
        game=game,
    ).get_strategy()

    for _ in range(trials):
        obs, q_vals = _simulate_observation_worker(x, y)
        assert len(obs) == len(q_vals)
        assert all(0 <= idx < game.num_p2_sequences for idx in obs)
        assert abs(np.sum(q_vals) - 1.0) <= 1e-9
    print(f"  Observation model produced {trials} valid samples.")


def main():
    game = get_kuhn_game()
    check_constraints(game)
    check_nash_consistency(game)
    check_random_opponent(game)
    check_observation_model(game)
    print("\nAll Kuhn engine smoke checks passed.")


if __name__ == "__main__":
    main()

