"""
FW vs PGD Debuging
Basically generating set of observations from fixed opponent -> run both 
side-by-side. Save results in CSV. 
"""

import os
import time
import numpy as np
import pandas as pd

from src.core import get_kuhn_game, get_uniform_strategy
from src.agents import DirichletOpponent
from src.engine.runner import _simulate_observation_worker
from src.optimization import ProjectedGradientDescent, FrankWolfe
from src.optimization.gradients import (
    compute_gradient,
    compute_objective,
    precompute_observation_data,
)


def generate_observations(opponent_strategy, num_observations: int, seed: int = 0):
    """Simulate observations using the runner's observation model."""
    from src.core import get_uniform_strategy, get_kuhn_game
    game = get_kuhn_game()

    agent_strategy = get_uniform_strategy(game.E, game.e)
    
    rng = np.random.default_rng(seed)
    observations = []
    for _ in range(num_observations):
        np.random.seed(rng.integers(0, 1_000_000))
        observations.append(_simulate_observation_worker(agent_strategy, opponent_strategy))
    return observations


def run_solver_debug(
    solver_name: str,
    game,
    observations,
    true_strategy,
    max_iters: int = 30,
):
    """Run a solver for a fixed number of iterations, collecting diagnostics."""
    if solver_name == "pgd":
        solver = ProjectedGradientDescent(
            F=game.F,
            f=game.f,
            alpha=np.ones(game.num_p2_sequences) * 2.0,
            verbose=False,
            max_iterations=max_iters,
            early_stopping_patience=None,
            tolerance=0.0,  # don't stop early -> control loop
        )
    elif solver_name == "fw":
        solver = FrankWolfe(
            F=game.F,
            f=game.f,
            alpha=np.ones(game.num_p2_sequences) * 2.0,
            verbose=False,
            max_iterations=max_iters,
            early_stopping_patience=None,
            tolerance=0.0,
            use_away_steps=True,
            step_size_rule="adaptive",
        )
    else:
        raise ValueError(f"Unknown solver {solver_name}")

    y = get_uniform_strategy(game.F, game.f)
    history = []
    precomputed = precompute_observation_data(observations)

    start = time.time()
    for k in range(max_iters):
        solver.iteration_count = k + 1

        grad = compute_gradient(
            y,
            solver.alpha,
            observations,
            epsilon=solver.epsilon,
            discount_weights=None,
            precomputed=precomputed,
        )

        current_obj = compute_objective(
            y,
            solver.alpha,
            observations,
            epsilon=solver.epsilon,
            discount_weights=None,
            precomputed=precomputed,
        )

        if solver_name == "pgd":
            y_next = solver.step(y, grad, observations, None, current_obj, precomputed)
        else: # FW doesn't need current objective unless we use line search
            y_next = solver.step(y, grad, observations, None, None, precomputed)

        delta = np.linalg.norm(y_next - y)
        obj_next = compute_objective(
            y_next,
            solver.alpha,
            observations,
            epsilon=solver.epsilon,
            discount_weights=None,
            precomputed=precomputed,
        )
        l2_err = np.linalg.norm(y_next - true_strategy)

        history.append(
            {
                "solver": solver_name.upper(),
                "iter": k + 1,
                "objective": obj_next,
                "gradient_norm": np.linalg.norm(grad),
                "delta": delta,
                "dual_gap": getattr(solver, "last_gap", np.nan),
                "step_size": getattr(solver, "last_step_size", np.nan),
                "l2_error": l2_err,
            }
        )

        y = y_next

    total_time = time.time() - start
    return history, total_time


def main():
    game = get_kuhn_game()
    np.random.seed(123)

    opponent = DirichletOpponent(
        alpha=np.ones(game.num_p2_sequences) * 2.0,
        num_sequences=game.num_p2_sequences,
        random_state=123,
        game=game,
    )
    true_strategy = opponent.get_strategy()

    observations = generate_observations(true_strategy, num_observations=200, seed=99)

    all_rows = []
    summaries = []

    for solver_name in ["pgd", "fw"]:
        rows, elapsed = run_solver_debug(
            solver_name, game, observations, true_strategy, max_iters=30
        )
        all_rows.extend(rows)

        last = rows[-1]
        summaries.append(
            {
                "solver": solver_name.upper(),
                "time_sec": elapsed,
                "final_obj": last["objective"],
                "final_l2_error": last["l2_error"],
                "final_gap": last["dual_gap"],
                "final_step": last["step_size"],
            }
        )

    df = pd.DataFrame(all_rows)
    os.makedirs("results", exist_ok=True)
    csv_path = "results/debug_fw_vs_pgd.csv"
    df.to_csv(csv_path, index=False)

    print("\nSaved per-iteration diagnostics to", csv_path)
    print("\nSummary (final iteration):")
    for s in summaries:
        print(
            f"{s['solver']:>4} | time={s['time_sec']:.4f}s "
            f"obj={s['final_obj']:.6f} l2={s['final_l2_error']:.6f} "
            f"gap={s['final_gap']:.6f} step={s['final_step']}"
        )
    print("\nInspect the CSV to compare trajectories (objective, gap, step sizes).")


if __name__ == "__main__":
    main()
