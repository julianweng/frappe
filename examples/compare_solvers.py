"""
Comparing Solvers betweeen PGD (QP-based) and Frank-Wolfe (Linear based)
Looking for convergence speed (in iterations), wall-clock time and final L2 error
"""

import numpy as np
import time
from src.core import get_kuhn_game
from src.agents import FMAPAgent, DirichletOpponent


def test_solver(
    game,
    opponent_strategy,
    solver_type: str,
    num_observations: int = 100
):
    """Test a single solver configuration"""

    agent = FMAPAgent(
        game=game,
        solver_type=solver_type,
        safety_mode="none",
        discount_factor=1.0,
        verbose=False
    )

    # Add observations
    for _ in range(num_observations):
        num_observable = np.random.randint(1, 5)
        observable_seqs = np.random.choice(
            range(game.num_p2_sequences),
            size=num_observable,
            replace=False
        ).tolist()
        q_values = np.ones(num_observable) / num_observable
        agent.add_observation(observable_seqs, q_values)

    start_time = time.time()
    info = agent.update_model()
    elapsed = time.time() - start_time

    l2_error = agent.get_model_l2_error(opponent_strategy)

    return {
        'solver': solver_type.upper(),
        'iterations': info['iterations'],
        'time': elapsed,
        'l2_error': l2_error,
        'converged': info['converged']
    }


def main():
    print("=" * 70)
    print("FMAP Solver Comparison: PGD vs Frank-Wolfe")
    print("=" * 70)

    game = get_kuhn_game()
    np.random.seed(42)

    opponent = DirichletOpponent(
        alpha=np.ones(game.num_p2_sequences) * 2.0,
        num_sequences=game.num_p2_sequences,
        random_state=42
    )
    opponent_strategy = opponent.get_strategy()

    observation_counts = [10, 50, 100, 200, 500]

    print("\nTesting scalability with different numbers of observations...\n")
    print(f"{'Observations':<15} {'Solver':<10} {'Time (s)':<12} "
          f"{'Iters':<8} {'L2 Error':<12} {'Converged':<10}")
    print("-" * 70)

    for num_obs in observation_counts:
        for solver in ['pgd', 'fw']:
            result = test_solver(game, opponent_strategy, solver, num_obs)

            print(f"{num_obs:<15} {result['solver']:<10} "
                  f"{result['time']:<12.6f} {result['iterations']:<8} "
                  f"{result['l2_error']:<12.6f} {result['converged']}")

    print("\n" + "=" * 70)
    print("Detailed Comparison (100 observations)")
    print("=" * 70)

    for solver in ['pgd', 'fw']:
        result = test_solver(game, opponent_strategy, solver, 100)

        print(f"\n{result['solver']}:")
        print(f"  Time: {result['time']:.6f} seconds")
        print(f"  Optimization iterations: {result['iterations']}")
        print(f"  L2 Error: {result['l2_error']:.6f}")
        print(f"  Converged: {result['converged']}")

    print("\nConclusion:")
    print("Frank-Wolfe should be faster (lower wall-clock time) while")
    print("achieving similar L2 error compared to PGD.")


if __name__ == "__main__":
    main()
