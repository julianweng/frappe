"""
Basic Example: Simple FMAP Agent

Demonstrates how to:
1. Create an opponent
2. Initialize an FMAP agent
3. Learn from observations
4. Compute best response
"""

import numpy as np
from src.core import get_kuhn_game
from src.agents import FMAPAgent, DirichletOpponent


def main():
    print("=" * 60)
    print("FMAP Basic Example")
    print("=" * 60)

    game = get_kuhn_game()
    print(f"\nGame: Kuhn Poker")
    print(f"  Player 1 sequences: {game.num_p1_sequences}")
    print(f"  Player 2 sequences: {game.num_p2_sequences}")

    #static opponent
    print("\n" + "-" * 60)
    print("Creating opponent...")
    opponent = DirichletOpponent(
        alpha=np.ones(game.num_p2_sequences) * 2.0,
        num_sequences=game.num_p2_sequences,
        random_state=42
    )
    true_strategy = opponent.get_strategy()
    print(f"Opponent strategy (first 5 components): {true_strategy[:5]}")

    # FMAP agent
    print("\n" + "-" * 60)
    print("Creating FMAP agent (using PGD)...")
    agent = FMAPAgent(
        game=game,
        solver_type='pgd',
        safety_mode="none",
        discount_factor=1.0,
        verbose=True
    )

    print("\n" + "-" * 60)
    print("Simulating learning process...")
    num_iterations = 100

    for t in range(num_iterations):
        agent_strategy = agent.act(iteration=t)

        expected_payoff = np.dot(
            agent_strategy,
            np.dot(game.A, true_strategy)
        )

        num_observable = np.random.randint(1, 5)
        observable_seqs = np.random.choice(
            range(game.num_p2_sequences),
            size=num_observable,
            replace=False
        ).tolist()
        q_values = np.ones(num_observable) / num_observable

        agent.add_observation(observable_seqs, q_values)

        if (t + 1) % 10 == 0:
            info = agent.update_model()
            l2_error = agent.get_model_l2_error(true_strategy)

            print(f"Iteration {t+1:3d}: "
                  f"Profit={expected_payoff:6.4f}, "
                  f"L2 Error={l2_error:8.6f}, "
                  f"Opt iters={info['iterations']:3d}")

    print("\n" + "=" * 60)
    print("Final Results")
    print("=" * 60)
    final_l2 = agent.get_model_l2_error(true_strategy)
    print(f"Final L2 Error: {final_l2:.6f}")

    # comparing with best response
    best_response = agent.compute_best_response(true_strategy)
    br_value = np.dot(best_response, np.dot(game.A, true_strategy))
    print(f"Best Response Value (oracle): {br_value:.4f}")

    model_br = agent.compute_best_response(agent.y_model)
    model_value = np.dot(model_br, np.dot(game.A, true_strategy))
    print(f"Learned Model Value: {model_value:.4f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
