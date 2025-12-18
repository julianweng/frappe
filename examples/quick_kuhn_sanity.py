import numpy as np

from src.core import get_kuhn_game
from src.agents import FMAPAgent, DirichletOpponent
from src.engine.runner import _simulate_observation_worker


def summarize_p2_behavior(y: np.ndarray) -> dict:
    return {
        "Q vs bet (call/fold)": (y[1], y[2]),
        "Q vs check (bet/check)": (y[3], y[4]),
        "J vs bet (call/fold)": (y[5], y[6]),
        "J vs check (bet/check)": (y[7], y[8]),
        "K vs bet (call/fold)": (y[9], y[10]),
        "K vs check (bet/check)": (y[11], y[12]),
    }


def print_behavior(label: str, y: np.ndarray, top_k: int = 4) -> None:
    print(f"{label}")
    summary = summarize_p2_behavior(y)
    for k, (a, b) in summary.items():
        print(f"  {k:24s}: {a:.3f} | {b:.3f}")
    pairs = [
        (float(y[idx]), name)
        for idx, name in enumerate(get_kuhn_game().p2_sequences)
        if idx != 0
    ]
    pairs.sort(reverse=True)
    head = ", ".join(f"{name}={prob:.3f}" for prob, name in pairs[:top_k])
    print(f"  top sequences: {head}")


def run_case(name: str, alpha: np.ndarray, num_iters: int = 120, seed: int = 0) -> None:
    np.random.seed(seed)
    game = get_kuhn_game()
    opponent = DirichletOpponent(
        alpha=alpha,
        num_sequences=game.num_p2_sequences,
        random_state=seed,
        game=game,
    )
    true_y = opponent.get_strategy()

    print("\n" + "=" * 72)
    print(f"{name}")
    print("=" * 72)
    print_behavior("Opponent behavior:", true_y)

    agent = FMAPAgent(
        game=game,
        solver_type="pgd",
        discount_factor=1.0,
        safety_mode="none",
        verbose=False,
    )

    checkpoints = {1, 5, 20, 40, 80, num_iters}
    last_obs = ([], np.array([]))

    # oracle best response value to the true strategy (upper bound)
    oracle_br = agent.compute_best_response(true_y)
    oracle_value = float(oracle_br @ game.A @ true_y)
    print(f"Oracle best-response value vs opponent: {oracle_value:.4f}")

    for t in range(num_iters):
        agent_strategy = agent.act(iteration=t)
        last_obs = _simulate_observation_worker(agent_strategy, true_y)
        agent.add_observation(*last_obs)
        agent.update_model()

        if (t + 1) in checkpoints:
            l2 = agent.get_model_l2_error(true_y)
            played_value = float(agent_strategy @ game.A @ true_y)
            br_to_model = agent.compute_best_response(agent.y_model)
            model_value = float(br_to_model @ game.A @ true_y)
            obs_seqs, q_vals = last_obs
            print(
                f"iter {t + 1:4d} | "
                f"l2={l2:6.4f} | "
                f"played_ev={played_value:7.4f} | "
                f"model_br_ev={model_value:7.4f} | "
                f"oracle_br_ev={oracle_value:7.4f} | "
                f"obs={obs_seqs}, q={np.round(q_vals, 3)}"
            )

    print_behavior("Learned opponent model:", agent.y_model)
    final_br = agent.compute_best_response(agent.y_model)
    final_value = float(final_br @ game.A @ true_y)
    print(f"Final best-response value vs true opponent: {final_value:.4f}")
    print(f"Final L2 error to true strategy: {agent.get_model_l2_error(true_y):.4f}")


def main():
    # Passive opponent
    passive_alpha = np.array([1, 8, 1, 8, 1, 8, 1, 8, 1, 8, 1, 8], dtype=float)
    run_case("Passive opponent (fold/check heavy)", passive_alpha, num_iters=120, seed=3)

    # Aggressive opponent
    aggressive_alpha = np.array([8, 1, 8, 1, 8, 1, 8, 1, 8, 1, 8, 1], dtype=float)
    run_case("Aggressive opponent (bet/call heavy)", aggressive_alpha, num_iters=120, seed=7)

    # Near-uniform opponent for sanity
    uniform_alpha = np.ones(12) * 2.0
    run_case("Near-uniform opponent", uniform_alpha, num_iters=120, seed=11)


if __name__ == "__main__":
    main()

