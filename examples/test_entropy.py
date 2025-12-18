"""
Test entropy regularization by comparing FMAP w/ & w/out entropy reg
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
from src.core import get_kuhn_game
from src.agents import FMAPAgent


def run_test(entropy_weight, n_obs, opponent, seed):
    np.random.seed(seed)
    game = get_kuhn_game()
    
    agent = FMAPAgent(game, solver_type='pgd', require_gurobi=True,
                     entropy_weight=entropy_weight)
    
    total_time = 0
    for i in range(n_obs):
        obs = [np.random.randint(1, 13)]
        q = np.array([1.0])
        agent.add_observation(obs, q)
        
        start = time.perf_counter()
        agent.update_model()
        total_time += time.perf_counter() - start
    
    l2_error = np.linalg.norm(agent.y_model - opponent)
    
    y = np.maximum(agent.y_model, 1e-12)
    entropy = -np.sum(y * np.log(y))
    
    return {
        'time_ms': total_time * 1000,
        'l2_error': l2_error,
        'entropy': entropy,
    }


def main():
    print("=" * 60)
    print("ENTROPY REGULARIZATION TEST")
    print("=" * 60)
    
    entropy_weights = [0.0, 0.001, 0.01, 0.05, 0.1]
    
    n_obs = 100  
    n_trials = 5
    
    print(f"\nRunning {n_trials} trials with {n_obs} observations each")
    print("(Fewer observations makes regularization effect more visible)\n")
    
    np.random.seed(42)
    opponents = [np.random.dirichlet(np.ones(13) * 2) for _ in range(n_trials)]
    
    results = {w: {'times': [], 'errors': [], 'entropies': []} for w in entropy_weights}
    
    for trial in range(n_trials):
        print(f"Trial {trial + 1}/{n_trials}...")
        opponent = opponents[trial]
        
        for w in entropy_weights:
            result = run_test(w, n_obs, opponent, seed=trial*100)
            results[w]['times'].append(result['time_ms'])
            results[w]['errors'].append(result['l2_error'])
            results[w]['entropies'].append(result['entropy'])
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    baseline_time = np.mean(results[0.0]['times'])
    baseline_error = np.mean(results[0.0]['errors'])
    
    print(f"\n{'λ (entropy)':<12} │ {'Time (ms)':>10} │ {'L2 Error':>10} │ {'Entropy':>10} │ {'Error Gap':>10}")
    print("-" * 70)
    
    for w in entropy_weights:
        t = np.mean(results[w]['times'])
        err = np.mean(results[w]['errors'])
        ent = np.mean(results[w]['entropies'])
        gap = (err - baseline_error) / baseline_error * 100
        
        label = "baseline" if w == 0 else f"λ={w}"
        print(f"{label:<12} │ {t:>10.1f} │ {err:>10.4f} │ {ent:>10.4f} │ {gap:>+9.1f}%")
    
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    print("""
- Higher λ → Higher entropy (more spread out/uncertain model)
- Higher λ → Slightly worse L2 error (regularization trades accuracy for robustness)
- The key benefit: Strong convexity gives faster convergence in theory

Recommended λ values:
- λ = 0.01: Light regularization, minimal accuracy impact
- λ = 0.05: Moderate regularization, noticeable entropy increase
- λ = 0.1:  Strong regularization, prevents overconfidence
""")


if __name__ == '__main__':
    main()

