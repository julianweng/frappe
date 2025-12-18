"""
Comparing SFW with and without SVRG variance reduction
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
from src.core import get_kuhn_game
from src.agents import FMAPAgent


def run_test(solver_type, solver_kwargs, n_obs, opponent, seed):
    np.random.seed(seed)
    game = get_kuhn_game()
    
    agent = FMAPAgent(game, solver_type=solver_type, require_gurobi=True,
                     solver_kwargs=solver_kwargs)
    
    total_time = 0
    for i in range(n_obs):
        obs = [np.random.randint(1, 13)]
        q = np.array([1.0])
        agent.add_observation(obs, q)
        
        start = time.perf_counter()
        agent.update_model()
        total_time += time.perf_counter() - start
    
    l2_error = np.linalg.norm(agent.y_model - opponent)
    
    return {
        'time_ms': total_time * 1000,
        'l2_error': l2_error,
    }


def main():
    print("=" * 60)
    print("SVRG VARIANCE REDUCTION TEST")
    print("=" * 60)
    
    configs = [
        ('PGD', 'pgd', {}),
        ('SFW (no VR)', 'sfw', {'batch_size': 100, 'use_variance_reduction': False}),
        ('SFW-SVRG', 'sfw', {'batch_size': 100, 'use_variance_reduction': True, 'variance_reduction_freq': 5}),
        ('SFW-SVRG-10', 'sfw', {'batch_size': 100, 'use_variance_reduction': True, 'variance_reduction_freq': 10}),
    ]
    
    n_obs = 300
    n_trials = 5
    
    print(f"\nRunning {n_trials} trials with {n_obs} observations each\n")
    
    np.random.seed(42)
    opponents = [np.random.dirichlet(np.ones(13) * 2) for _ in range(n_trials)]
    
    results = {name: {'times': [], 'errors': []} for name, _, _ in configs}
    
    for trial in range(n_trials):
        print(f"Trial {trial + 1}/{n_trials}...")
        opponent = opponents[trial]
        
        for name, solver_type, kwargs in configs:
            result = run_test(solver_type, kwargs, n_obs, opponent, seed=trial*100)
            results[name]['times'].append(result['time_ms'])
            results[name]['errors'].append(result['l2_error'])
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    pgd_time = np.mean(results['PGD']['times'])
    pgd_error = np.mean(results['PGD']['errors'])
    
    print(f"\n{'Config':<15} │ {'Time (ms)':>10} │ {'Speedup':>8} │ {'L2 Error':>10} │ {'Error Gap':>10}")
    print("-" * 65)
    
    for name, _, _ in configs:
        t = np.mean(results[name]['times'])
        t_std = np.std(results[name]['times'])
        err = np.mean(results[name]['errors'])
        err_std = np.std(results[name]['errors'])
        speedup = pgd_time / t
        gap = (err - pgd_error) / pgd_error * 100
        
        print(f"{name:<15} │ {t:>10.1f} │ {speedup:>7.2f}x │ {err:>10.4f} │ {gap:>+9.1f}%")
    
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    print("""
- SVRG should have LOWER error than plain SFW at similar speed
- SVRG computes a full gradient every N iterations for variance reduction
- Trade-off: More accurate but slightly slower per iteration
""")


if __name__ == '__main__':
    main()

