"""
Optimization test to validate SFW + SVRG performance. Comparing optimizations against baseline using Gurobi 
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
import pandas as pd
from multiprocessing import Pool, cpu_count
from src.core import get_kuhn_game
from src.agents import FMAPAgent


def run_experiment(args):
    config_name, solver_type, solver_kwargs, n_obs, opponent, trial_seed = args
    
    np.random.seed(trial_seed)
    game = get_kuhn_game()
    
    agent = FMAPAgent(
        game, 
        solver_type=solver_type, 
        require_gurobi=True,
        solver_kwargs=solver_kwargs
    )
    
    total_time = 0
    for i in range(n_obs): #rng
        obs = [np.random.randint(1, 13)]
        q = np.array([1.0])
        agent.add_observation(obs, q)
        
        start = time.perf_counter()
        agent.update_model()
        total_time += time.perf_counter() - start
    
    l2_error = np.linalg.norm(agent.y_model - opponent)
    strategy = agent.act(iteration=n_obs-1)
    profit = np.dot(strategy, np.dot(game.A, opponent))
    
    return {
        'config': config_name,
        'solver': solver_type,
        'n_obs': n_obs,
        'trial': trial_seed,
        'total_time_ms': total_time * 1000,
        'avg_update_ms': (total_time * 1000) / n_obs,
        'l2_error': l2_error,
        'profit': profit,
    }


def main():
    print("=" * 70)
    print("FULL OPTIMIZATION TEST: SFW + SVRG Validation")
    print("=" * 70)
    
    configs = [
        ('PGD', 'pgd', {}),
        ('FW', 'fw', {}),
        
        # our optimizations
        ('SFW-100', 'sfw', {'batch_size': 100, 'use_variance_reduction': False}),
        ('SFW-SVRG-100', 'sfw', {'batch_size': 100, 'use_variance_reduction': True, 'variance_reduction_freq': 10}),
        
        ('SFW-200', 'sfw', {'batch_size': 200, 'use_variance_reduction': False}),
        ('SFW-SVRG-200', 'sfw', {'batch_size': 200, 'use_variance_reduction': True, 'variance_reduction_freq': 10}),
    ]
    
    observation_counts = [100, 300, 500]
    n_opponents = 5
    n_trials = 3
    
    print(f"\nConfigurations: {[c[0] for c in configs]}")
    print(f"Observations: {observation_counts}")
    print(f"Opponents: {n_opponents}, Trials per opponent: {n_trials}")
    
    np.random.seed(42)
    opponents = [np.random.dirichlet(np.ones(13) * 2) for _ in range(n_opponents)]
    
    experiments = []
    for config_name, solver_type, solver_kwargs in configs:
        for n_obs in observation_counts:
            for opp_idx, opponent in enumerate(opponents):
                for trial in range(n_trials):
                    seed = opp_idx * 1000 + trial * 100 + n_obs
                    experiments.append((config_name, solver_type, solver_kwargs, n_obs, opponent, seed))
    
    total_experiments = len(experiments)
    print(f"\nTotal experiments: {total_experiments}")
    
    n_workers = min(cpu_count(), 8)
    print(f"Using {n_workers} workers...\n")
    
    start_time = time.time()
    with Pool(n_workers) as pool:
        results = pool.map(run_experiment, experiments)
    
    elapsed = time.time() - start_time
    print(f"Completed {total_experiments} experiments in {elapsed:.1f} seconds")
    
    df = pd.DataFrame(results)
    
    print("\n" + "=" * 70)
    print("SUMMARY BY CONFIGURATION")
    print("=" * 70)
    
    summary = df.groupby('config').agg({
        'total_time_ms': 'mean',
        'avg_update_ms': 'mean',
        'l2_error': ['mean', 'std'],
        'profit': 'mean',
    }).round(4)
    
    print(summary.to_string())
    
    print("\n" + "=" * 70)
    print("SPEEDUP vs PGD BASELINE")
    print("=" * 70)
    
    pgd_time = df[df['config'] == 'PGD']['total_time_ms'].mean()
    pgd_error = df[df['config'] == 'PGD']['l2_error'].mean()
    
    print(f"\n{'Config':<15} │ {'Time (ms)':>10} │ {'Speedup':>8} │ {'L2 Error':>10} │ {'Error Gap':>10}")
    print("-" * 70)
    
    for config_name, _, _ in configs:
        subset = df[df['config'] == config_name]
        t = subset['total_time_ms'].mean()
        err = subset['l2_error'].mean()
        speedup = pgd_time / t
        gap = (err - pgd_error) / pgd_error * 100
        print(f"{config_name:<15} │ {t:>10.1f} │ {speedup:>7.2f}x │ {err:>10.4f} │ {gap:>+9.1f}%")
    
    print("\n" + "=" * 70)
    print("PERFORMANCE BY OBSERVATION COUNT")
    print("=" * 70)
    
    for n_obs in observation_counts:
        print(f"\n--- {n_obs} observations ---")
        subset = df[df['n_obs'] == n_obs]
        pgd_subset = subset[subset['config'] == 'PGD']
        pgd_t = pgd_subset['total_time_ms'].mean()
        
        for config_name, _, _ in configs:
            config_subset = subset[subset['config'] == config_name]
            t = config_subset['total_time_ms'].mean()
            err = config_subset['l2_error'].mean()
            speedup = pgd_t / t
            print(f"  {config_name:<15}: {t:>8.1f}ms ({speedup:>5.2f}x), L2={err:.4f}")
    
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = os.path.join(output_dir, 'full_optimization_test.csv')
    df.to_csv(csv_path, index=False)
    
    log_path = os.path.join(output_dir, 'full_optimization_test.log')
    with open(log_path, 'w') as f:
        f.write("FULL OPTIMIZATION TEST RESULTS\n")
        f.write("=" * 70 + "\n\n")
        f.write(summary.to_string())
        f.write("\n\n")
        f.write("Speedup Analysis:\n")
        f.write("-" * 70 + "\n")
        for config_name, _, _ in configs:
            subset = df[df['config'] == config_name]
            t = subset['total_time_ms'].mean()
            err = subset['l2_error'].mean()
            speedup = pgd_time / t
            gap = (err - pgd_error) / pgd_error * 100
            f.write(f"{config_name:<15}: {t:>10.1f}ms, {speedup:>5.2f}x speedup, L2={err:.4f} ({gap:+.1f}%)\n")
    
    print(f"\n\nResults saved to:\n  {csv_path}\n  {log_path}")
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()

