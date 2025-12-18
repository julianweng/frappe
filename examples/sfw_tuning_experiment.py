"""
Full experiment of SFW batch_size tuning to compare different SFW batch_size 
configurations against PGD and FW baselines.
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
    solver_type, batch_size, n_obs, opponent_seed, trial = args
    
    np.random.seed(opponent_seed + trial * 1000)
    game = get_kuhn_game()
    opponent = np.random.dirichlet(np.ones(13) * 2)
    
    solver_kwargs = {}
    if solver_type == 'sfw':
        solver_kwargs['batch_size'] = batch_size
    
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
    strategy = agent.act(iteration=n_obs-1)
    profit = np.dot(strategy, np.dot(game.A, opponent))
    
    return {
        'solver': solver_type,
        'batch_size': batch_size,
        'n_obs': n_obs,
        'trial': trial,
        'total_time_ms': total_time * 1000,
        'l2_error': l2_error,
        'profit': profit,
    }


def main():
    print("=" * 70)
    print("FULL SFW TUNING EXPERIMENT")
    print("=" * 70)
    
    configs = [
        ('pgd', None),
        ('fw', None),
        ('sfw', 25),
        ('sfw', 50),
        ('sfw', 100),
        ('sfw', 200),
        ('sfw', 500),
    ]
    
    observation_counts = [100, 300, 500]
    n_opponents = 4
    n_trials = 3
    
    experiments = []
    for solver, batch in configs:
        for n_obs in observation_counts:
            for opp in range(n_opponents):
                for trial in range(n_trials):
                    experiments.append((solver, batch, n_obs, opp, trial))
    
    print(f"Running {len(experiments)} experiments...")
    print(f"Configs: PGD, FW, SFW-25, SFW-50, SFW-100, SFW-200, SFW-500")
    print(f"Observations: {observation_counts}")
    print(f"Opponents: {n_opponents}, Trials: {n_trials}")
    
    n_workers = min(cpu_count(), 8)
    print(f"Using {n_workers} workers...")
    
    start_time = time.time()
    with Pool(n_workers) as pool:
        results = pool.map(run_experiment, experiments)
    
    print(f"Completed in {time.time() - start_time:.1f} seconds")
    
    df = pd.DataFrame(results)
    
    df['label'] = df.apply(
        lambda r: f"SFW-{int(r['batch_size'])}" if r['solver'] == 'sfw' 
                  else r['solver'].upper(), axis=1
    )
    
    print("\n" + "=" * 70)
    print("RESULTS BY SOLVER CONFIGURATION")
    print("=" * 70)
    
    summary = df.groupby('label').agg({
        'total_time_ms': 'mean',
        'l2_error': ['mean', 'std'],
        'profit': 'mean',
    }).round(4)
    
    print(summary.to_string())
    
    print("\n" + "=" * 70)
    print("SPEEDUP AND ACCURACY VS PGD BASELINE")
    print("=" * 70)
    
    pgd_time = df[df['label'] == 'PGD']['total_time_ms'].mean()
    pgd_l2 = df[df['label'] == 'PGD']['l2_error'].mean()
    
    print(f"\n{'Config':<12} │ {'Time (ms)':>10} │ {'Speedup':>8} │ {'L2 Error':>10} │ {'Gap':>8}")
    print("-" * 65)
    
    for label in ['PGD', 'FW', 'SFW-25', 'SFW-50', 'SFW-100', 'SFW-200', 'SFW-500']:
        subset = df[df['label'] == label]
        if len(subset) == 0:
            continue
        t = subset['total_time_ms'].mean()
        l2 = subset['l2_error'].mean()
        speedup = pgd_time / t
        gap = (l2 - pgd_l2) / pgd_l2 * 100
        print(f"{label:<12} │ {t:>10.1f} │ {speedup:>7.1f}x │ {l2:>10.4f} │ {gap:>+7.1f}%")
    
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = os.path.join(output_dir, 'sfw_tuning_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    print("\nExperiment complete!")


if __name__ == '__main__':
    main()


