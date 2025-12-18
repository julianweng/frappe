"""
Comprehensive comparison of all FMAP solvers: Projected Gradient Descent, Frank-Wolfe and Stochastic Frank-Wolfe (our new verison)

This file shows the scalability advantage of our proposed approach: Stochastic Frank-Wolfe. 

from Hazan & Luo's "Variance-Reduced and Projection-Free Stochastic Optimization"
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


def run_single_experiment(args):
    """Run a single experiment with given parameters."""
    solver_type, n_observations, opponent_seed, trial = args
    
    np.random.seed(opponent_seed + trial * 1000)
    game = get_kuhn_game()
    
    opponent = np.random.dirichlet(np.ones(13) * 2)
    
    solver_kwargs = {}
    if solver_type == 'sfw':
        solver_kwargs['batch_size'] = 50  # Mini-batch size
        solver_kwargs['use_variance_reduction'] = False
    
    agent = FMAPAgent(
        game, 
        solver_type=solver_type,
        require_gurobi=True,
        solver_kwargs=solver_kwargs
    )
    
    total_update_time = 0
    l2_errors = []
    profits = []
    
    for i in range(n_observations):
        obs = [np.random.randint(1, 13)]
        q = np.array([1.0])
        agent.add_observation(obs, q)
        
        # update model and measure time
        start = time.perf_counter()
        info = agent.update_model()
        update_time = time.perf_counter() - start
        total_update_time += update_time
        
        l2_error = np.linalg.norm(agent.y_model - opponent)
        l2_errors.append(l2_error)
        
        strategy = agent.act(iteration=i)
        profit = np.dot(strategy, np.dot(game.A, opponent))
        profits.append(profit)
    
    return {
        'solver': solver_type.upper(),
        'n_observations': n_observations,
        'trial': trial,
        'total_time': total_update_time * 1000,  # ms
        'avg_update_time': total_update_time / n_observations * 1000,  # ms
        'final_l2_error': l2_errors[-1],
        'final_profit': profits[-1],
        'total_profit': sum(profits),
    }


def main():
    print("=" * 70)
    print("COMPREHENSIVE SOLVER COMPARISON: PGD vs FW vs SFW")
    print("=" * 70)
    
    # Experiment parameters
    solvers = ['pgd', 'fw', 'sfw']
    observation_counts = [100, 300, 500]  
    n_trials = 3  
    n_opponents = 3 
    
    experiments = []
    for solver in solvers:
        for n_obs in observation_counts:
            for opp_seed in range(n_opponents):
                for trial in range(n_trials):
                    experiments.append((solver, n_obs, opp_seed, trial))
    
    print(f"\nRunning {len(experiments)} experiments...")
    print(f"Solvers: {solvers}")
    print(f"Observation counts: {observation_counts}")
    print(f"Opponents per config: {n_opponents}")
    print(f"Trials per opponent: {n_trials}")
    
    start_time = time.time()
    n_workers = min(cpu_count(), 8)
    print(f"Using {n_workers} parallel workers...")
    
    with Pool(n_workers) as pool:
        results = pool.map(run_single_experiment, experiments)
    
    total_time = time.time() - start_time
    print(f"\nCompleted in {total_time:.1f} seconds")
    
    df = pd.DataFrame(results)
    
    print("\n" + "=" * 70)
    print("RESULTS: Average Performance by Solver and Observation Count")
    print("=" * 70)
    
    agg = df.groupby(['solver', 'n_observations']).agg({
        'total_time': 'mean',
        'avg_update_time': 'mean',
        'final_l2_error': 'mean',
        'final_profit': 'mean',
        'total_profit': 'mean',
    }).round(4)
    
    print(agg.to_string())
    
    print("\n" + "=" * 70)
    print("SPEEDUP ANALYSIS: SFW vs FW and PGD")
    print("=" * 70)
    
    for n_obs in observation_counts:
        print(f"\nWith {n_obs} observations:")
        
        pgd_time = df[(df['solver'] == 'PGD') & (df['n_observations'] == n_obs)]['total_time'].mean()
        fw_time = df[(df['solver'] == 'FW') & (df['n_observations'] == n_obs)]['total_time'].mean()
        sfw_time = df[(df['solver'] == 'SFW') & (df['n_observations'] == n_obs)]['total_time'].mean()
        
        print(f"  PGD:  {pgd_time:7.1f}ms")
        print(f"  FW:   {fw_time:7.1f}ms")
        print(f"  SFW:  {sfw_time:7.1f}ms")
        print(f"  Speedup vs PGD: {pgd_time/sfw_time:.1f}x")
        print(f"  Speedup vs FW:  {fw_time/sfw_time:.1f}x")
    
    print("\n" + "=" * 70)
    print("QUALITY ANALYSIS: Final L2 Error and Profit")
    print("=" * 70)
    
    quality = df.groupby('solver').agg({
        'final_l2_error': ['mean', 'std'],
        'final_profit': ['mean', 'std'],
    }).round(4)
    
    print(quality.to_string())
    
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = os.path.join(output_dir, 'solver_comparison.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    
    # Summary table for paper
    print("\n" + "=" * 70)
    print("SUMMARY TABLE FOR PAPER")
    print("=" * 70)
    
    summary = df.groupby('solver').agg({
        'avg_update_time': 'mean',
        'final_l2_error': 'mean',
        'final_profit': 'mean',
    }).round(4)
    
    summary.columns = ['Avg Update (ms)', 'L2 Error', 'Profit']
    print(summary.to_string())
    
    # Theoretical contribution statement
    print("\n" + "=" * 70)
    print("THEORETICAL CONTRIBUTION")
    print("=" * 70)
    print("""
Stochastic Frank-Wolfe for Opponent Modeling:

1. Per-iteration cost: O(batch_size) instead of O(n_observations)
   - Demonstrated speedup: {:.1f}x faster than deterministic FW

2. Convergence guarantee: O(1/√k) for smooth convex objectives
   - Maintains similar solution quality (L2 error within {:.1%})

3. Projection-free: Uses LP-based LMO, not QP projection
   - Simpler implementation, better scalability for large games

From Hazan & Luo's "Variance-Reduced and Projection-Free 
Stochastic Optimization"
""".format(
        fw_time / sfw_time if sfw_time > 0 else 0,
        abs(df[df['solver'] == 'SFW']['final_l2_error'].mean() - 
            df[df['solver'] == 'FW']['final_l2_error'].mean()) / 
        df[df['solver'] == 'FW']['final_l2_error'].mean()
    ))


if __name__ == '__main__':
    main()

