"""
Experiment Runner to orchestrate experiments comparing different algorithms and configurations
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass, field
import time
from multiprocessing import Pool, cpu_count
import zlib
from functools import partial
from tqdm import tqdm

from ..core import get_kuhn_game
from ..agents import FMAPAgent, NashAgent, DirichletOpponent, SwitchingOpponent, AdversarialOpponent
from .dealer import KuhnDealer


def _simulate_observation_worker(agent_strategy: np.ndarray, opponent_strategy: np.ndarray) -> tuple:
    """
    Simulate an observation from playing against opponent.
    Key idea: q_values are the normalized chance probabilities p_j, NOT the opponent's strategy.
    The chance probability depends on the card deal, not on what the opponent chose to do.
    """
    def _prob(a: float, b: float) -> float:
        denom = a + b
        if denom <= 1e-12:
            return 0.5
        return float(np.clip(a / denom, 0.0, 1.0))
    
    cards = ['K', 'Q', 'J']
    p1_card = np.random.choice(cards)
    remaining = [c for c in cards if c != p1_card]
    p2_card = np.random.choice(remaining)
    
    # P1 sequences: 1=B_K, 2=Ch_K, 5=B_Q, 6=Ch_Q, 9=B_J, 10=Ch_J
    if p1_card == 'K':
        p1_bet_prob = _prob(agent_strategy[1], agent_strategy[2])
    elif p1_card == 'Q':
        p1_bet_prob = _prob(agent_strategy[5], agent_strategy[6])
    else:  # J
        p1_bet_prob = _prob(agent_strategy[9], agent_strategy[10])
    
    p1_bets = np.random.random() < p1_bet_prob
    
    if p1_bets:
        if p2_card == 'Q':
            call_prob = opponent_strategy[1]
            fold_prob = opponent_strategy[2]
        elif p2_card == 'J':
            call_prob = opponent_strategy[5]
            fold_prob = opponent_strategy[6]
        else:  # K
            call_prob = opponent_strategy[9]
            fold_prob = opponent_strategy[10]
        
        total = call_prob + fold_prob
        if total > 1e-10:
            p2_calls = np.random.random() < (call_prob / total)
        else:
            p2_calls = np.random.random() < 0.5
        
        if p2_calls:
            if p2_card == 'Q':
                return [1], np.array([1.0])  
            elif p2_card == 'J':
                return [5], np.array([1.0])
            else:
                return [9], np.array([1.0]) 
        else:
            # P2 folded
            if p1_card == 'K':
                return [2, 6], np.array([0.5, 0.5]) 
            elif p1_card == 'Q':
                return [6, 10], np.array([0.5, 0.5]) 
            else:  # J
                return [2, 10], np.array([0.5, 0.5]) 
    else:
        # P1 checked
        if p2_card == 'Q':
            bet_prob = opponent_strategy[3]
            check_prob = opponent_strategy[4]
        elif p2_card == 'J':
            bet_prob = opponent_strategy[7]
            check_prob = opponent_strategy[8]
        else:  # K
            bet_prob = opponent_strategy[11]
            check_prob = opponent_strategy[12]
        
        total = bet_prob + check_prob
        if total > 1e-10:
            p2_bets = np.random.random() < (bet_prob / total)
        else:
            p2_bets = np.random.random() < 0.5
        
        if p2_bets:
            # P2 bet after P1 check
            if p1_card == 'K':
                p1_call_prob = _prob(agent_strategy[3], agent_strategy[4])
            elif p1_card == 'Q':
                p1_call_prob = _prob(agent_strategy[7], agent_strategy[8])
            else:  # J
                p1_call_prob = _prob(agent_strategy[11], agent_strategy[12])
            
            p1_calls = np.random.random() < p1_call_prob
            
            if p1_calls:
                if p2_card == 'Q':
                    return [3], np.array([1.0])  
                elif p2_card == 'J':
                    return [7], np.array([1.0])  
                else:  # K
                    return [11], np.array([1.0]) 
            else:
                # P1 folded
                if p1_card == 'K':
                    return [3, 7], np.array([0.5, 0.5]) 
                elif p1_card == 'Q':
                    return [7, 11], np.array([0.5, 0.5])  
                else:  # J
                    return [3, 11], np.array([0.5, 0.5]) 
        else:
            # Both checked - showdown
            if p2_card == 'Q':
                return [4], np.array([1.0])  
            elif p2_card == 'J':
                return [8], np.array([1.0])
            else:
                return [12], np.array([1.0])  


def _run_agent_against_opponent_worker(
    opponent_item,  
    agent_config: Dict,
    game,
    num_iterations: int,
    base_seed: int = 0
) -> Dict:
    if isinstance(opponent_item, tuple) and len(opponent_item) == 2:
        opponent_idx, opponent = opponent_item
    else:
        opponent_idx, opponent = -1, opponent_item

    agent_name = str(agent_config.get('name', 'agent'))
    agent_name_seed = zlib.adler32(agent_name.encode("utf-8")) & 0xFFFFFFFF
    seed = (int(base_seed) + (opponent_idx + 1) * 1_000_003 + agent_name_seed) % (2**32 - 1)
    np.random.seed(seed)

    config = agent_config.copy()
    config.pop('name', None)
    agent = FMAPAgent(game=game, **config)

    agent.reset()

    l2_errors = []
    profits = []
    update_times = []

    is_adversarial = isinstance(opponent, AdversarialOpponent)

    for t in range(num_iterations):
        agent_strategy = agent.act(iteration=t)

        if is_adversarial:
            opponent.set_agent_strategy(agent_strategy)

        opponent_strategy = opponent.get_strategy(iteration=t)

        expected_payoff = np.dot(
            agent_strategy,
            np.dot(game.A, opponent_strategy)
        )

        profits.append(expected_payoff)

        observable_seqs, q_values = _simulate_observation_worker(agent_strategy, opponent_strategy)

        agent.add_observation(observable_seqs, q_values)

        start_time = time.time()
        agent.update_model()
        update_time = time.time() - start_time
        update_times.append(update_time)

        l2_error = agent.get_model_l2_error(opponent_strategy)
        l2_errors.append(l2_error)

    return {
        'l2_errors': l2_errors,
        'profits': profits,
        'update_times': update_times,
    }


@dataclass
class ExperimentConfig:
    name: str
    num_iterations: int = 3000
    num_opponents: int = 100
    opponent_alpha: Optional[np.ndarray] = None
    random_state: Optional[int] = None

    opponent_type: str = 'dirichlet'  # 'dirichlet', 'switching', or 'adversarial'
    switch_iteration: int = 100 

    agent_configs: List[Dict] = field(default_factory=list)

    # Benchmarks to include
    include_best_response: bool = True
    include_best_nash: bool = True


class ExperimentRunner:
    """
    Runs experiments comparing FMAP variants and benchmarks
    """

    def __init__(self, config: ExperimentConfig):
        """
        Initialize experiment runner by taking in experiment configuration
        """
        self.config = config
        self.game = get_kuhn_game()
        self.dealer = KuhnDealer(random_state=config.random_state)

        if config.random_state is not None:
            np.random.seed(config.random_state)

        self.results = {}

    def create_agent(self, agent_config: Dict) -> FMAPAgent:
        """
        Create an FMAP agent from configuration
        """
        config = agent_config.copy()
        config.pop('name', None)

        return FMAPAgent(
            game=self.game,
            **config
        )

    def run_against_opponent(
        self,
        agent: FMAPAgent,
        opponent,  
        num_iterations: int
    ) -> Dict:
        """
        Run one agent against one opponent for specified iterations.
        """
        agent.reset()
        
        is_adversarial = isinstance(opponent, AdversarialOpponent)

        l2_errors = []
        profits = []
        cumulative_profit = 0.0
        update_times = []

        for t in range(num_iterations):
            agent_strategy = agent.act(iteration=t)
            
            if is_adversarial:
                opponent.set_agent_strategy(agent_strategy)
            
            opponent_strategy = opponent.get_strategy(iteration=t)

            expected_payoff = np.dot(
                agent_strategy,
                np.dot(self.game.A, opponent_strategy)
            )

            profits.append(expected_payoff)
            cumulative_profit += expected_payoff

            obs_seqs, q_vals = self._simulate_observation(agent_strategy, opponent_strategy)

            agent.add_observation(obs_seqs, q_vals)

            start_time = time.time()
            update_info = agent.update_model()
            update_time = time.time() - start_time
            update_times.append(update_time)

            l2_error = agent.get_model_l2_error(opponent_strategy)
            l2_errors.append(l2_error)

        return {
            'l2_errors': l2_errors,
            'profits': profits,
            'cumulative_profit': cumulative_profit,
            'update_times': update_times,
            'final_l2_error': l2_errors[-1] if l2_errors else np.inf
        }

    def _simulate_observation(
        self,
        agent_strategy: np.ndarray,
        opponent_strategy: np.ndarray
    ) -> tuple:
        return _simulate_observation_worker(agent_strategy, opponent_strategy)

    def compute_benchmark_values(
        self,
        opponent 
    ) -> Dict[str, float]:
       
        opponent_strategy = opponent.get_strategy(iteration=0)

        benchmarks = {}

        if self.config.include_best_response:
            temp_agent = FMAPAgent(self.game, solver_type="pgd")
            best_response = temp_agent.compute_best_response(opponent_strategy)
            br_value = np.dot(
                best_response,
                np.dot(self.game.A, opponent_strategy)
            )
            benchmarks['best_response'] = br_value

        # Best Nash
        if self.config.include_best_nash:
            nash_agent = NashAgent(self.game, player=1)
            nash_value = nash_agent.compute_value_against(opponent_strategy)
            benchmarks['best_nash'] = nash_value

        return benchmarks

    #full experiment
    def run_experiment(self, verbose: bool = True, n_jobs: int = -1) -> pd.DataFrame:
        config = self.config
        if verbose:
            print(f"Running experiment: {config.name}")
            print(f"  Opponents: {config.num_opponents}")
            print(f"  Iterations: {config.num_iterations}")
            print(f"  Agents: {len(config.agent_configs)}")

        if verbose:
            print("Generating opponents...")

        opponents = []
        iterator = tqdm(range(config.num_opponents), desc="  Generating", disable=not verbose)
        for i in iterator:
            if config.opponent_type == 'adversarial':
                opponent = AdversarialOpponent(
                    game=self.game,
                    random_state=config.random_state + i if config.random_state else None
                )
            elif config.opponent_type == 'switching':
                opponent1 = DirichletOpponent(
                    alpha=config.opponent_alpha,
                    num_sequences=self.game.num_p2_sequences,
                    random_state=config.random_state + i * 2 if config.random_state else None,
                    game=self.game
                )
                if config.opponent_alpha is not None:
                    alpha2 = config.opponent_alpha.copy()
                    for j in range(0, len(alpha2), 2):
                        if j + 1 < len(alpha2):
                            alpha2[j], alpha2[j+1] = alpha2[j+1], alpha2[j]
                else:
                    alpha2 = None
                opponent2 = DirichletOpponent(
                    alpha=alpha2,
                    num_sequences=self.game.num_p2_sequences,
                    random_state=config.random_state + i * 2 + 1 if config.random_state else None,
                    game=self.game
                )
                opponent = SwitchingOpponent(
                    strategy1=opponent1.strategy,
                    strategy2=opponent2.strategy,
                    switch_iteration=config.switch_iteration
                )
            else:
                opponent = DirichletOpponent(
                    alpha=config.opponent_alpha,
                    num_sequences=self.game.num_p2_sequences,
                    random_state=config.random_state + i if config.random_state else None,
                    game=self.game
                )
            opponents.append(opponent)

        if n_jobs == -1:
            n_jobs = cpu_count()
        elif n_jobs <= 0:
            n_jobs = 1

        if verbose:
            print(f"Using {n_jobs} parallel workers")
            print(f"Each opponent: {config.num_iterations} iterations")

            if len(opponents) > 0:
                warmup_iters = min(50, config.num_iterations) 
                print(f"\nRunning warmup ({warmup_iters} iterations) to estimate timing...")
                warmup_config = config.agent_configs[0].copy()
                warmup_config.pop('name', None)
                warmup_agent = FMAPAgent(game=self.game, **warmup_config)

                import time
                start = time.time()
                self.run_against_opponent(warmup_agent, opponents[0], warmup_iters)
                warmup_elapsed = time.time() - start

                time_per_iter = warmup_elapsed / warmup_iters
                estimated_per_opponent = time_per_iter * config.num_iterations

                total_opponents = config.num_opponents * len(config.agent_configs)
                est_parallel = (config.num_opponents * estimated_per_opponent * len(config.agent_configs)) / n_jobs

                print(f"  Warmup completed in {warmup_elapsed:.1f}s ({warmup_iters} iterations)")
                print(f"  Estimated time per iteration: ~{time_per_iter:.3f}s")
                print(f"  Estimated time per opponent ({config.num_iterations} iterations): ~{estimated_per_opponent:.1f}s")
                print(f"  Estimated total time (parallel, {n_jobs} workers): ~{est_parallel/60:.1f} minutes")
                print(f"  First progress update expected in: ~{estimated_per_opponent:.1f}s")

                if est_parallel > 1800:  
                    print(f"\n  ⚠️  WARNING: Estimated time is {est_parallel/60:.0f} minutes!")
                    print(f"     Consider using examples/run_quick_experiment.py for faster results")

            if n_jobs > 1:
                print(f"\nNote: Progress updates as each opponent completes")
                print(f"      {n_jobs} opponents processing in parallel")

        all_results = []

        for agent_idx, agent_config in enumerate(config.agent_configs):
            agent_name = agent_config.get('name', f"Agent_{agent_idx}")

            if verbose:
                print(f"\nRunning {agent_name}...")
                print(f"  Starting parallel processing...")

            agent_results = {
                'agent_name': agent_name,
                'l2_errors_avg': [],
                'profits_avg': [],
                'cumulative_profit_avg': [],
                'update_times_avg': [],
            }

            # Run against each opponent (parallelized!)
            if n_jobs == 1:
                opponent_results = []
                iterator = tqdm(opponents, desc=f"  Processing opponents", disable=not verbose)
                for opponent in iterator:
                    agent = self.create_agent(agent_config)
                    result = self.run_against_opponent(
                        agent, opponent, config.num_iterations
                    )
                    opponent_results.append(result)
            else:
                opponent_items = list(enumerate(opponents))
                base_seed = (config.random_state or 0) + agent_idx * 10_000_000
                run_fn = partial(
                    _run_agent_against_opponent_worker,
                    agent_config=agent_config,
                    game=self.game,
                    num_iterations=config.num_iterations,
                    base_seed=base_seed
                )

                with Pool(processes=n_jobs) as pool:
                    if verbose:
                        import sys
                        sys.stdout.flush()  
                        opponent_results = list(tqdm(
                            pool.imap_unordered(run_fn, opponent_items, chunksize=1),
                            total=len(opponent_items),
                            desc=f"  Processing opponents",
                            unit="opponent",
                            smoothing=0.1 
                        ))
                    else:
                        opponent_results = pool.map(run_fn, opponent_items)

            for opp_idx, results in enumerate(opponent_results):
                if opp_idx == 0:
                    for key in ['l2_errors', 'profits', 'update_times']:
                        agent_results[f'{key}_avg'] = np.array(results[key])
                else:
                    for key in ['l2_errors', 'profits', 'update_times']:
                        agent_results[f'{key}_avg'] += np.array(results[key])

            for key in ['l2_errors_avg', 'profits_avg', 'update_times_avg']:
                agent_results[key] = agent_results[key] / config.num_opponents

            agent_results['cumulative_profit_avg'] = np.cumsum(agent_results['profits_avg'])

            all_results.append(agent_results)

            if verbose:
                final_profit = agent_results['profits_avg'][-1]
                final_l2 = agent_results['l2_errors_avg'][-1]
                print(f"  Final profit: {final_profit:.4f}")
                print(f"  Final L2 error: {final_l2:.6f}")

        self.results = all_results

        return self._results_to_dataframe()

    def _results_to_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame"""
        rows = []

        for agent_result in self.results:
            agent_name = agent_result['agent_name']
            num_iters = len(agent_result['l2_errors_avg'])

            for t in range(num_iters):
                rows.append({
                    'agent': agent_name,
                    'iteration': t,
                    'l2_error': agent_result['l2_errors_avg'][t],
                    'profit': agent_result['profits_avg'][t],
                    'cumulative_profit': agent_result['cumulative_profit_avg'][t],
                    'update_time': agent_result['update_times_avg'][t],
                })

        return pd.DataFrame(rows)

    def save_results(self, filepath: str):
        df = self._results_to_dataframe()
        df.to_csv(filepath, index=False)

    def plot_results(self, metric: str = 'profit', filepath: Optional[str] = None):
        import plotly.graph_objects as go

        df = self._results_to_dataframe()

        fig = go.Figure()

        for agent_name in df['agent'].unique():
            agent_df = df[df['agent'] == agent_name]

            fig.add_trace(go.Scatter(
                x=agent_df['iteration'],
                y=agent_df[metric],
                mode='lines',
                name=agent_name
            ))

        fig.update_layout(
            title=f'{self.config.name} - {metric.replace("_", " ").title()}',
            xaxis_title='Iteration',
            yaxis_title=metric.replace('_', ' ').title(),
            hovermode='x unified'
        )

        if filepath:
            fig.write_html(filepath)

        return fig
