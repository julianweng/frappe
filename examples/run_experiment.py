"""
Running FMAP experiments with presets:
- quick: Fast testing
- final: Full paper experiment (3000 iterations, 100 opponents)
- rnr-worst-case: Tests true RNR tradeoffs vs adversarial and typical opponents
Run: uv run examples/run_experiment.py <flag>
"""

import sys
import numpy as np
from src.core import get_kuhn_game
from src.engine import ExperimentRunner, ExperimentConfig

PRESETS = {
    'quick': {
        'name': 'Quick Test',
        'num_iterations': 100,
        'num_opponents': 10,
        'opponent_alpha': None,
        'description': 'Fast integration test (~30 seconds)',
        'agents': [
            {'name': 'FMAP-PGD', 'solver_type': 'pgd', 'safety_mode': 'none', 'discount_factor': 1.0},
            {'name': 'FMAP-FW', 'solver_type': 'fw', 'safety_mode': 'none', 'discount_factor': 1.0},
        ],
        'include_baselines': False,
    },
    'final': {
        'name': 'Full Paper Experiment',
        'num_iterations': 3000,
        'num_opponents': 100,
        'opponent_alpha': None,
        'description': 'Full paper experiment (3000 iterations, 100 opponents, several hours)',
        'agents': [
            {'name': 'FMAP-PGD', 'solver_type': 'pgd', 'safety_mode': 'none', 'discount_factor': 1.0},
            {'name': 'FMAP-FW', 'solver_type': 'fw', 'safety_mode': 'none', 'discount_factor': 1.0,
             'solver_kwargs': {'monotone_backtracking': False},},
            {
                'name': 'FMAP-SFW-100',
                'solver_type': 'sfw',
                'safety_mode': 'none',
                'discount_factor': 1.0,
                'solver_kwargs': {'batch_size': 100, 'use_variance_reduction': False},
            },
            {
                'name': 'FMAP-SFW-SVRG-100',
                'solver_type': 'sfw',
                'safety_mode': 'none',
                'discount_factor': 1.0,
                'solver_kwargs': {'batch_size': 100, 'use_variance_reduction': True, 'variance_reduction_freq': 10},
            },
            {
                'name': 'FMAP-SFW-200',
                'solver_type': 'sfw',
                'safety_mode': 'none',
                'discount_factor': 1.0,
                'solver_kwargs': {'batch_size': 200, 'use_variance_reduction': False},
            },
            {
                'name': 'FMAP-SFW-SVRG-200',
                'solver_type': 'sfw',
                'safety_mode': 'none',
                'discount_factor': 1.0,
                'solver_kwargs': {'batch_size': 200, 'use_variance_reduction': True, 'variance_reduction_freq': 10},
            },
        ],
        'include_baselines': True,
    },
    'rnr-worst-case': {
        'name': 'True RNR Worst-Case Safety',
        'num_iterations': 100,
        'num_opponents': 30,
        'scenarios': [
            {
                'id': 'worst_case',
                'label': 'Worst-Case (Adversarial Best Response)',
                'opponent_type': 'adversarial',
                'opponent_alpha': None,
            },
            {
                'id': 'typical',
                'label': 'Typical (Dirichlet α=2)',
                'opponent_type': 'dirichlet',
                'opponent_alpha': None,
            },
        ],
        'opponent_type': 'adversarial',
        'opponent_alpha': None,
        'description': 'Tests true Johanson RNR worst-case guarantees against adversarial opponents.',
        'agents': [
            {
                'name': 'SFW-SVRG Pure Exploit',
                'solver_type': 'sfw',
                'safety_mode': 'none',
                'discount_factor': 1.0,
                'solver_kwargs': {'batch_size': 100, 'use_variance_reduction': True, 'variance_reduction_freq': 10},
            },
            {
                'name': 'SFW-SVRG NE–BR Mixture (p=0.5)',
                'solver_type': 'sfw',
                'safety_mode': 'ne_br_mixture',
                'safety_p': 0.5,
                'discount_factor': 1.0,
                'solver_kwargs': {'batch_size': 100, 'use_variance_reduction': True, 'variance_reduction_freq': 10},
            },
            {
                'name': 'SFW-SVRG True RNR (p=0.5)',
                'solver_type': 'sfw',
                'safety_mode': 'rnr',
                'safety_p': 0.5,
                'discount_factor': 1.0,
                'solver_kwargs': {'batch_size': 100, 'use_variance_reduction': True, 'variance_reduction_freq': 10},
            },
            {
                'name': 'SFW-SVRG Pure Nash (p=0)',
                'solver_type': 'sfw',
                'safety_mode': 'rnr',
                'safety_p': 0.0,
                'discount_factor': 1.0,
                'solver_kwargs': {'batch_size': 100, 'use_variance_reduction': True, 'variance_reduction_freq': 10},
            },
        ],
        'include_baselines': False,
    },
}


def _build_config_for_preset(game, preset, override: dict | None = None) -> ExperimentConfig:
    override = override or {}

    opponent_alpha = override.get('opponent_alpha', preset.get('opponent_alpha'))
    if opponent_alpha is None:
        opponent_alpha = np.ones(game.num_p2_sequences) * 2.0

    return ExperimentConfig(
        name=override.get('name', preset['name']),
        num_iterations=override.get('num_iterations', preset['num_iterations']),
        num_opponents=override.get('num_opponents', preset['num_opponents']),
        opponent_alpha=opponent_alpha,
        opponent_type=override.get('opponent_type', preset.get('opponent_type', 'dirichlet')),
        switch_iteration=override.get('switch_iteration', preset.get('switch_iteration', 100)),
        agent_configs=override.get('agents', preset['agents']),
        include_best_response=override.get('include_baselines', preset['include_baselines']),
        include_best_nash=override.get('include_baselines', preset['include_baselines']),
        random_state=override.get('random_state', 42),
    )


def _summarize_results(results_df):
    summary = {}
    for agent_name in results_df['agent'].unique():
        agent_df = results_df[results_df['agent'] == agent_name]
        summary[agent_name] = {
            'final_profit': float(agent_df['profit'].iloc[-1]),
            'final_l2': float(agent_df['l2_error'].iloc[-1]),
            'total_profit': float(agent_df['cumulative_profit'].iloc[-1]),
            'avg_update_time': float(agent_df['update_time'].mean()),
        }
    return summary


def _run_and_save(
    config: ExperimentConfig,
    output_prefix: str,
    preset_name: str,
    description: str,
    *,
    show_safety_analysis: bool = True,
):
    print(f"\nConfiguration:")
    print(f"  Description: {description}")
    print(f"  Opponents: {config.num_opponents}")
    print(f"  Iterations per opponent: {config.num_iterations}")
    print(f"  Algorithms: {len(config.agent_configs)}")

    runner = ExperimentRunner(config)

    print("\n" + "=" * 70)
    print("Running experiment...")
    print("=" * 70)

    results_df = runner.run_experiment(verbose=True)

    import os
    os.makedirs('results', exist_ok=True)

    print(f"\nSaving results to {output_prefix}_*.csv/html...")
    runner.save_results(f'{output_prefix}_results.csv')
    runner.plot_results(metric='profit', filepath=f'{output_prefix}_profit.html')
    runner.plot_results(metric='l2_error', filepath=f'{output_prefix}_convergence.html')
    runner.plot_results(metric='cumulative_profit', filepath=f'{output_prefix}_cumulative.html')

    print("\n" + "=" * 70)
    print("Summary Statistics")
    print("=" * 70)

    for agent_name, s in _summarize_results(results_df).items():
        print(f"\n{agent_name}:")
        print(f"  Final Profit: {s['final_profit']:.4f}")
        print(f"  Final L2 Error: {s['final_l2']:.6f}")
        print(f"  Total Profit: {s['total_profit']:.2f}")
        print(f"  Avg Update Time: {s['avg_update_time']*1000:.2f} ms")

    if show_safety_analysis and ('mix' in preset_name or preset_name.startswith('rnr-')) and 'worst-case' in preset_name:
        from scipy.optimize import linprog

        print("\n" + "=" * 70)
        print("Safety-Specific Analysis")
        print("=" * 70)
        print("\nTheoretical Worst-Case Analysis:")
        print("(Value when opponent PERFECTLY best-responds to our strategy)")
        print()

        game = get_kuhn_game()
        for agent_config in config.agent_configs:
            agent_name = agent_config['name']
            cfg = {k: v for k, v in agent_config.items() if k != 'name'}
            from src.agents import FMAPAgent
            agent = FMAPAgent(game=game, **cfg)

            x = agent.act(iteration=0)
            c = game.A.T @ x
            result = linprog(c, A_eq=game.F, b_eq=game.f, bounds=[(0, None)]*13, method='highs')
            if result.success:
                y_br = result.x
                worst_case = x @ game.A @ y_br
                print(f"  {agent_name:25s}: {worst_case:+.4f}")

        print()
        print("KEY INSIGHT: Lower (less negative) is better for worst-case.")
        print("  - Pure Nash has the BEST worst-case guarantee (minimax optimal)")
        print("  - Pure Exploit has the WORST worst-case")
        print("  - True RNR provides a tunable tradeoff between exploitation and safety")

    return results_df


def run_standard_experiment(preset_name):
    """Run a standard experiment with a given preset."""
    if preset_name not in PRESETS:
        print(f"Error: Unknown preset '{preset_name}'")
        print(f"\nAvailable presets:")
        for name, config in PRESETS.items():
            print(f"  - {name:20s}: {config['description']}")
        sys.exit(1)

    preset = PRESETS[preset_name]

    print("=" * 70)
    print(f"FMAP Experiment: {preset['name']}")
    print("=" * 70)

    game = get_kuhn_game()

    # Multi-scenario presets for tradeoff comparisons
    scenarios = preset.get('scenarios')
    if scenarios:
        scenario_results = {}
        for scenario in scenarios:
            scenario_name = f"{preset['name']} — {scenario['label']}"
            config = _build_config_for_preset(game, preset, override={
                'name': scenario_name,
                'opponent_type': scenario.get('opponent_type'),
                'opponent_alpha': scenario.get('opponent_alpha'),
            })

            print("\n" + "=" * 70)
            print(f"Scenario: {scenario['label']}")
            print("=" * 70)

            output_prefix = f"results/{preset_name}_{scenario['id']}"
            results_df = _run_and_save(
                config,
                output_prefix,
                preset_name,
                f"{preset['description']} Scenario: {scenario['label']}",
                show_safety_analysis=(scenario['id'] == 'worst_case'),
            )
            scenario_results[scenario['id']] = _summarize_results(results_df)

        if len(scenario_results) >= 2:
            print("\n" + "=" * 70)
            print("Scenario Comparison (Final Profit / Worst-Case Safety Tradeoff)")
            print("=" * 70)

            agent_names = list(next(iter(scenario_results.values())).keys())
            for agent_name in agent_names:
                parts = []
                for scenario in scenarios:
                    s = scenario_results[scenario['id']][agent_name]
                    parts.append(f"{scenario['id']}: profit={s['final_profit']:+.4f}, total={s['total_profit']:+.2f}")
                print(f"{agent_name}: " + " | ".join(parts))
    else:
        config = _build_config_for_preset(game, preset)
        output_prefix = f"results/{preset_name}"
        _run_and_save(config, output_prefix, preset_name, preset['description'])

    print("\n" + "=" * 70)
    print("Experiment complete!")
    if preset.get('scenarios'):
        print(f"Results saved to results/{preset_name}_*")
    else:
        print(f"Results saved to {output_prefix}_*.csv/html")
    print("=" * 70)


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python examples/run_experiment.py <preset>")
        print("\nAvailable presets:")
        for name, config in PRESETS.items():
            print(f"  - {name:20s}: {config['description']}")
        sys.exit(1)

    preset_name = sys.argv[1]
    run_standard_experiment(preset_name)


if __name__ == "__main__":
    main()
