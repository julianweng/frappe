# Scalable, Safe, and Adaptive Opponent Modeling

Implementation of the FMAP (Full Max A Posteriori) algorithm from Ganzfried (2025) with novel extensions for improved scalability, safety, and adaptability in imperfect-information games.

## Project Overview

This project reproduces the FMAP algorithm and extends it with:

1. **Scalability Extension**: Frank-Wolfe optimizer that replaces QP projection with LP-based Linear Minimization Oracle
2. **Safety Extensions**: (i) an NE–BR mixture heuristic and (ii) true Restricted Nash Response (RNR; Johanson et al.) to balance exploitation with robustness
3. **Adaptability Extension**: Discounted likelihood for learning against non-stationary opponents

## Features

- **Clean Architecture**: Modular design separating game logic, optimization, agents, and simulation
- **Multiple Solvers**: Both Projected Gradient Descent (PGD) and Frank-Wolfe (FW) implementations
- **Flexible Configuration**: Easy-to-configure experiments via Python API or script
- **Efficient Implementation**: Optimized with Gurobi (with scipy fallback) and vectorized NumPy operations

## Installation

### Prerequisites

- Python 3.12+
- `uv` package manager
- Gurobi (optional, but recommended for performance)

### Setup with uv

```bash
# Clone the repository
git clone <repository-url>
cd opponent-modeling

# Install dependencies (without Gurobi)
uv sync

# OR install with Gurobi support (requires license)
uv sync --extra gurobi

# Activate virtual environment
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows
```

### Gurobi Setup (Optional but Recommended)

For best performance, install Gurobi:

1. Download from https://www.gurobi.com/downloads/
2. Get a free academic license or trial license: https://www.gurobi.com/academia/academic-program-and-licenses/
3. Install with: `uv add gurobipy` or `pip install gurobipy`

**Note**: If Gurobi is not available, the system will automatically fall back to scipy solvers (slower but no license required).

## Quick Start

### 1. Fast Validation
For a quick test of the algorithms (approx. 30 seconds):
```bash
uv run examples/run_experiment.py quick
```

### 2. Paper Experiment (Full Scale)
To run the full paper experiment with 100 opponents and 3000 iterations (several hours):
```bash
uv run examples/run_experiment.py final
```

### Python API Example
```python
from src.core import get_kuhn_game
from src.agents import FMAPAgent, DirichletOpponent
from src.engine import ExperimentRunner, ExperimentConfig

# Get Kuhn Poker game
game = get_kuhn_game()

# Configure experiment
config = ExperimentConfig(
    num_iterations=1000,
    num_opponents=50,
    agent_configs=[
        {'name': 'FMAP-PGD', 'solver_type': 'pgd'},
        {'name': 'FMAP-FW', 'solver_type': 'fw'},
    ]
)

# Run and save
runner = ExperimentRunner(config)
results_df = runner.run_experiment(verbose=True)
runner.save_results('results.csv')
```

## Project Structure

```
opponent-modeling/
├── src/
│   ├── core/              # Game Definitions
│   │   ├── __init__.py
│   │   ├── kuhn.py        # Kuhn Poker matrices (E, F, A)
│   │   └── utils.py       # Observability maps & utilities
│   ├── optimization/      # Optimization Algorithms
│   │   ├── __init__.py
│   │   ├── base.py        # Abstract optimizer base class
│   │   ├── gradients.py   # Gradient computations
│   │   ├── pgd.py         # Projected Gradient Descent
│   │   └── fw.py          # Frank-Wolfe
│   ├── agents/            # Agent Implementations
│   │   ├── __init__.py
│   │   ├── opponent.py    # Static & switching opponents
│   │   ├── fmap.py        # FMAP learning agent
│   │   └── nash.py        # Nash equilibrium agent
    └── engine/            # Simulation Engine
        ├── __init__.py
        ├── dealer.py      # Card dealing & game logic
        └── runner.py      # Experiment orchestration
├── examples/              # Example scripts including experiments
├── pyproject.toml         # Project configuration
└── README.md
```

## Acknowledgments

- Based on the FMAP algorithm by Sam Ganzfried (2025)
- Kuhn Poker implementation follows the sequence-form representation from Koller, Megiddo, and von Stengel (1994)
- Built with NumPy, SciPy, and Gurobi
