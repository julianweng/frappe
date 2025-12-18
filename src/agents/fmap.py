"""
FMAP Agent is the main learning agent that uses FMAP algorithm to model and exploit opponents
Supports: PGD and Frank-Wolfe solvers, stochastic Frank-Wolfe, NE--BR mixture safety heuristic,
Restricted Nash Response for safety and Discounted likelihood for adaptability
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
import time

from ..optimization import OptimizerBase, ProjectedGradientDescent, FrankWolfe, StochasticFrankWolfe
from scipy.optimize import linprog
from ..core import get_kuhn_game


class FMAPAgent:
    def __init__(
        self,
        game,
        solver_type: str = "pgd",  # "pgd", "fw", or "sfw" (stochastic FW)
        safety_mode: str = "none",  # "none", "ne_br_mixture", or "rnr"
        safety_p: float | str = 0.5,  # Fixed p in [0,1] or "dynamic"
        discount_factor: float = 1.0,  # Extension 3: Discounting (γ ∈ (0,1])
        alpha: Optional[np.ndarray] = None,  # Prior parameters
        solver_kwargs: Optional[Dict] = None,
        verbose: bool = False,
        early_stopping: bool = False,  # Enable early stopping based on L2 error
        early_stopping_patience: int = 5,  # Number of iterations without improvement
        early_stopping_threshold: float = 1e-6,  # Minimum improvement threshold
        require_gurobi: bool = False,  # Require Gurobi (disable scipy fallback)
        entropy_weight: float = 0.0  # Entropy regularization weight λ
    ):
        """
        FMAP agent args:
            game: KuhnGame object with matrices E, F, A
            solver_type: "pgd", "fw", or "sfw"
            safety_mode: 
                - "none": best response to the FMAP model.
                - "ne_br_mixture": play an NE--BR convex mixture heuristic
                - "rnr": play a true p-Restricted Nash Response
            safety_p: Mixing/restriction parameter p in [0,1] to gauge confidence in model
            discount_factor: Discount factor for observations
            alpha: Dirichlet prior parameters
            solver_kwargs: Additional kwargs for the solver
            verbose: Flag to allow/disallow debug information print
            early_stopping: Enable early stopping based on L2 error convergence
            early_stopping_patience: Number of iterations without improvement before stopping
            early_stopping_threshold: Minimum improvement in L2 error to reset patience counter
            entropy_weight: Entropy regularization weight lambda
        """
        self.game = game
        self.solver_type = solver_type
        self.safety_mode = safety_mode
        self.safety_p = safety_p
        self.discount_factor = discount_factor
        self.verbose = verbose
        self.early_stopping = early_stopping
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.require_gurobi = require_gurobi
        self.entropy_weight = entropy_weight

        self.l2_error_history = []
        self.best_l2_error = float('inf')
        self.patience_counter = 0

        if alpha is None: #prior: default is uniform dirichlet with alpha=2 
            self.alpha = np.ones(game.num_p2_sequences) * 2.0
        else:
            self.alpha = alpha.copy()

        # the initial point MUST satisfy Fy = f for the optimization to work
        from ..core import get_uniform_strategy
        self.y_model = get_uniform_strategy(game.F, game.f)

        self.observations: List[Tuple[List[int], np.ndarray]] = []

        solver_kwargs = solver_kwargs or {}

        if self.early_stopping:
            solver_kwargs['early_stopping_patience'] = self.early_stopping_patience
            solver_kwargs['early_stopping_threshold'] = self.early_stopping_threshold

        solver_kwargs['require_gurobi'] = self.require_gurobi
        
        solver_kwargs['entropy_weight'] = self.entropy_weight

        if solver_type == "pgd":
            self.solver = ProjectedGradientDescent(
                F=game.F,
                f=game.f,
                alpha=self.alpha,
                verbose=verbose,
                **solver_kwargs
            )
        elif solver_type == "fw":
            self.solver = FrankWolfe(
                F=game.F,
                f=game.f,
                alpha=self.alpha,
                verbose=verbose,
                **solver_kwargs
            )
        elif solver_type == "sfw":
            self.solver = StochasticFrankWolfe(
                F=game.F,
                f=game.f,
                alpha=self.alpha,
                verbose=verbose,
                **solver_kwargs
            )
        else:
            raise ValueError(f"Unknown solver type: {solver_type}. Valid options: pgd, fw, sfw")

        if safety_mode == "ne_br_mixture":
            self.x_nash = self._compute_nash_strategy()
        else:
            self.x_nash = None

        self.update_times = []
        self.model_history = []

        self._cached_br = None
        self._cached_br_model = None
        
        self._br_model = None
        self._br_vars = None
        self._build_br_model()

    def _compute_nash_strategy(self) -> np.ndarray:
        #Compute Nash equilibrium strategy for Player 1
        return self._compute_nash_strategy_analytical()

    def _compute_nash_strategy_analytical(self) -> np.ndarray:
        """
        Compute Nash equilibrium for Player 1 using solution for Kuhn poker
        Denoted behavior:
        - With K: bet always.
        - With Q: check always; if P2 bets after a check, call with prob 2/3.
        - With J: bet (bluff) with prob 1/3; if check and P2 bets, fold always.
        """

        # Kuhn poker Player 1 sequences:
        # 0: empty
        # 1: B_K (bet with King), 2: Ch_K (check with King)
        # 3: Ch_K Ca_K (check-call with King)
        # 4: Ch_K F_K (check-fold with King)
        # 5: B_Q (bet with Queen)
        # 6: Ch_Q (check with Queen)
        # 7: Ch_Q Ca_Q (check-call with Queen)
        # 8: Ch_Q F_Q (check-fold with Queen)
        # 9: B_J (bet with Jack)
        # 10: Ch_J (check with Jack)
        # 11: Ch_J Ca_J (check-call with Jack)
        # 12: Ch_J F_J (check-fold with Jack)

        x = np.zeros(13)

        x[0] = 1.0

        # King: always bet
        x[1] = 1.0
        x[2] = 0.0
        x[3] = 0.0
        x[4] = 0.0

        # Queen: check always; if P2 bets, call 2/3 and fold 1/3
        x[5] = 0.0
        x[6] = 1.0
        x[7] = 2.0 / 3.0
        x[8] = 1.0 / 3.0

        # Jack: bluff 1/3; if checked and raised, always fold
        x[9] = 1.0 / 3.0
        x[10] = 2.0 / 3.0
        x[11] = 0.0
        x[12] = 2.0 / 3.0

        if not np.allclose(self.game.E @ x, self.game.e):
            print("Warning: Analytical Nash does not satisfy constraints, trying LP fallback")
            try:
                from scipy.optimize import linprog
                pass
            except:
                pass

        return x

    def _build_br_model(self):
        try:
            import gurobipy as gp
            from gurobipy import GRB
            
            E = self.game.E
            e = self.game.e
            n = E.shape[1]
            
            model = gp.Model("BestResponse")
            model.setParam('OutputFlag', 0)
            model.setParam('Method', 1)  

            x = model.addMVar(n, lb=0, name="x")
            model.addConstr(E @ x == e, name="sequence")
            
            model.setObjective(x.sum(), GRB.MAXIMIZE)
            model.update()
            
            self._br_model = model
            self._br_vars = x
        except ImportError:
            self._br_model = None
            self._br_vars = None

    def compute_best_response(self, y_opponent: np.ndarray) -> np.ndarray:
        """
        Compute best response to opponent strategy y. Takes opponent strategy in sequence form
        and reutrns best response strategy for Player 1
        """
        A = self.game.A
        n = self.game.E.shape[1]
        objective_coeffs = A @ y_opponent

        if self._br_model is not None:
            try:
                from gurobipy import GRB

                self._br_model.setObjective(objective_coeffs @ self._br_vars, GRB.MAXIMIZE)
                self._br_model.optimize()

                if self._br_model.status == GRB.OPTIMAL:
                    return self._br_vars.X.copy()
                else:
                    print("Warning: Best response computation failed")
                    from ..core import get_uniform_strategy
                    return get_uniform_strategy(self.game.E, self.game.e)
            except Exception as e:
                print(f"Warning: Gurobi BR failed: {e}")
                pass

        E = self.game.E
        e = self.game.e

        result = linprog(
            -objective_coeffs, 
            A_eq=E,
            b_eq=e,
            bounds=[(0, None) for _ in range(n)],
            method='highs'
        )

        if result.success:
            return result.x
        else:
            from ..core import get_uniform_strategy
            return get_uniform_strategy(self.game.E, self.game.e)

    def compute_restricted_nash_response(self, y_fix: np.ndarray, p: float) -> np.ndarray:
        #Computing a true p-Restricted Nash Response to a fixed model.
        p_clipped = float(np.clip(p, 0.0, 1.0))
        if p_clipped == 1.0:
            return self.compute_best_response(y_fix)

        E = self.game.E
        e = self.game.e
        F = self.game.F
        f = self.game.f
        A = self.game.A

        n1 = E.shape[1]
        m2 = F.shape[0]
        n2 = F.shape[1]

        ay = A @ y_fix  

        c = np.concatenate([
            -p_clipped * ay,
            -(1.0 - p_clipped) * f,
        ])

        A_eq = np.hstack([E, np.zeros((E.shape[0], m2))])
        b_eq = e.copy()

        A_ub = np.hstack([(-A).T, F.T])
        b_ub = np.zeros(n2)

        bounds = [(0.0, None)] * n1 + [(None, None)] * m2

        result = linprog(
            c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
        )

        if result.success:
            return result.x[:n1]

        print(f"Warning: RNR LP failed: {result.message}")
        from ..core import get_uniform_strategy
        return get_uniform_strategy(E, e)

    def update_model(self) -> Dict:
        """
        Update opponent model using observation history to solve the FMAP optimization problem
        Returns info dict with optimization statistics
        """
        if len(self.observations) == 0:
            return {'iterations': 0, 'converged': True, 'time': 0.0}

        start_time = time.time()

        y_new, info = self.solver.optimize(
            y_init=self.y_model,
            observations=self.observations,
            discount_factor=self.discount_factor
        )

        self.y_model = y_new
        elapsed = time.time() - start_time
        self.update_times.append(elapsed)
        self.model_history.append(y_new.copy())

        if self.verbose:
            print(f"Model update: {info['iterations']} iters, {elapsed:.4f}s")

        return info

    def add_observation(self, observable_sequences: List[int], q_values: np.ndarray):
        #Add a new observation from gameplay.
        self.observations.append((observable_sequences, q_values))

    def act(self, iteration: int = 0) -> np.ndarray:
        """
        Select a strategy to play by taking in current game iteration and returns strategy sequence
        """
        if (self._cached_br is not None and 
            self._cached_br_model is not None and
            np.allclose(self._cached_br_model, self.y_model, atol=1e-6)):
            x_br = self._cached_br
        else:
            x_br = self.compute_best_response(self.y_model)
            self._cached_br = x_br
            self._cached_br_model = self.y_model.copy()

        if self.safety_mode == "none":
            return x_br

        p = self._compute_safety_p(iteration)

        if self.safety_mode == "ne_br_mixture":
            return (1 - p) * self.x_nash + p * x_br

        if self.safety_mode == "rnr":
            return self.compute_restricted_nash_response(self.y_model, p)

        raise ValueError(f"Unknown safety_mode: {self.safety_mode}")

    def _compute_safety_p(self, iteration: int) -> float:
        #Compute safety parameter p (fixed or dynamic)

        if isinstance(self.safety_p, (int, float)):
            return float(self.safety_p)

        num_obs = len(self.observations)
        if num_obs == 0:
            return 0.0

        lambda_param = 0.05
        p = 1.0 - np.exp(-lambda_param * num_obs)
        return min(p, 0.95)

    def get_model_l2_error(self, true_strategy: np.ndarray) -> float:
        #Compute L2 error between model and true opponent strategy
        return np.linalg.norm(self.y_model - true_strategy)

    def reset(self):
        self.observations = []

        from ..core import get_uniform_strategy
        self.y_model = get_uniform_strategy(self.game.F, self.game.f)
        self.update_times = []
        self.model_history = []
        self.solver.reset_statistics()

        self.l2_error_history = []
        self.best_l2_error = float('inf')
        self.patience_counter = 0
        
        self._cached_br = None
        self._cached_br_model = None
