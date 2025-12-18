"""
Frank-Wolfe optimizer for FMAP: implements Frank-Wolfe algorithm that replaces expensive QP projection
with LP-based Linear Minimization Oracle (LMO).
"""

import numpy as np
from typing import List, Tuple, Optional
try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False
    from scipy.optimize import linprog

from .base import OptimizerBase


class FrankWolfe(OptimizerBase):
    def __init__(
        self,
        F: np.ndarray,
        f: np.ndarray,
        alpha: np.ndarray,
        step_size_rule: str = "adaptive",  
        constant_step_size: float = 0.1,
        use_away_steps: bool = False,  
        step_size_safety: float = 0.9,
        epsilon: float = 1e-6,
        monotone_backtracking: bool = True,
        backtrack_beta: float = 0.5,
        backtrack_iters: int = 5,
        monotone_tolerance: float = 1e-6,
        use_gurobi: bool = True,
        gurobi_time_limit: Optional[float] = None, 
        gurobi_mip_gap: float = 1e-4, 
        tolerance: float = 0.5,  
        max_iterations: int = 30,  
        early_stopping_patience: int = 5, 
        early_stopping_threshold: float = 1e-3,  
        **kwargs
    ):
        """
        FW Args:
            F: Constraint matrix
            f: Constraint vector
            alpha: Prior parameters
            step_size_rule: How to choose step size y_k
                - "adaptive": Duality-gap based with curvature estimate (default)
                - "line_search": Exact line search
                - "2/(k+2)": Standard FW step size
                - "constant": Fixed step size
                - "damped": Adaptive with additional damping for boundary safety
            constant_step_size: Step size if using constant rule
            use_away_steps: Pairwise to remove mass from bad atoms
            step_size_safety: Fraction of feasible step to allow 
            epsilon: Strictly positive lower bound to stay in the interior
            monotone_backtracking: If true, shrink step when objective increases
            backtrack_beta: Multiplicative shrink factor for backtracking
            backtrack_iters: Max backtracking iterations
            monotone_tolerance: Allowed objective increase before shrinking
            use_gurobi: Whether to use Gurobi or scipy
            gurobi_time_limit: Time limit for each Gurobi LMO solve 
            gurobi_mip_gap: MIP gap tolerance for Gurobi optimization
            early_stopping_patience: Stop if no improvement for this many iterations 
            early_stopping_threshold: Minimum objective improvement to reset patience
        """
        super().__init__(
            F, f, alpha,
            tolerance=tolerance,
            max_iterations=max_iterations,
            early_stopping_patience=early_stopping_patience,
            early_stopping_threshold=early_stopping_threshold,
            epsilon=epsilon,
            **kwargs
        )

        self.step_size_rule = step_size_rule
        self.constant_step_size = constant_step_size
        self.use_away_steps = use_away_steps
        self.step_size_safety = step_size_safety
        self.monotone_backtracking = monotone_backtracking
        self.backtrack_beta = backtrack_beta
        self.backtrack_iters = backtrack_iters
        self.monotone_tolerance = monotone_tolerance
        self.use_gurobi = use_gurobi and GUROBI_AVAILABLE
        
        if self.require_gurobi and not self.use_gurobi:
            raise RuntimeError(
                "Gurobi is required for Frank-Wolfe but not available. "
                "Install Gurobi or set require_gurobi=False to use scipy fallback."
            )
        self.gurobi_time_limit = gurobi_time_limit
        self.gurobi_mip_gap = gurobi_mip_gap
        
        self.last_gap = float('inf')
        self.last_step_size = None
        self.last_lmo_solution = None 
        self.last_gradient = None  

        if self.use_gurobi:
            self._build_gurobi_model()
        else:
            self.gurobi_model = None

    def _build_gurobi_model(self):
        if not GUROBI_AVAILABLE:
            print("Warning: Gurobi not available, falling back to scipy")
            self.use_gurobi = False
            self.gurobi_model = None
            return

        try:
            n = self.F.shape[1] 

            model = gp.Model("FW_LMO")
            model.setParam('OutputFlag', 0) 
            model.setParam('Method', 1)  

            if self.gurobi_time_limit is not None:
                model.setParam('TimeLimit', self.gurobi_time_limit)
            model.setParam('OptimalityTol', 1e-6) 

            s_vars = model.addMVar(n, lb=self.epsilon, name="s")

            model.addConstr(self.F @ s_vars == self.f, name="sequence_constraints")

            model.setObjective(0 * s_vars.sum(), GRB.MINIMIZE)

            model.update()
            self.gurobi_model = model
            self.gurobi_vars = s_vars

        except Exception as e:
            print(f"Warning: Failed to build Gurobi model: {e}")
            print("Falling back to scipy")
            self.use_gurobi = False
            self.gurobi_model = None

    def _lmo_gurobi(self, gradient: np.ndarray) -> np.ndarray:
        """
        Solve Linear Minimization Oracle using Gurobi by taking in current gradient to return optimal vertex
        """
        try:
            model = self.gurobi_model
            s_vars = self.gurobi_vars

            model.setObjective(gradient @ s_vars, GRB.MINIMIZE)

            if self.last_lmo_solution is not None:
                for i, val in enumerate(self.last_lmo_solution):
                    s_vars[i].Start = val

            model.optimize()

            if model.status == GRB.OPTIMAL:
                self.last_lmo_solution = s_vars.X.copy()
                self.last_gradient = gradient.copy()
                return s_vars.X.copy()
            elif model.status == GRB.TIME_LIMIT:
                if model.SolCount > 0:
                    return s_vars.X.copy()
                else:
                    print(f"Warning: Gurobi LMO time limit reached without feasible solution")
                    n = len(gradient)
                    s = np.ones(n) * self.epsilon
                    return s / np.sum(s) * np.sum(self.f)
            else:
                print(f"Warning: Gurobi LMO failed with status {model.status}")
                n = len(gradient)
                s = np.ones(n) * self.epsilon
                return s / np.sum(s) * np.sum(self.f)

        except Exception as e:
            print(f"Warning: Gurobi LMO failed: {e}")
            n = len(gradient)
            s = np.ones(n) * self.epsilon
            return s / np.sum(s) * np.sum(self.f)

    def _lmo_scipy(self, gradient: np.ndarray) -> np.ndarray:
        from scipy.optimize import linprog

        c = gradient

        A_eq = self.F
        b_eq = self.f

        bounds = [(0, None) for _ in range(len(gradient))]

        x0 = None
        if self.last_lmo_solution is not None and self.last_gradient is not None:
            grad_cos_sim = np.dot(gradient, self.last_gradient) / (
                np.linalg.norm(gradient) * np.linalg.norm(self.last_gradient) + 1e-12
            )
            if grad_cos_sim > 0.8:  
                x0 = self.last_lmo_solution

        result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs', x0=x0)

        if result.success:
            s = np.maximum(result.x, self.epsilon)
            self.last_lmo_solution = s.copy()
            self.last_gradient = gradient.copy()
            return s
        else:
            if False: 
                print(f"Warning: scipy linprog failed - {result.message}, status={result.status}")
            n = len(gradient)
            s = np.ones(n) * self.epsilon
            s = s / np.sum(s) * np.sum(self.f)
            self.last_lmo_solution = s.copy()
            self.last_gradient = gradient.copy()
            return s

    def lmo(self, gradient: np.ndarray) -> np.ndarray:
        """
        Linear Minimization Oracle: solve min <gradient, s> over feasible set.
        """
        if self.use_gurobi and self.gurobi_model is not None:
            return self._lmo_gurobi(gradient)
        else:
            return self._lmo_scipy(gradient)

    def _max_feasible_step(self, y_current: np.ndarray, direction: np.ndarray) -> float:
        """
        Compute the largest step we can take along `direction` while staying in the interior.
        """
        negative = direction < 0
        if np.any(negative):
            return float(np.min((y_current[negative] - self.epsilon) / (-direction[negative] + 1e-16)))
        return 1.0

    def compute_step_size(
        self,
        iteration: int,
        y_current: np.ndarray,
        s: np.ndarray,
        gradient: np.ndarray,
        observations: List[Tuple[List[int], np.ndarray]],
        discount_weights: np.ndarray = None,
        precomputed: dict = None,
        direction: np.ndarray = None,
        max_step: float = 1.0
    ) -> float:
        """
        Compute step size y_k for Frank-Wolfe update
        Takes args:
            iteration: Current iteration number 
            y_current: Current iterate
            s: LMO solution
            gradient: Current gradient
            observations: Observation history
            direction: Search direction 
            max_step: Maximum feasible step that keeps us inside the polytope
        """
        if direction is None:
            direction = s - y_current

        max_step = max_step if max_step > 0 else 0.0
        safe_cap = max_step * self.step_size_safety

        if self.step_size_rule == "2/(k+2)":
            gamma = 2.0 / (iteration + 2)
            return min(gamma, safe_cap)

        elif self.step_size_rule == "adaptive":
            d = direction
            d_norm_sq = np.dot(d, d)
            if d_norm_sq < 1e-12:
                return 1.0
            
            gap = -np.dot(gradient, d) 
            if gap <= 0:
                return 0.0
            min_y = float(np.min(y_current))
            min_y = max(min_y, self.epsilon)
            if discount_weights is not None:
                effective_samples = float(np.sum(discount_weights))
            else:
                effective_samples = float(len(observations)) if observations is not None else 0.0

            prior_scale = float(np.max(np.abs(self.alpha - 1.0))) if self.alpha is not None else 1.0
            L_estimate = (prior_scale + effective_samples) / (min_y * min_y)
            if self.entropy_weight > 0:
                L_estimate += float(self.entropy_weight) / min_y
            gamma = min(gap / (L_estimate * d_norm_sq + 1e-12), 1.0)
            gamma = min(gamma, safe_cap)
            floor = min(safe_cap, 1e-4) if safe_cap > 0 else 0.0
            return max(gamma, floor) 
        elif self.step_size_rule == "damped":
            d = direction
            d_norm_sq = np.dot(d, d)
            if d_norm_sq < 1e-12:
                return 0.0

            gap = -np.dot(gradient, d)
            if gap <= 0:
                return 0.0

            min_y = float(np.min(y_current))
            min_y = max(min_y, self.epsilon)
            if discount_weights is not None:
                effective_samples = float(np.sum(discount_weights))
            else:
                effective_samples = float(len(observations)) if observations is not None else 0.0

            prior_scale = float(np.max(np.abs(self.alpha - 1.0))) if self.alpha is not None else 1.0
            L_estimate = (prior_scale + effective_samples) / (min_y * min_y)
            if self.entropy_weight > 0:
                L_estimate += float(self.entropy_weight) / min_y
            gamma = min(gap / (L_estimate * d_norm_sq + 1e-12), 0.5)
            gamma = min(gamma, safe_cap)
            floor = min(safe_cap, 1e-4) if safe_cap > 0 else 0.0
            return max(gamma, floor)

        elif self.step_size_rule == "constant":
            return min(self.constant_step_size, safe_cap if safe_cap > 0 else 0.0)

        elif self.step_size_rule == "line_search":
            from .gradients import compute_objective

            def objective_at_gamma(gamma):
                y_new = y_current + gamma * direction
                return compute_objective(y_new, self.alpha, observations, self.epsilon, discount_weights, precomputed)

            from scipy.optimize import minimize_scalar

            upper_bound = safe_cap if safe_cap > 0 else 0.0
            if upper_bound <= 1e-12:
                return 0.0

            result = minimize_scalar(
                objective_at_gamma,
                bounds=(0, upper_bound),
                method='bounded',
                options={'xatol': 1e-6}
            )

            return result.x

        else:
            raise ValueError(f"Unknown step size rule: {self.step_size_rule}")

    def step(
        self,
        y_current: np.ndarray,
        gradient: np.ndarray,
        observations: List[Tuple[List[int], np.ndarray]],
        discount_weights: np.ndarray = None,
        current_obj: float = None,
        precomputed: dict = None
    ) -> np.ndarray:
        """
        Perform one Frank-Wolfe step

        Takes args:
            y_current: Current iterate
            gradient: Gradient at current point
            observations: Observation history
        Returns:
            Next iterate y_{k+1}
        """
        s_toward = self.lmo(gradient)
        toward_dir = s_toward - y_current
        step_cap = self._max_feasible_step(y_current, toward_dir)

        chosen_dir = toward_dir

        if self.use_away_steps:
            s_away = self.lmo(-gradient) 
            pair_dir = s_toward - s_away
            pair_slope = np.dot(gradient, pair_dir)
            toward_slope = np.dot(gradient, toward_dir)
            pair_cap = self._max_feasible_step(y_current, pair_dir)

            if pair_cap > 0 and pair_slope < toward_slope - 1e-12:
                chosen_dir = pair_dir
                step_cap = pair_cap

        self.last_gap = np.dot(gradient, y_current - s_toward)

        if step_cap <= 0:
            return y_current

        gamma = self.compute_step_size(
            self.iteration_count,
            y_current,
            s_toward,
            gradient,
            observations,
            discount_weights,
            precomputed,
            direction=chosen_dir,
            max_step=step_cap
        )

        y_candidate = y_current + gamma * chosen_dir
        final_gamma = gamma

        if self.monotone_backtracking and self.step_size_rule != "line_search":
            from .gradients import compute_objective
            if current_obj is None:
                current_obj = compute_objective(
                    y_current,
                    self.alpha,
                    observations,
                    self.epsilon,
                    discount_weights,
                    precomputed,
                )

            obj_candidate = compute_objective(
                y_candidate,
                self.alpha,
                observations,
                self.epsilon,
                discount_weights,
                precomputed,
            )

            tol = self.monotone_tolerance * (abs(current_obj) + 1.0)
            bt_iter = 0
            while (
                obj_candidate > current_obj + tol
                and bt_iter < self.backtrack_iters
                and gamma > 1e-12
            ):
                gamma *= self.backtrack_beta
                y_candidate = y_current + gamma * chosen_dir
                obj_candidate = compute_objective(
                    y_candidate,
                    self.alpha,
                    observations,
                    self.epsilon,
                    discount_weights,
                    precomputed,
                )
                bt_iter += 1

            final_gamma = gamma

        self.last_step_size = final_gamma
        return y_candidate

    def has_converged(self, y_current: np.ndarray, y_next: np.ndarray, delta: float) -> bool:
        return self.last_gap < self.tolerance
