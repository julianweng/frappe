"""
Projected Gradient Descent (PGD) optimizer for FMAP
"""

import numpy as np
from typing import List, Tuple, Optional
try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False
    from scipy.optimize import minimize

from .base import OptimizerBase


class ProjectedGradientDescent(OptimizerBase):
    """
    Projected Gradient Descent with Quadratic Programming projection.
    """

    def __init__(
        self,
        F: np.ndarray,
        f: np.ndarray,
        alpha: np.ndarray,
        learning_rate: float = 1.0,
        backtrack_c: float = 1e-4,
        backtrack_beta: float = 0.5,
        min_learning_rate: float = 1e-16,
        use_gurobi: bool = True,
        gurobi_time_limit: Optional[float] = None, 
        gurobi_mip_gap: float = 1e-4, 
        tolerance: float = 5e-2,  
        max_iterations: int = 30, 
        early_stopping_patience: int = 3,  
        early_stopping_threshold: float = 1e-3,  
        **kwargs
    ):
        """
        PGD optimizer args:
            F: Constraint matrix
            f: Constraint vector
            alpha: Prior parameters
            learning_rate: Initial learning rate n
            backtrack_c: Armijo condition parameter
            backtrack_beta: Backtracking multiplier
            min_learning_rate: Minimum n before halting
            use_gurobi: Whether to use Gurobi or scipy 
            gurobi_time_limit: Time limit for each Gurobi QP solve in seconds
            gurobi_mip_gap: MIP gap tolerance for Gurobi optimization
        """
        super().__init__(
            F, f, alpha,
            tolerance=tolerance,
            max_iterations=max_iterations,
            early_stopping_patience=early_stopping_patience,
            early_stopping_threshold=early_stopping_threshold,
            **kwargs
        )

        self.learning_rate_init = learning_rate
        self.backtrack_c = backtrack_c
        self.backtrack_beta = backtrack_beta
        self.min_learning_rate = min_learning_rate
        self.use_gurobi = use_gurobi and GUROBI_AVAILABLE
        
        if self.require_gurobi and not self.use_gurobi:
            raise RuntimeError(
                "Gurobi is required for PGD but not available. "
                "Install Gurobi or set require_gurobi=False to use scipy fallback."
            )
        self.gurobi_time_limit = gurobi_time_limit
        self.gurobi_mip_gap = gurobi_mip_gap
        self.last_step_size = None

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

            model = gp.Model("PGD_Projection")
            model.setParam('OutputFlag', 0)  

            if self.gurobi_time_limit is not None:
                model.setParam('TimeLimit', self.gurobi_time_limit)
            model.setParam('MIPGap', self.gurobi_mip_gap)

            y_vars = model.addMVar(n, lb=self.epsilon, name="y")

            model.addConstr(self.F @ y_vars == self.f, name="sequence_constraints")

            model.setObjective(y_vars @ y_vars, GRB.MINIMIZE)

            model.update()
            self.gurobi_model = model
            self.gurobi_vars = y_vars

        except Exception as e:
            print(f"Warning: Failed to build Gurobi model: {e}")
            print("Falling back to scipy")
            self.use_gurobi = False
            self.gurobi_model = None

    def _project_gurobi(self, z: np.ndarray) -> np.ndarray:
        #Project z onto feasible set using Gurobi QP solver
        try:
            model = self.gurobi_model
            y_vars = self.gurobi_vars

            obj = y_vars @ y_vars - 2 * z @ y_vars
            model.setObjective(obj, GRB.MINIMIZE)

            model.optimize()

            if model.status == GRB.OPTIMAL:
                return y_vars.X.copy()
            elif model.status == GRB.TIME_LIMIT:
                
                if model.SolCount > 0:
                    return y_vars.X.copy()
                else:
                    print(f"Warning: Gurobi projection time limit reached without feasible solution")
                    return z   
            else:
                print(f"Warning: Gurobi optimization failed with status {model.status}")
                return z  

        except Exception as e:
            print(f"Warning: Gurobi projection failed: {e}")
            return z

    def _project_scipy(self, z: np.ndarray) -> np.ndarray:
        # Project z onto feasible set using scipy optimizer
        from scipy.optimize import minimize, LinearConstraint

        n = len(z)

        def objective(y):
            diff = y - z
            return np.dot(diff, diff)

        def grad_objective(y):
            return 2 * (y - z)
 
        constraint = LinearConstraint(self.F, self.f, self.f)
 
        bounds = [(self.epsilon, None) for _ in range(n)]
 
        y0 = np.maximum(z, self.epsilon)

        result = minimize(
            objective,
            y0,
            method='SLSQP',
            jac=grad_objective,
            constraints=constraint,
            bounds=bounds,
            options={'ftol': 1e-9}
        )

        return result.x

    def project(self, z: np.ndarray) -> np.ndarray:
        """
        Project z onto the feasible set {y: Fy = f, y >= ε}.
        """
        if self.use_gurobi and self.gurobi_model is not None:
            return self._project_gurobi(z)
        else:
            return self._project_scipy(z)

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
        Perform one PGD step with backtracking line search.
        """
        from .gradients import compute_objective

        eta = self.learning_rate_init
        
        if current_obj is None:
            current_obj = compute_objective(y_current, self.alpha, observations, self.epsilon, discount_weights, precomputed)

        while eta > self.min_learning_rate:
            # Gradient step
            z = y_current - eta * gradient

            y_next = self.project(z)

            # Check Armijo condition
            next_obj = compute_objective(y_next, self.alpha, observations, self.epsilon, discount_weights, precomputed)

            direction = y_current - y_next
            descent = self.backtrack_c * np.dot(gradient, direction)

            if next_obj <= current_obj - descent:
                self.last_step_size = eta
                return y_next

            eta *= self.backtrack_beta

        z = y_current - self.min_learning_rate * gradient
        self.last_step_size = self.min_learning_rate
        return self.project(z)
