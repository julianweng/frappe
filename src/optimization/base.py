"""
Base class for FMAP optimizers
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple, Optional
import time


class OptimizerBase(ABC):
    #Abstract base class for FMAP optimization algorithms.
    def __init__(
        self,
        F: np.ndarray,
        f: np.ndarray,
        alpha: np.ndarray,
        max_iterations: int = 1000,
        tolerance: float = 1e-7,
        epsilon: float = 1e-12,
        verbose: bool = False,
        early_stopping_patience: Optional[int] = None,
        early_stopping_threshold: float = 1e-6,
        require_gurobi: bool = False,
        entropy_weight: float = 0.0
    ):
        """
        Optimizer args:
            F: Constraint matrix
            f: Constraint vector
            alpha: Dirichlet prior parameters
            max_iterations: Maximum number of optimization iterations
            tolerance: Convergence tolerance
            epsilon: Numerical stability constant
            verbose: Whether to print progress
            early_stopping_patience: If set, stop after this many iterations without improvement
            early_stopping_threshold: Minimum objective improvement to reset patience counter
            require_gurobi: If True, raise error if Gurobi is not available (disable scipy fallback)
            entropy_weight: Weight lambda for entropy regularization 
        """
        self.F = F
        self.f = f
        self.alpha = alpha
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.epsilon = epsilon
        self.verbose = verbose
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.require_gurobi = require_gurobi
        self.entropy_weight = entropy_weight

        self.iteration_count = 0
        self.convergence_history = []
        self.objective_history = []
        self.time_history = []

        self.best_objective = float('inf')
        self.patience_counter = 0

    def has_converged(self, y_current: np.ndarray, y_next: np.ndarray, delta: float) -> bool:
        #Check if optimization has converged
        return delta < self.tolerance

    @abstractmethod
    def step(
        self,
        y_current: np.ndarray,
        gradient: np.ndarray,
        observations: List[Tuple[List[int], np.ndarray]],
        discount_weights: np.ndarray = None,
        current_obj: float = None,
        precomputed: dict = None
    ) -> np.ndarray:
        pass

    def optimize(
        self,
        y_init: np.ndarray,
        observations: List[Tuple[List[int], np.ndarray]],
        discount_factor: float = 1.0
    ) -> Tuple[np.ndarray, dict]:
        """
        Run the optimization algorithm to convergence.

        Args:
            y_init: Initial point 
            observations: List of (observable_sequences, q_values) from gameplay
            discount_factor: Discount factor for observations

        Returns:
            Tuple of (optimal_y, info_dict) where info_dict contains:
            - iterations, converged, objective, and time
        """
        from .gradients import compute_gradient, compute_objective, compute_discounted_weights, precompute_observation_data

        start_time = time.time()

        if discount_factor < 1.0:
            discount_weights = compute_discounted_weights(len(observations), discount_factor)
        else:
            discount_weights = None

        precomputed = precompute_observation_data(observations)

        y_current = y_init.copy()
        converged = False

        self.iteration_count = 0
        self.convergence_history = []
        self.objective_history = []
        self.time_history = []

        self.best_objective = float('inf')
        self.patience_counter = 0
        early_stopped = False

        for iteration in range(self.max_iterations):
            self.iteration_count = iteration + 1

            gradient = compute_gradient(
                y_current, self.alpha, observations,
                epsilon=self.epsilon,
                discount_weights=discount_weights,
                precomputed=precomputed,
                entropy_weight=self.entropy_weight
            )

            if self.__class__.__name__ == 'ProjectedGradientDescent':
                current_obj = compute_objective(
                    y_current, self.alpha, observations,
                    epsilon=self.epsilon,
                    discount_weights=discount_weights,
                    precomputed=precomputed,
                    entropy_weight=self.entropy_weight
                )
                y_next = self.step(y_current, gradient, observations, discount_weights, current_obj, precomputed)
            elif (
                hasattr(self, 'step_size_rule')
                and (
                    self.step_size_rule == "line_search"
                    or getattr(self, "monotone_backtracking", False)
                )
            ):
                current_obj = compute_objective(
                    y_current, self.alpha, observations,
                    epsilon=self.epsilon,
                    discount_weights=discount_weights,
                    precomputed=precomputed,
                    entropy_weight=self.entropy_weight
                )
                y_next = self.step(y_current, gradient, observations, discount_weights, current_obj, precomputed)
            else:
                y_next = self.step(y_current, gradient, observations, discount_weights, None, precomputed)

            delta = np.linalg.norm(y_next - y_current)
            self.convergence_history.append(delta)

            if self.early_stopping_patience is not None or iteration == self.max_iterations - 1:
                obj_value = compute_objective(
                    y_next, self.alpha, observations,
                    epsilon=self.epsilon,
                    discount_weights=discount_weights,
                    precomputed=precomputed,
                    entropy_weight=self.entropy_weight
                )
                self.objective_history.append(obj_value)
            else:
                obj_value = float('inf')

            elapsed_time = time.time() - start_time
            self.time_history.append(elapsed_time)

            if self.verbose and iteration % 10 == 0:
                print(f"Iter {iteration}: obj={obj_value:.6f}, delta={delta:.2e}")

            if self.has_converged(y_current, y_next, delta):
                converged = True
                if self.verbose:
                    print(f"Converged in {iteration + 1} iterations")
                break

            if self.early_stopping_patience is not None:
                improvement = self.best_objective - obj_value
                if improvement > self.early_stopping_threshold:
                    self.best_objective = obj_value
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1

                    if self.patience_counter >= self.early_stopping_patience:
                        early_stopped = True
                        if self.verbose:
                            print(f"Early stopping at iteration {iteration + 1}: "
                                  f"no improvement for {self.patience_counter} iterations")
                        break

            y_current = y_next

        total_time = time.time() - start_time

        if not self.objective_history:
            obj_value = compute_objective(
                y_current, self.alpha, observations,
                epsilon=self.epsilon,
                discount_weights=discount_weights,
                precomputed=precomputed
            )
            self.objective_history.append(obj_value)

        info = {
            'iterations': self.iteration_count,
            'converged': converged,
            'early_stopped': early_stopped,
            'objective': self.objective_history[-1] if self.objective_history else np.inf,
            'time': total_time,
            'convergence_history': self.convergence_history,
            'objective_history': self.objective_history,
            'time_history': self.time_history
        }

        return y_current, info

    def reset_statistics(self):
        self.iteration_count = 0
        self.convergence_history = []
        self.objective_history = []
        self.time_history = []

        # Reset early stopping tracking
        self.best_objective = float('inf')
        self.patience_counter = 0
