"""
Stochastic Frank-Wolfe optimizer for FMAP: uses mini-batch gradients
for faster per-iteration cost while maintaining convergence guarantees.
Main idea is that it is better for per-iteration cost, 
convergence rate and that it is projection-free: maintains FW's LP-based approach

Reference: Hazan & Luo's "Variance-Reduced and Projection-Free 
"""

import numpy as np
from typing import List, Tuple, Optional
import time

from .fw import FrankWolfe
from .gradients import (
    compute_stochastic_gradient,
    compute_gradient,
    compute_objective,
    compute_discounted_weights,
    precompute_observation_data
)


class StochasticFrankWolfe(FrankWolfe): 
    def __init__(
        self,
        F: np.ndarray,
        f: np.ndarray,
        alpha: np.ndarray,
        batch_size: int = 50,  # Number of observations per gradient estimate
        use_variance_reduction: bool = False,  # Enable SVRG-style variance reduction
        variance_reduction_freq: int = 10,  # Compute full gradient every N iterations
        learning_rate_decay: bool = True,  # Decay step size for convergence
        seed: Optional[int] = None,  # Random seed for reproducibility
        **kwargs
    ):
        """
        SFW Args:
            F: Constraint matrix
            f: Constraint vector
            alpha: Prior parameters
            batch_size: Number of observations to sample per iteration
            use_variance_reduction: If True, use SVRG-style variance reduction
            variance_reduction_freq: How often to compute full gradient 
            learning_rate_decay: If True, use decreasing step sizes
            seed: Random seed for reproducibility
            **kwargs: Additional arguments passed to FrankWolfe
        """
        kwargs.setdefault('step_size_rule', 'stochastic')   
        kwargs.setdefault('max_iterations', 100)   
        kwargs.setdefault('tolerance', 1.0)   
        kwargs.setdefault('monotone_backtracking', False)  
        
        super().__init__(F, f, alpha, **kwargs)
        
        self.batch_size = batch_size
        self.use_variance_reduction = use_variance_reduction
        self.variance_reduction_freq = variance_reduction_freq
        self.learning_rate_decay = learning_rate_decay
        
        self.rng = np.random.default_rng(seed)
        
        self.full_gradient_snapshot = None
        self.y_snapshot = None
        self.snapshot_iteration = 0
        self.snapshot_observation_count = 0

        self.global_iteration = 0   
        self._current_global_iteration = 0
    
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
        if direction is None:
            direction = s - y_current
        
        max_step = max_step if max_step > 0 else 0.0
        safe_cap = max_step * self.step_size_safety

        global_k = getattr(self, "_current_global_iteration", iteration)
        
        if self.step_size_rule == "stochastic":
            if self.learning_rate_decay:
                # Decreasing step size for convergence guarantee
                gamma = 1.0 / np.sqrt(global_k + 1.0)
            else:
                gamma = 0.1
            return min(gamma, safe_cap)
        else:
            return super().compute_step_size(
                global_k, y_current, s, gradient, observations,
                discount_weights, precomputed, direction, max_step
            )
    
    def optimize(
        self,
        y_init: np.ndarray,
        observations: List[Tuple[List[int], np.ndarray]],
        discount_factor: float = 1.0
    ) -> Tuple[np.ndarray, dict]:
        start_time = time.time()
        
        if discount_factor < 1.0:
            discount_weights = compute_discounted_weights(len(observations), discount_factor)
        else:
            discount_weights = None
        
        precomputed = precompute_observation_data(observations)

        num_observations = len(observations)
        
        y_current = y_init.copy()
        converged = False
        
        self.iteration_count = 0
        self.convergence_history = []
        self.objective_history = []
        self.time_history = []
        
        self.best_objective = float('inf')
        self.patience_counter = 0
        early_stopped = False
        
        total_gradient_samples = 0
        
        for iteration in range(self.max_iterations):
            self.iteration_count = iteration + 1
            self._current_global_iteration = self.global_iteration
            
            if self.use_variance_reduction:
                gradient, used_full = self._compute_variance_reduced_gradient(
                    y_current, observations, discount_weights,
                    precomputed, self._current_global_iteration, num_observations
                )
                total_gradient_samples += self.batch_size
                if used_full:
                    total_gradient_samples += num_observations  
            else:
                # Pure stochastic gradient
                gradient = compute_stochastic_gradient(
                    y_current,
                    self.alpha,
                    observations,
                    batch_size=self.batch_size,
                    epsilon=self.epsilon,
                    discount_weights=discount_weights,
                    rng=self.rng,
                    entropy_weight=self.entropy_weight
                )
                total_gradient_samples += self.batch_size
            
            y_next = self.step(
                y_current, gradient, observations,
                discount_weights, None, precomputed
            )
            
            delta = np.linalg.norm(y_next - y_current)
            self.convergence_history.append(delta)
            
            elapsed = time.time() - start_time
            self.time_history.append(elapsed)

            self.global_iteration += 1
            
            if iteration % 10 == 0 or iteration == self.max_iterations - 1:
                obj_value = compute_objective(
                    y_next, self.alpha, observations,
                    epsilon=self.epsilon,
                    discount_weights=discount_weights,
                    precomputed=precomputed,
                    entropy_weight=self.entropy_weight
                )
                self.objective_history.append(obj_value)
                
                if self.verbose:
                    print(f"SFW Iter {iteration}: obj={obj_value:.6f}, delta={delta:.2e}, "
                          f"samples={total_gradient_samples}")
            
            if delta < self.tolerance * 0.01:  
                converged = True
                if self.verbose:
                    print(f"Converged in {iteration + 1} iterations (delta={delta:.2e})")
                break
            
            if self.early_stopping_patience is not None and self.objective_history:
                obj_value = self.objective_history[-1]
                improvement = self.best_objective - obj_value
                if improvement > self.early_stopping_threshold:
                    self.best_objective = obj_value
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.early_stopping_patience:
                        early_stopped = True
                        if self.verbose:
                            print(f"Early stopping at iteration {iteration + 1}")
                        break
            
            y_current = y_next
        
        total_time = time.time() - start_time
        
        final_obj = compute_objective(
            y_current, self.alpha, observations,
            epsilon=self.epsilon,
            discount_weights=discount_weights,
            precomputed=precomputed,
            entropy_weight=self.entropy_weight
        )
        self.objective_history.append(final_obj)
        
        info = {
            'iterations': self.iteration_count,
            'converged': converged,
            'early_stopped': early_stopped,
            'objective': self.objective_history[-1] if self.objective_history else np.inf,
            'time': total_time,
            'convergence_history': self.convergence_history,
            'objective_history': self.objective_history,
            'time_history': self.time_history,
            'total_gradient_samples': total_gradient_samples, 
            'batch_size': self.batch_size,
            'effective_passes': total_gradient_samples / max(1, len(observations))
        }
        
        return y_current, info
    
    def _compute_variance_reduced_gradient(
        self,
        y_current: np.ndarray,
        observations: List[Tuple[List[int], np.ndarray]],
        discount_weights: np.ndarray,
        precomputed: dict,
        global_iteration: int,
        num_observations: int
    ) -> Tuple[np.ndarray, bool]:
        """
        Compute variance-reduced stochastic gradient (SVRG-style).
        """
        need_snapshot = (
            self.full_gradient_snapshot is None
            or self.snapshot_observation_count != num_observations
            or (global_iteration - self.snapshot_iteration) % self.variance_reduction_freq == 0
        )

        full_grad_used = False

        if need_snapshot:
            self.full_gradient_snapshot = compute_gradient(
                y_current, self.alpha, observations,
                epsilon=self.epsilon,
                discount_weights=discount_weights,
                precomputed=precomputed,
                entropy_weight=self.entropy_weight
            )
            self.y_snapshot = y_current.copy()
            self.snapshot_iteration = global_iteration
            self.snapshot_observation_count = num_observations
            full_grad_used = True
        
        if self.full_gradient_snapshot is None:
            grad = compute_gradient(
                y_current, self.alpha, observations,
                epsilon=self.epsilon,
                discount_weights=discount_weights,
                precomputed=precomputed,
                entropy_weight=self.entropy_weight
            )
            return grad, True
        
        actual_batch_size = min(self.batch_size, num_observations)
        batch_indices = self.rng.choice(num_observations, size=actual_batch_size, replace=False)
        
        batch_observations = [observations[i] for i in batch_indices]
        
        if discount_weights is not None:
            batch_weights = discount_weights[batch_indices]
            batch_weights = batch_weights * (num_observations / actual_batch_size)
        else:
            batch_weights = None
        
        stoch_grad_current = self._compute_batch_gradient(
            y_current, batch_observations, batch_weights, num_observations
        )
        
        # Compute stochastic gradient at snapshot point using SAME batch
        stoch_grad_snapshot = self._compute_batch_gradient(
            self.y_snapshot, batch_observations, batch_weights, num_observations
        )
        
        vr_gradient = stoch_grad_current - stoch_grad_snapshot + self.full_gradient_snapshot
        
        return vr_gradient, full_grad_used
    
    def _compute_batch_gradient(
        self,
        y: np.ndarray,
        batch_observations: list,
        batch_weights: np.ndarray,
        num_observations: int
    ) -> np.ndarray:
        """Compute gradient on a fixed batch of observations."""
        y_safe = np.maximum(y, self.epsilon)
        
        gradient = (1.0 - self.alpha) / y_safe
        
        scale_factor = num_observations / len(batch_observations)
        
        for i, (observable_seqs, q_values) in enumerate(batch_observations):
            seq_indices = np.array(observable_seqs)
            q_vals = np.array(q_values)
            
            denominator = np.dot(q_vals, y_safe[seq_indices])
            denominator = max(denominator, self.epsilon)
            
            weight = batch_weights[i] if batch_weights is not None else scale_factor
            gradient[seq_indices] -= weight * q_vals / denominator
        
        if self.entropy_weight > 0:
            gradient += self.entropy_weight * (1.0 + np.log(y_safe))
        
        return gradient

    def reset_statistics(self):
        """Reset optimization statistics and global VR state."""
        super().reset_statistics()
        self.global_iteration = 0
        self._current_global_iteration = 0
        self.full_gradient_snapshot = None
        self.y_snapshot = None
        self.snapshot_iteration = 0
        self.snapshot_observation_count = 0
