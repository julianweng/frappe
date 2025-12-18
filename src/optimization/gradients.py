"""
Gradient calculations for FMAP optimization with extended with entropy regularization for:
- Strong convexity (faster convergence: O(1/k^2) instead of O(1/k))
- Regularization
- Connection to maximum entropy principle
"""

import numpy as np
from typing import List, Tuple


def compute_gradient(
    y: np.ndarray,
    alpha: np.ndarray,
    observations: List[Tuple[List[int], np.ndarray]],
    epsilon: float = 1e-12,
    discount_weights: np.ndarray = None,
    precomputed: dict = None,
    entropy_weight: float = 0.0
) -> np.ndarray:
    """
    Compute the gradient of the negative log-posterior with optional entropy regularization.

    Takes args:
        y: Current opponent model 
        alpha: Dirichlet prior parameters
        observations: List of (observable_sequences, q_values) tuples, one per game iteration
        epsilon: Small value to prevent division by zero
        discount_weights: Optional weights for each observation (for discounted likelihood)
        entropy_weight: Weight lambda for entropy regularization
    """
    n = len(y)

    y_safe = np.maximum(y, epsilon)

    gradient = (1.0 - alpha) / y_safe

    num_observations = len(observations)
    if num_observations == 0:
        return gradient

    if discount_weights is None:
        discount_weights = np.ones(num_observations)

    if precomputed is not None and 'seq_indices_list' in precomputed:
        seq_indices_list = precomputed['seq_indices_list']
        q_vals_list = precomputed['q_vals_list']
        
        for t in range(num_observations):
            seq_indices = seq_indices_list[t]
            q_vals = q_vals_list[t]
            
            denominator = np.dot(q_vals, y_safe[seq_indices])
            denominator = max(denominator, epsilon)
            gradient[seq_indices] -= discount_weights[t] * q_vals / denominator
    else:
        for t, (observable_seqs, q_values) in enumerate(observations):
            seq_indices = np.array(observable_seqs)
            q_vals = np.array(q_values)

            denominator = np.dot(q_vals, y_safe[seq_indices])
            denominator = max(denominator, epsilon)
            gradient[seq_indices] -= discount_weights[t] * q_vals / denominator

    if entropy_weight > 0:
        gradient += entropy_weight * (1.0 + np.log(y_safe))

    return gradient


def precompute_observation_data(observations: List[Tuple[List[int], np.ndarray]]) -> dict:
    if not observations:
        return {}
    
    seq_indices_list = [np.array(obs[0]) for obs in observations]
    q_vals_list = [np.array(obs[1]) for obs in observations]
    
    return {
        'seq_indices_list': seq_indices_list,
        'q_vals_list': q_vals_list,
    }


def compute_objective(
    y: np.ndarray,
    alpha: np.ndarray,
    observations: List[Tuple[List[int], np.ndarray]],
    epsilon: float = 1e-12,
    discount_weights: np.ndarray = None,
    precomputed: dict = None,
    entropy_weight: float = 0.0
) -> float:
    """
    Compute the negative log-posterior objective value with optional entropy regularization.

    Args:
        y: Current opponent model
        alpha: Dirichlet prior parameters
        observations: List of (observable_sequences, q_values) tuples
        epsilon: Small value to prevent log(0)
        entropy_weight: Weight λ for entropy regularization
    """
    y_safe = np.maximum(y, epsilon)

    log_prior = np.sum((alpha - 1.0) * np.log(y_safe))

    num_observations = len(observations)
    if num_observations == 0:
        return -log_prior

    if discount_weights is None:
        discount_weights = np.ones(num_observations)

    log_likelihood = 0.0

    if precomputed is not None and 'seq_indices_list' in precomputed:
        seq_indices_list = precomputed['seq_indices_list']
        q_vals_list = precomputed['q_vals_list']

        for t in range(num_observations):
            seq_indices = seq_indices_list[t]
            q_vals = q_vals_list[t]
            likelihood_t = np.dot(q_vals, y_safe[seq_indices])
            likelihood_t = max(likelihood_t, epsilon)
            log_likelihood += discount_weights[t] * np.log(likelihood_t)
    else:
        for t, (observable_seqs, q_values) in enumerate(observations):
            seq_indices = np.array(observable_seqs)
            q_vals = np.array(q_values)
            likelihood_t = np.dot(q_vals, y_safe[seq_indices])
            likelihood_t = max(likelihood_t, epsilon)
            log_likelihood += discount_weights[t] * np.log(likelihood_t)

    entropy_term = 0.0
    if entropy_weight > 0:
        entropy_term = entropy_weight * np.sum(y_safe * np.log(y_safe))

    return -(log_prior + log_likelihood) + entropy_term


def compute_discounted_weights(num_observations: int, discount_factor: float) -> np.ndarray:
    """
    Compute discount weights for observations: takes in num_observations & discount_factor
    """
    if discount_factor >= 1.0:
        return np.ones(num_observations)

    T = num_observations
    weights = np.array([discount_factor ** (T - t - 1) for t in range(T)])

    weights = weights * (num_observations / np.sum(weights))

    return weights


def compute_stochastic_gradient(
    y: np.ndarray,
    alpha: np.ndarray,
    observations: List[Tuple[List[int], np.ndarray]],
    batch_size: int,
    epsilon: float = 1e-12,
    discount_weights: np.ndarray = None,
    rng: np.random.Generator = None,
    entropy_weight: float = 0.0
) -> np.ndarray:
    """
    Compute stochastic gradient using a mini-batch of observations.
    Instead of computing gradient over ALL observations O(n), sample a batch of size B and scale: O(B) per iteration
    Mathematical convergence: O(1/√k) for smooth convex objectives
    Args:
        y: Current opponent model (sequence-form probabilities)
        alpha: Dirichlet prior parameters
        observations: Full list of observations
        batch_size: Number of observations to sample
        epsilon: Numerical stability constant
        discount_weights: Optional weights for observations
        rng: Random number generator (for reproducibility)
        entropy_weight: Weight lambda for entropy regularization 
    """
    n = len(y)
    num_observations = len(observations)
    
    if num_observations == 0 or batch_size <= 0:
        y_safe = np.maximum(y, epsilon)
        return (1.0 - alpha) / y_safe
    
    if rng is None:
        rng = np.random.default_rng()
    
    actual_batch_size = min(batch_size, num_observations)
    batch_indices = rng.choice(num_observations, size=actual_batch_size, replace=False)
    
    batch_observations = [observations[i] for i in batch_indices]
    
    if discount_weights is not None:
        batch_weights = discount_weights[batch_indices]
        batch_weights = batch_weights * (num_observations / actual_batch_size)
    else:
        batch_weights = None
    
    y_safe = np.maximum(y, epsilon)
    
    gradient = (1.0 - alpha) / y_safe
    
    scale_factor = num_observations / actual_batch_size
    
    for i, (observable_seqs, q_values) in enumerate(batch_observations):
        seq_indices = np.array(observable_seqs)
        q_vals = np.array(q_values)
        
        denominator = np.dot(q_vals, y_safe[seq_indices])
        denominator = max(denominator, epsilon)
        
        weight = batch_weights[i] if batch_weights is not None else scale_factor
        gradient[seq_indices] -= weight * q_vals / denominator
    
    if entropy_weight > 0:
        gradient += entropy_weight * (1.0 + np.log(y_safe))
    
    return gradient


def check_gradient(
    y: np.ndarray,
    alpha: np.ndarray,
    observations: List[Tuple[List[int], np.ndarray]],
    epsilon: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray, float]:
    analytical = compute_gradient(y, alpha, observations)
    numerical = np.zeros_like(y)

    for i in range(len(y)):
        y_plus = y.copy()
        y_plus[i] += epsilon

        y_minus = y.copy()
        y_minus[i] -= epsilon

        f_plus = compute_objective(y_plus, alpha, observations)
        f_minus = compute_objective(y_minus, alpha, observations)

        numerical[i] = (f_plus - f_minus) / (2 * epsilon)

    max_diff = np.max(np.abs(analytical - numerical))

    return analytical, numerical, max_diff
