"""
Opponent Models for Kuhn Poker
"""

import numpy as np
from typing import Optional, Tuple


class DirichletOpponent:
    """
    Static opponent that plays a fixed strategy sampled from a Dirichlet distribution (non-adaptive)
    
    Action set(12 values for 6 information sets, 2 actions each):
        [alpha_call_Q, alpha_fold_Q,    # Facing bet with Q
         alpha_bet_Q, alpha_check_Q,    # Facing check with Q  
         alpha_call_J, alpha_fold_J,    # Facing bet with J
         alpha_bet_J, alpha_check_J,    # Facing check with J
         alpha_call_K, alpha_fold_K,    # Facing bet with K
         alpha_bet_K, alpha_check_K]    # Facing check with K
    
    Basically, higher the alpha, the more likely to take that action.
    """

    def __init__(
        self,
        strategy: Optional[np.ndarray] = None,
        alpha: Optional[np.ndarray] = None,
        num_sequences: int = 13,
        random_state: Optional[int] = None,
        game = None
    ):
        """
        Dirichlet Args:
            strategy: Fixed strategy in sequence-form
            alpha: Dirichlet parameters for sampling
            num_sequences: Number of sequences 
            random_state: Random seed for reproducibility
            game: KuhnGame object
        """
        if random_state is not None:
            np.random.seed(random_state)

        if strategy is not None:
            self.strategy = strategy.copy()
        else:
            if game is None:
                from ..core import get_kuhn_game
                game = get_kuhn_game()

            if alpha is None:
                alphas = {
                    'bet_Q': [2.0, 2.0],   
                    'check_Q': [2.0, 2.0], 
                    'bet_J': [2.0, 2.0],
                    'check_J': [2.0, 2.0],
                    'bet_K': [2.0, 2.0],
                    'check_K': [2.0, 2.0],
                }
            elif np.isscalar(alpha) or len(alpha) == 1:
                a = float(alpha) if np.isscalar(alpha) else float(alpha[0])
                alphas = {
                    'bet_Q': [a, a],
                    'check_Q': [a, a],
                    'bet_J': [a, a],
                    'check_J': [a, a],
                    'bet_K': [a, a],
                    'check_K': [a, a],
                }
            elif len(alpha) >= 12:
                alphas = {
                    'bet_Q': [alpha[0], alpha[1]],    
                    'check_Q': [alpha[2], alpha[3]],   
                    'bet_J': [alpha[4], alpha[5]],     
                    'check_J': [alpha[6], alpha[7]], 
                    'bet_K': [alpha[8], alpha[9]],    
                    'check_K': [alpha[10], alpha[11]], 
                }
            else:
                raise ValueError(f"alpha must be None, scalar, or have 12+ elements, got {len(alpha)}")

            # Each behavioral strategy is a probability distribution over actions at that infoset
            behav_bet_Q = np.random.dirichlet(alphas['bet_Q'])    
            behav_bet_J = np.random.dirichlet(alphas['bet_J'])    
            behav_bet_K = np.random.dirichlet(alphas['bet_K'])    
            behav_check_Q = np.random.dirichlet(alphas['check_Q'])  
            behav_check_J = np.random.dirichlet(alphas['check_J']) 
            behav_check_K = np.random.dirichlet(alphas['check_K'])  

            y = np.zeros(13)
            y[0] = 1.0

            # At each information set, the sequence weights ARE the behavioral probabilities
            y[1] = behav_bet_Q[0]  
            y[2] = behav_bet_Q[1]   
            
            y[3] = behav_check_Q[0]   
            y[4] = behav_check_Q[1]   
            
            y[5] = behav_bet_J[0]   
            y[6] = behav_bet_J[1]   
            
            y[7] = behav_check_J[0]   
            y[8] = behav_check_J[1]  
            
            y[9] = behav_bet_K[0]  
            y[10] = behav_bet_K[1]  
            
            y[11] = behav_check_K[0]   
            y[12] = behav_check_K[1]  

            self.strategy = y

        self.num_sequences = len(self.strategy)

    def get_strategy(self, iteration: int = 0) -> np.ndarray:
        # Get the opponent's strategy in sequence form
        return self.strategy.copy()

    def act(self, state: dict) -> str:
        # Note that these experiments use get_strategy() and compute expected payoffs analytically via x^T A y,
        # so this method is never called. Observations are sampled directly from sequence probabilities.
        raise NotImplementedError("Action selection not implemented for sequence-form Dirichlet opponent")


class AdversarialOpponent:
    """
    Adversarial opponent that best-responds to the agent's strategy
    """
    
    def __init__(self, game, random_state: Optional[int] = None):
        """
        Adversarial args:
            game: KuhnGame object with constraint matrices
            random_state: Random seed
        """
        self.game = game
        self.F = game.F
        self.f = game.f
        self.A = game.A
        self.current_strategy = None
        
    def set_agent_strategy(self, agent_strategy: np.ndarray):
        #Update the opponent's best response based on agent's current strategy
        from scipy.optimize import linprog
        
        c = self.A.T @ agent_strategy
        
        result = linprog(c, A_eq=self.F, b_eq=self.f, bounds=[(0, None)]*len(c), method='highs')
        
        if result.success:
            self.current_strategy = result.x
        else:
            from ..core import get_uniform_strategy
            self.current_strategy = get_uniform_strategy(self.F, self.f)
    
    def get_strategy(self, iteration: int = 0) -> np.ndarray:
        #Get the opponent's current best-response strategy in sequence form
        if self.current_strategy is None:
            from ..core import get_uniform_strategy
            self.current_strategy = get_uniform_strategy(self.F, self.f)
        return self.current_strategy.copy()


class SwitchingOpponent:
    """
    Non-stationary opponent that switches between two strategies.
    """

    def __init__(
        self,
        strategy1: np.ndarray,
        strategy2: np.ndarray,
        switch_iteration: int = 1000
    ):
        """
        Switiching args:
            strategy1: Initial strategy (played for first `switch_iteration` hands)
            strategy2: Second strategy which is played after switch
            switch_iteration: When to switch strategies
        """
        self.strategy1 = strategy1.copy()
        self.strategy2 = strategy2.copy()
        self.switch_iteration = switch_iteration

    def get_strategy(self, iteration: int) -> np.ndarray:
        if iteration < self.switch_iteration:
            return self.strategy1.copy()
        else:
            return self.strategy2.copy()

    def has_switched(self, iteration: int) -> bool:
        return iteration >= self.switch_iteration


class PassiveAggressiveSwitcher(SwitchingOpponent):
    """
    Passive-to-aggressive switching opponent that plays "Passive" for first 1000 hands, then switches to "Aggressive"
    """

    def __init__(
        self,
        F: np.ndarray,
        f: np.ndarray,
        switch_iteration: int = 1000,
        random_state: Optional[int] = None
    ):
        """
        P-A Args:
            F: Constraint matrix
            f: Constraint vector
            switch_iteration: When to switch (default 1000)
            random_state: Random seed
        """
        if random_state is not None:
            np.random.seed(random_state)

        passive = self._create_passive_strategy(F, f)

        aggressive = self._create_aggressive_strategy(F, f)

        super().__init__(passive, aggressive, switch_iteration)

    def _create_passive_strategy(self, F: np.ndarray, f: np.ndarray) -> np.ndarray:
        """
        Create a passive strategy that favors checking and folding.

        This is a heuristic construction - not necessarily optimal.
        Uses Dirichlet with lower values for aggressive actions
        """
        alpha_passive = np.ones(F.shape[1]) * 3.0

        strategy = np.random.dirichlet(alpha_passive)
        return strategy / np.sum(strategy)

    def _create_aggressive_strategy(self, F: np.ndarray, f: np.ndarray) -> np.ndarray:
        """
        Create an aggressive strategy that favors betting and raising.
        """
        alpha_aggressive = np.ones(F.shape[1]) * 3.0
        strategy = np.random.dirichlet(alpha_aggressive)
        return strategy / np.sum(strategy)


def create_random_opponent(
    F: np.ndarray,
    f: np.ndarray,
    alpha: Optional[np.ndarray] = None,
    random_state: Optional[int] = None
) -> DirichletOpponent:
    """
    Random Dirichlet opponent.

    Random Args:
        F: Constraint matrix
        f: Constraint vector
        alpha: Dirichlet parameters
        random_state: Random seed
    """
    num_sequences = F.shape[1]

    if alpha is None:
        alpha = np.ones(num_sequences) * 2.0

    return DirichletOpponent(
        alpha=alpha,
        num_sequences=num_sequences,
        random_state=random_state
    )
