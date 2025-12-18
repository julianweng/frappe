"""
Kuhn Poker Dealer to handles card dealing, game logic, and observability mapping
"""

import numpy as np
from enum import Enum
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


class Card(Enum):
    JACK = 0
    QUEEN = 1
    KING = 2

    def __str__(self):
        return self.name[0]  # J, Q, K

    def beats(self, other: 'Card') -> bool:
        return self.value > other.value


class Action(Enum):
    BET = "b"
    CHECK = "ch"
    CALL = "ca"
    FOLD = "f"

    def __str__(self):
        return self.value


@dataclass
class GameState:
    p1_card: Card
    p2_card: Card
    history: List[Tuple[int, Action]]  
    pot: int = 2  
    terminal: bool = False
    payoff_p1: Optional[float] = None


class KuhnDealer:
    def __init__(self, random_state: Optional[int] = None):
        if random_state is not None:
            np.random.seed(random_state)

        self.cards = [Card.JACK, Card.QUEEN, Card.KING]

        from ..core.utils import build_observability_map
        self.observability_map = build_observability_map()

    def deal(self) -> Tuple[Card, Card]:
        #Deal two cards (one to each player) returns tuple 
        dealt = np.random.choice(self.cards, size=2, replace=False)
        return dealt[0], dealt[1]

    def create_game(self) -> GameState:
        p1_card, p2_card = self.deal()
        return GameState(
            p1_card=p1_card,
            p2_card=p2_card,
            history=[],
            pot=2,
            terminal=False
        )

    def is_terminal(self, state: GameState) -> bool:
        history = state.history

        if len(history) == 0:
            return False

        actions = [a for _, a in history]

        if len(actions) == 1:
            return False

        if len(actions) == 2:
            # Patterns: [Check, Check] or [Bet, Fold] or [Bet, Call]
            if actions[0] == Action.CHECK and actions[1] == Action.CHECK:
                return True
            if actions[0] == Action.BET and actions[1] in [Action.FOLD, Action.CALL]:
                return True
            return False

        if len(actions) == 3:
            # Patterns: [Check, Bet, Fold] or [Check, Bet, Call]
            if actions[0] == Action.CHECK and actions[1] == Action.BET:
                if actions[2] in [Action.FOLD, Action.CALL]:
                    return True
            return False

        return False

    def compute_payoff(self, state: GameState) -> float:
        actions = [a for _, a in state.history]
        p1_card = state.p1_card
        p2_card = state.p2_card

        if Action.FOLD in actions:
            if actions[-1] == Action.FOLD:
                folder = state.history[-1][0]
                if folder == 1:
                    return -(state.pot // 2)  
                else:
                    return state.pot // 2  
            return 0.0

        pot_size = state.pot

        if p1_card.beats(p2_card):
            return pot_size / 2 
        else:
            return -pot_size / 2  

    def get_leaf_node_name(self, state: GameState) -> str:
        #Get the leaf node name for observability lookup.
        p1_card = state.p1_card
        p2_card = state.p2_card
        actions = [a for _, a in state.history]

        if len(actions) == 2 and actions[0] == Action.BET and actions[1] == Action.CALL:
            return f"B_{p1_card} ca_{p2_card}"
        if len(actions) == 2 and actions[0] == Action.BET and actions[1] == Action.FOLD:
            return f"B_{p1_card} f_{p2_card}"
        if len(actions) == 2 and actions[0] == Action.CHECK and actions[1] == Action.CHECK:
            return f"Ch_{p1_card} ch_{p2_card}"
        if len(actions) == 3 and actions[0] == Action.CHECK and actions[1] == Action.BET and actions[2] == Action.CALL:
            return f"Ch_{p1_card} b_{p2_card} Ca_{p1_card}"
        if len(actions) == 3 and actions[0] == Action.CHECK and actions[1] == Action.BET and actions[2] == Action.FOLD:
            return f"Ch_{p1_card} b_{p2_card} F_{p1_card}"

        raise ValueError(f"Unrecognized terminal history: {actions}")

    def get_observable_sequences(
        self,
        state: GameState,
        player: int = 1
    ) -> Tuple[List[int], np.ndarray]:
        leaf_name = self.get_leaf_node_name(state)

        player_key = f'p{player}'
        observable_seqs = self.observability_map[player_key][leaf_name]
        q_values = np.ones(len(observable_seqs)) / float(len(observable_seqs))
        return observable_seqs, q_values

    def play_hand(
        self,
        p1_strategy: np.ndarray,
        p2_strategy: np.ndarray,
        return_observations: bool = True
    ) -> Tuple[float, Optional[Tuple[List[int], np.ndarray]]]:
       
        from ..core import get_kuhn_game
        game = get_kuhn_game()

        expected_payoff = np.dot(p1_strategy, np.dot(game.A, p2_strategy))

        if return_observations:
            from .runner import _simulate_observation_worker
            obs_seqs, q_vals = _simulate_observation_worker(p1_strategy, p2_strategy)
            return expected_payoff, (obs_seqs, q_vals)

        return expected_payoff, None
