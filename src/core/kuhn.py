"""
Kuhn Poker Sequence-Form Matrices as seen in Ganzfried (2025).
"""

import numpy as np
from typing import Tuple, Dict
from dataclasses import dataclass


@dataclass
class KuhnGame:
    E: np.ndarray  
    F: np.ndarray 
    A: np.ndarray 
    e: np.ndarray  
    f: np.ndarray 

    p1_sequences: list
    p2_sequences: list

    @property
    def num_p1_sequences(self) -> int:
        return len(self.p1_sequences)

    @property
    def num_p2_sequences(self) -> int:
        return len(self.p2_sequences)


def build_matrix_E() -> np.ndarray:
    """
    Build Player 1 constraint matrix E (Table 1 in paper)
    Sequences: empty, B_K, Ch_K, Ch_K Ca_K, Ch_K F_K, B_Q, Ch_Q, Ch_Q Ca_Q, Ch_Q F_Q,
               B_J, Ch_J, Ch_J Ca_J, Ch_J F_J
    """
    E = np.zeros((7, 13))

    E[0, 0] = 1  

    # IS1: Dealt K, choosing Bet or Check
    E[1, 0] = -1  
    E[1, 1] = 1   
    E[1, 2] = 1  

    # IS2: Dealt K, checked, opponent bet, choosing Call or Fold
    E[2, 2] = -1  
    E[2, 3] = 1   
    E[2, 4] = 1   

    # IS3: Dealt Q, choosing Bet or Check
    E[3, 0] = -1 
    E[3, 5] = 1   
    E[3, 6] = 1   

    # IS4: Dealt Q, checked, opponent bet, choosing Call or Fold
    E[4, 6] = -1  
    E[4, 7] = 1 
    E[4, 8] = 1 

    # IS5: Dealt J, choosing Bet or Check
    E[5, 0] = -1  
    E[5, 9] = 1   
    E[5, 10] = 1  

    # IS6: Dealt J, checked, opponent bet, choosing Call or Fold
    E[6, 10] = -1  
    E[6, 11] = 1  
    E[6, 12] = 1 

    return E


def build_matrix_F() -> np.ndarray:
    """
    Build Player 2 constraint matrix F
    Sequences: ∅, ca_Q, f_Q, b_Q, ch_Q, ca_J, f_J, b_J, ch_J, ca_K, f_K, b_K, ch_K
    """
    F = np.zeros((7, 13))

    # IS0: Root
    F[0, 0] = 1 

    # IS1: Dealt Q, p1 bet, choosing Call or Fold
    F[1, 0] = -1  
    F[1, 1] = 1   
    F[1, 2] = 1   

    # IS2: Dealt Q, P1 checked, choosing Bet or Check
    F[2, 0] = -1  
    F[2, 3] = 1  
    F[2, 4] = 1 

    # IS3: Dealt J, P1 bet, choosing call or Fold
    F[3, 0] = -1  
    F[3, 5] = 1 
    F[3, 6] = 1  

    # IS4: Dealt J, P1 checked, choosing bet or check
    F[4, 0] = -1 
    F[4, 7] = 1   
    F[4, 8] = 1  

    # IS5: Dealt K, P1 bet, choosing Call or Fold
    F[5, 0] = -1  
    F[5, 9] = 1  
    F[5, 10] = 1  

    # IS6: Dealt K, P1 checked, choosing Bet or Check
    F[6, 0] = -1  
    F[6, 11] = 1  
    F[6, 12] = 1  

    return F


def build_matrix_A() -> np.ndarray:
    """
    Build payoff matrix A for Player 1
    Payoffs (standard Kuhn):
      - bet-call showdown: winner +2, loser -2
      - bet-fold: bettor +1
      - check-check showdown: winner +1, loser -1
      - check-bet-call showdown: winner +2, loser -2
      - check-bet-fold (P1 folds): P1 -1
    """
    A = np.zeros((13, 13))

    p1_seq_to_idx = {
        ("K", "B"): 1,
        ("K", "Ch"): 2,
        ("K", "ChCa"): 3,
        ("K", "ChF"): 4,
        ("Q", "B"): 5,
        ("Q", "Ch"): 6,
        ("Q", "ChCa"): 7,
        ("Q", "ChF"): 8,
        ("J", "B"): 9,
        ("J", "Ch"): 10,
        ("J", "ChCa"): 11,
        ("J", "ChF"): 12,
    }

    p2_seq_to_idx = {
        ("Q", "Ca"): 1,
        ("Q", "F"): 2,
        ("Q", "b"): 3,
        ("Q", "ch"): 4,
        ("J", "Ca"): 5,
        ("J", "F"): 6,
        ("J", "b"): 7,
        ("J", "ch"): 8,
        ("K", "Ca"): 9,
        ("K", "F"): 10,
        ("K", "b"): 11,
        ("K", "ch"): 12,
    }

    rank = {"J": 0, "Q": 1, "K": 2}
    deals = [("K", "Q"), ("K", "J"), ("Q", "K"), ("Q", "J"), ("J", "K"), ("J", "Q")]
    prob = 1.0 / 6.0

    for c1, c2 in deals:
        win = 1.0 if rank[c1] > rank[c2] else -1.0

        # P1 bets, P2 calls=
        A[p1_seq_to_idx[(c1, "B")], p2_seq_to_idx[(c2, "Ca")]] += (2.0 * win) * prob

        # P1 bets, P2 folds
        A[p1_seq_to_idx[(c1, "B")], p2_seq_to_idx[(c2, "F")]] += 1.0 * prob

        # P1 checks, P2 checks 
        A[p1_seq_to_idx[(c1, "Ch")], p2_seq_to_idx[(c2, "ch")]] += (1.0 * win) * prob

        # P1 checks, P2 bets, P1 calls
        A[p1_seq_to_idx[(c1, "ChCa")], p2_seq_to_idx[(c2, "b")]] += (2.0 * win) * prob

        # P1 checks, P2 bets, P1 folds 
        A[p1_seq_to_idx[(c1, "ChF")], p2_seq_to_idx[(c2, "b")]] += (-1.0) * prob

    return A


def get_kuhn_matrices() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Get all Kuhn Poker matrices
    """
    E = build_matrix_E()
    F = build_matrix_F()
    A = build_matrix_A()

    e = np.zeros(7)
    e[0] = 1.0

    f = np.zeros(7)
    f[0] = 1.0

    return E, F, A, e, f


P1_SEQUENCES = [
    "∅",       
    "B_K",     
    "Ch_K",   
    "Ch_K Ca_K",  
    "Ch_K F_K",   
    "B_Q",    
    "Ch_Q",    
    "Ch_Q Ca_Q",  
    "Ch_Q F_Q",   
    "B_J",     
    "Ch_J",    
    "Ch_J Ca_J",  
    "Ch_J F_J",   
]

P2_SEQUENCES = [
    "∅",       # 0
    "ca_Q",    # 1: call with Q (facing bet)
    "f_Q",     # 2: fold with Q (facing bet)
    "b_Q",     # 3: bet with Q (facing check)
    "ch_Q",    # 4: check with Q (facing check)
    "ca_J",    # 5: call with J (facing bet)
    "f_J",     # 6: fold with J (facing bet)
    "b_J",     # 7: bet with J (facing check)
    "ch_J",    # 8: check with J (facing check)
    "ca_K",    # 9: call with K (facing bet)
    "f_K",     # 10: fold with K (facing bet)
    "b_K",     # 11: bet with K (facing check)
    "ch_K",    # 12: check with K (facing check)
]


KUHN_GAME = None


def get_kuhn_game() -> KuhnGame:
    """Get or create the Kuhn Poker game singleton"""
    global KUHN_GAME
    if KUHN_GAME is None:
        E, F, A, e, f = get_kuhn_matrices()
        KUHN_GAME = KuhnGame(
            E=E, F=F, A=A, e=e, f=f,
            p1_sequences=P1_SEQUENCES,
            p2_sequences=P2_SEQUENCES
        )
    return KUHN_GAME
