"""Agent implementations for opponent modeling experiments"""

from .opponent import DirichletOpponent, SwitchingOpponent, PassiveAggressiveSwitcher, AdversarialOpponent
from .fmap import FMAPAgent
from .nash import NashAgent

__all__ = [
    "DirichletOpponent",
    "SwitchingOpponent",
    "PassiveAggressiveSwitcher",
    "AdversarialOpponent",
    "FMAPAgent",
    "NashAgent",
]
