"""
Nash Equilibrium Agent that plays nash strategy for Kuhn Poker: used as benchmark
"""

import numpy as np
from typing import Optional, Tuple

try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False


class NashAgent:
    """Plays a static Nash equilibrium strategy for Kuhn Poker"""

    def __init__(self, game, player: int = 1, use_gurobi: bool = True):
        """
        Nahs args:
            game: KuhnGame object
            player: Player 1 or player 2
            use_gurobi: Whether to use Gurobi (which is faster) or scipy
        """
        self.game = game
        self.player = player
        self.use_gurobi = use_gurobi and GUROBI_AVAILABLE

        self.nash_strategy = self._compute_nash()

    def _compute_nash(self) -> np.ndarray:
        if self.player == 1:
            return self._compute_nash_p1()
        else:
            return self._compute_nash_p2()

    def _compute_nash_p1(self) -> np.ndarray:
        if self.use_gurobi:
            return self._compute_nash_p1_gurobi()
        else:
            return self._compute_nash_p1_scipy()

    def _compute_nash_p1_gurobi(self) -> np.ndarray:
        try:
            E = self.game.E
            F = self.game.F
            A = self.game.A
            e = self.game.e
            f = self.game.f
            n1 = E.shape[1]
            n2 = F.shape[0]

            model = gp.Model("Nash_P1")
            model.setParam('OutputFlag', 0)

            x = model.addMVar(n1, lb=0, name="x")
            q = model.addMVar(n2, lb=-GRB.INFINITY, name="q")

            model.setObjective(-q @ f, GRB.MAXIMIZE)

            model.addConstr((-A).T @ x - F.T @ q <= 0, name="payoff")

            model.addConstr(E @ x == e, name="sequence")

            model.optimize()

            if model.status == GRB.OPTIMAL:
                return x.X.copy()
            else:
                from ..core import get_uniform_strategy
                return get_uniform_strategy(E, e)

        except Exception as e:
            print(f"Warning: Gurobi Nash computation failed: {e}")
            from ..core import get_uniform_strategy
            return get_uniform_strategy(self.game.E, self.game.e)

    def _compute_nash_p1_scipy(self) -> np.ndarray:
        """
        Compute P1 Nash using scipy for the primal formulation for finding a maximin strategy
        """
        from ..core import get_uniform_strategy
        try:
            from scipy.optimize import linprog
        except ImportError:
            print("Warning: scipy not available; falling back to uniform P1 strategy")
            return get_uniform_strategy(self.game.E, self.game.e)

        E = self.game.E
        F = self.game.F
        A = self.game.A
        e = self.game.e
        f = self.game.f
        n1 = E.shape[1]  # P1 sequences
        m2 = F.shape[0]  # P2 information sets 
        n2 = F.shape[1]  # P2 sequences

        c = np.concatenate([np.zeros(n1), f])

        A_eq = np.hstack([E, np.zeros((E.shape[0], m2))])
        b_eq = e.copy()

        A_ub = np.hstack([(-A).T, -F.T])
        b_ub = np.zeros(n2)

        bounds = [(0, None)] * n1 + [(None, None)] * m2

        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                         bounds=bounds, method='highs')

        if result.success:
            return result.x[:n1]
        return get_uniform_strategy(E, e)

    def _compute_nash_p2(self) -> np.ndarray:
        if self.use_gurobi:
            return self._compute_nash_p2_gurobi()
        else:
            return self._compute_nash_p2_scipy()

    def _compute_nash_p2_gurobi(self) -> np.ndarray:
        try:
            E = self.game.E
            F = self.game.F
            A = self.game.A
            e = self.game.e
            f = self.game.f
            n1 = E.shape[0]
            n2 = F.shape[1]

            model = gp.Model("Nash_P2")
            model.setParam('OutputFlag', 0)

            y = model.addMVar(n2, lb=0, name="y")
            p = model.addMVar(n1, lb=-GRB.INFINITY, name="p")

            model.setObjective(e @ p, GRB.MINIMIZE)

            model.addConstr(-A @ y + E.T @ p >= 0, name="payoff")

            model.addConstr(F @ y == f, name="sequence")

            model.optimize()

            if model.status == GRB.OPTIMAL:
                return y.X.copy()
            else:
                print(f"Warning: Nash computation failed with status {model.status}")
                from ..core import get_uniform_strategy
                return get_uniform_strategy(F, f)

        except Exception as e:
            print(f"Warning: Gurobi Nash computation failed: {e}")
            from ..core import get_uniform_strategy
            return get_uniform_strategy(self.game.F, self.game.f)

    def _compute_nash_p2_scipy(self) -> np.ndarray:
        """
        Compute P2 Nash using scipy by calculating the dual formulation for finding a minimax strategy.
        """
        from ..core import get_uniform_strategy
        try:
            from scipy.optimize import linprog
        except ImportError:
            print("Warning: scipy not available; falling back to uniform P2 strategy")
            return get_uniform_strategy(self.game.F, self.game.f)

        E = self.game.E
        F = self.game.F
        A = self.game.A
        e = self.game.e
        f = self.game.f
        m1 = E.shape[0]  # P1 information sets
        n1 = E.shape[1]  # P1 sequences
        n2 = F.shape[1]  # P2 sequences

        c = np.concatenate([np.zeros(n2), e])

        A_eq = np.hstack([F, np.zeros((F.shape[0], m1))])
        b_eq = f.copy()

        A_ub = np.hstack([A, -E.T])
        b_ub = np.zeros(n1)

        bounds = [(0, None)] * n2 + [(None, None)] * m1

        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                         bounds=bounds, method='highs')

        if result.success:
            return result.x[:n2]
        else:
            print(f"Warning: scipy Nash P2 computation failed: {result.message}")
            return get_uniform_strategy(F, f)

    def get_strategy(self) -> np.ndarray: #get nash strat
        return self.nash_strategy.copy()

    def act(self) -> np.ndarray:
        return self.get_strategy()

    def compute_value_against(self, opponent_strategy: np.ndarray) -> float:
        #Compute expected value of Nash strategy against opponent.
        if self.player == 1:
            return np.dot(self.nash_strategy, np.dot(self.game.A, opponent_strategy))
        else:
            return -np.dot(opponent_strategy, np.dot(self.game.A, self.nash_strategy))


def compute_best_nash(game, opponent_strategy: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Find the best Nash equilibrium against a specific opponent returned as a tuple (best_nash_strategy, value_against_opponent)
    """
    agent = NashAgent(game, player=1)
    nash_strategy = agent.get_strategy()
    value = np.dot(nash_strategy, np.dot(game.A, opponent_strategy))

    return nash_strategy, value


def compute_game_value(game) -> float:
    #Compute the game value (Nash equilibrium value).
    nash_p1 = NashAgent(game, player=1).get_strategy()
    nash_p2 = NashAgent(game, player=2).get_strategy()
    value = np.dot(nash_p1, np.dot(game.A, nash_p2))
    return value
