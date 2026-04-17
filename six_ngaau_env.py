import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

# ── Types ──────────────────────────────────────────────────────────────────────

class Phase(Enum):
    SILENT   = "silent"    # Turn 1: simultaneous
    OPEN     = "open"      # Turn 2+: sequential AB-BA

class Rank(Enum):
    S = "S"  # Goal reached, token surplus, low turns
    A = "A"  # Goal reached or strategic surrender
    B = "B"  # Logical bankruptcy
    C = "C"  # Win by opponent collapse

@dataclass
class PlayerState:
    goal:    np.ndarray         # Private G, shape (6,)
    tokens:  int   = 3
    active:  bool  = True
    rank:    Optional[Rank] = None

@dataclass
class StepResult:
    player:       str
    action:       list[int]
    cost:         int
    collision:    list[int]     # Bits that collided (simultaneous turn only)
    net_progress: int           # Delta Hamming (negative = good)
    tokens_after: int
    status:       str           # CONTINUE | GOAL_REACHED | RANK_B_COLLAPSE
    rank:         Optional[Rank] = None


# ── Environment ────────────────────────────────────────────────────────────────

class MultiAgentSixNgaauEnv:
    """
    Six Ngaau: Two-player adversarial environment.

    Shared:   state S ∈ {0,1}^6, bit_history N ∈ ℤ^6 (the Entropy Pool)
    Private:  goal G_A for Player A, goal G_B for Player B
    Economy:  +3 tokens/turn, Cost(flip_i) = 2^(n_i - 1), exactly 2 flips/turn
    """

    N_BITS  = 6
    INCOME  = 3
    N_FLIPS = 2          # Inertia constraint

    def __init__(self, seed: int = None):
        self.rng = np.random.default_rng(seed)

        # ── Shared state ──
        self.state       = self.rng.integers(0, 2, self.N_BITS)
        self.bit_history = np.zeros(self.N_BITS, dtype=int)   # The Entropy Pool

        # ── Private goals (guaranteed distinct) ──
        g_a = self.rng.integers(0, 2, self.N_BITS)
        g_b = self.rng.integers(0, 2, self.N_BITS)
        while np.array_equal(g_a, g_b):
            g_b = self.rng.integers(0, 2, self.N_BITS)

        self.players = {
            "A": PlayerState(goal=g_a),
            "B": PlayerState(goal=g_b),
        }

        self.turn      = 0
        self.phase     = Phase.SILENT
        self.ab_ba_seq = self._ab_ba_generator()   # yields player order each turn

    # ── Public API ────────────────────────────────────────────────────────────

    def observe(self, player_id: str) -> dict:
        """
        Returns the observation vector for an agent.
        Agents see the full shared state and FULL bit_history.
        They do NOT see the opponent's goal.
        """
        p = self.players[player_id]
        return {
            "state":       self.state.copy(),
            "bit_history": self.bit_history.copy(),    # The "heat map" of the board
            "goal":        p.goal.copy(),
            "hamming":     int(np.sum(self.state != p.goal)),
            "tokens":      p.tokens,
            "turn":        self.turn,
            "phase":       self.phase.value,
            # Cost forecast: what each bit would cost to flip RIGHT NOW
            "cost_vector": self._cost_vector(),
        }

    def step_silent(
        self,
        action_a: list[int],
        action_b: list[int],
    ) -> tuple[StepResult, StepResult]:
        """
        Turn 1: simultaneous submission. 
        Neither player sees the other's action before committing.
        Returns results for both players after collision resolution.
        """
        assert self.phase == Phase.SILENT, "step_silent only valid on Turn 1."
        self._validate_action(action_a)
        self._validate_action(action_b)

        # Income posts before cost assessment (explicit rule from prior session)
        for p in self.players.values():
            p.tokens += self.INCOME
        self.turn += 1

        result_a, result_b = self._resolve_simultaneous(action_a, action_b)

        self.phase = Phase.OPEN
        return result_a, result_b

    def step_sequential(self, player_id: str, action: list[int]) -> StepResult:
        """
        Turn 2+: sequential moves following AB-BA order.
        The second mover in each round observes updated state + bit_history.
        """
        assert self.phase == Phase.OPEN, "step_sequential only valid after Turn 1."
        expected = next(self.ab_ba_seq)
        assert player_id == expected, (
            f"Turn order violation: expected {expected}, got {player_id}."
        )
        self._validate_action(action)

        # Income posts at the START of each player's sub-turn in AB-BA
        p = self.players[player_id]
        p.tokens += self.INCOME
        self.turn += 1

        result = self._apply_action(player_id, action, collisions=[])
        return result

    # ── Core Mechanics ────────────────────────────────────────────────────────

    def _resolve_simultaneous(
        self,
        action_a: list[int],
        action_b: list[int],
    ) -> tuple[StepResult, StepResult]:
        """
        Collision resolution for the Silent Turn.

        Collision rule: if both flip the same bit,
          - The bit is incremented TWICE in bit_history (entropy still accumulates)
          - The bit's value returns to original (net-zero state progress)
          - Both players pay their respective incremental cost
          - Order: A resolves first → B's cost is computed against post-A history
        """
        collisions = list(set(action_a) & set(action_b))

        # ── Phase 1: Apply A's flips to the shared entropy pool ──
        # (B observes the "hotter" board even in a simultaneous turn,
        #  because entropy is settled in submission order A→B)
        result_a = self._apply_action("A", action_a, collisions)

        # ── Phase 2: B pays against post-A history ──
        # This IS the cruelest mechanic. B submitted blind but pays informed costs.
        result_b = self._apply_action("B", action_b, collisions)

        # ── Phase 3: Un-flip collision bits in STATE only (not in history) ──
        # The entropy remains; the physical bit is restored.
        for bit in collisions:
            self.state[bit] ^= 1   # flip back: net effect = no state change

        return result_a, result_b

    def _apply_action(
        self,
        player_id: str,
        action: list[int],
        collisions: list[int],
    ) -> StepResult:
        """
        Commits a single player's action:
        1. Compute cost against current (possibly already-updated) bit_history
        2. Bankruptcy check
        3. Commit: update bit_history, state, tokens
        4. Check terminal conditions
        """
        p = self.players[player_id]
        hamming_before = int(np.sum(self.state != p.goal))

        # Cost uses CURRENT history (post any prior resolution in this turn)
        cost = self.query_cost(action)

        if p.tokens < cost:
            p.active = False
            p.rank   = Rank.B
            # Award opponent Rank C if still active
            opp_id = "B" if player_id == "A" else "A"
            if self.players[opp_id].active:
                self.players[opp_id].rank = Rank.C
            return StepResult(
                player       = player_id,
                action       = action,
                cost         = cost,
                collision    = collisions,
                net_progress = 0,
                tokens_after = p.tokens,
                status       = "RANK_B_COLLAPSE",
                rank         = Rank.B,
            )

        # Commit: update entropy pool and state
        p.tokens -= cost
        for idx in action:
            self.bit_history[idx] += 1
            self.state[idx] ^= 1

        hamming_after = int(np.sum(self.state != p.goal))

        # Terminal: goal reached?
        if hamming_after == 0:
            p.active = False
            p.rank   = self._score(player_id)
            return StepResult(
                player       = player_id,
                action       = action,
                cost         = cost,
                collision    = collisions,
                net_progress = hamming_after - hamming_before,
                tokens_after = p.tokens,
                status       = "GOAL_REACHED",
                rank         = p.rank,
            )

        return StepResult(
            player       = player_id,
            action       = action,
            cost         = cost,
            collision    = collisions,
            net_progress = hamming_after - hamming_before,
            tokens_after = p.tokens,
            status       = "CONTINUE",
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def query_cost(self, flip_indices: list[int]) -> int:
        """Pure read: cost of flipping these bits at current entropy levels."""
        return sum(2 ** self.bit_history[i] for i in flip_indices)
        # Note: bit_history[i] + 1 - 1 = bit_history[i], since n = history+1
        # and cost = 2^(n-1) = 2^(history[i])

    def _cost_vector(self) -> np.ndarray:
        """The 'heat map': cost to flip each bit right now."""
        return 2 ** self.bit_history   # elementwise

    def _score(self, player_id: str) -> Rank:
        p = self.players[player_id]
        if self.turn <= 3 and p.tokens >= 3:
            return Rank.S
        return Rank.A

    def _validate_action(self, action: list[int]):
        assert len(action) == self.N_FLIPS, (
            f"Inertia constraint violated: need exactly {self.N_FLIPS} flips."
        )
        assert len(set(action)) == self.N_FLIPS, "Duplicate indices in action."
        assert all(0 <= i < self.N_BITS for i in action), "Bit index out of range."

    @staticmethod
    def _ab_ba_generator():
        """Yields player order: A, B, B, A, A, B, B, A ... (AB-BA pattern)"""
        pattern = ["A", "B", "B", "A"]
        i = 0
        while True:
            yield pattern[i % 4]
            i += 1
