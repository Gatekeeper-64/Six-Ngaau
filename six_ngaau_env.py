# ==========================================
# 1. 基礎設定與環境 (V.24 積分賽制 | 23維觀測 | 租金加速燈)
# ==========================================
import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Optional

class Phase(Enum):
    SILENT = "silent"
    OPEN = "open"

class Rank(Enum):
    S = "S" # 20回合內達成 64 分
    A = "A" # 達成 64 分
    B = "B" # 破產
    C = "C" # 對手破產

@dataclass
class StepResult:
    player: str
    action: list[int]
    cost: int
    collision: list[int]
    net_progress: int
    tokens_after: int
    status: str
    turn: int = 0
    rank: Optional[Rank] = None

class Player:
    def __init__(self, id, goal, tokens):
        self.id = id
        self.goal = goal
        self.tokens = tokens
        self.active = True
        self.rank = None
        self.last_score_match = 0
        self.total_points = 0
        self.last_round_rent = 0  # 💥 這裡：新增租金記憶變數

class MultiAgentSixNgaauEnv:
    N_BITS = 6
    INCOME = 5 
    N_FLIPS = 2
    GOAL_SCORE = 64

    def __init__(self, seed: int = None):
        self.reset(seed)

    def reset(self, seed: int = None):
        self.rng = np.random.default_rng(seed)
        self.state = self.rng.integers(0, 2, self.N_BITS)
        self.bit_history = np.zeros(self.N_BITS, dtype=int)
        
        g_a = self.rng.integers(0, 2, self.N_BITS)
        while np.sum(self.state != g_a) % 2 != 0:
            g_a = self.rng.integers(0, 2, self.N_BITS)
        g_b = self.rng.integers(0, 2, self.N_BITS)
        while np.array_equal(g_a, g_b) or np.sum(self.state != g_b) % 2 != 0:
            g_b = self.rng.integers(0, 2, self.N_BITS)
            
        self.players = {"A": Player("A", g_a, 5), "B": Player("B", g_b, 5)}
        self.turn = 0
        self.sub_turn = 0
        self.phase = Phase.SILENT
        self.ab_ba_seq = self._ab_ba_generator()
        
        self.global_light_1 = 0.0 # 💥 一般燈 (租金加速)
        self.global_light_2 = 0.0 # 💥 32分燈 (進入下半場)
        
        return self.observe("A"), self.observe("B")

    def observe(self, player_id: str):
        p = self.players[player_id]
        opp_id = "B" if player_id == "A" else "A"
        opp = self.players[opp_id]
        
        current_match = np.sum(self.state == p.goal)
        delta_light = 1.0 if (self.turn > 1 and current_match > p.last_score_match) else 0.0
        threshold_light = 1.0 if current_match >= 4 else 0.0
        p.last_score_match = current_match 
        
        obs = np.concatenate([
            self.state.astype(float),
            p.goal.astype(float),
            (2 ** self.bit_history).astype(float),
            np.array([float(p.tokens)]),
            np.array([self.global_light_1, self.global_light_2]), # 公共燈號
            np.array([p.total_points / float(self.GOAL_SCORE), opp.total_points / float(self.GOAL_SCORE)])
        ])
        return obs

    def step_silent(self, action_a: list[int], action_b: list[int]):
        collisions = list(set(action_a) & set(action_b))
        res_a = self._apply_action("A", action_a, collisions)
        res_b = self._apply_action("B", action_b, collisions)
        for bit in collisions: self.state[bit] ^= 1
        self.phase = Phase.OPEN
        self.turn += 1
        return self._to_dict(res_a), self._to_dict(res_b)

    def step_sequential(self, player_id: str, action: list[int]) -> dict:
        _ = next(self.ab_ba_seq)
        res = self._apply_action(player_id, action, collisions=[])
        self.sub_turn += 1
        self.turn += 1
        out = self._to_dict(res)
        out["round_complete"] = (self.sub_turn % 2 == 0)
        return out

    def _apply_action(self, player_id: str, action: list[int], collisions: list[int]) -> StepResult:
        p = self.players[player_id]
        cost = sum(2 ** self.bit_history[i] for i in action)
        if p.tokens < cost:
            p.active, p.rank = False, Rank.B
            opp_id = "B" if player_id == "A" else "A"
            if self.players[opp_id].active: self.players[opp_id].rank = Rank.C
            return StepResult(player_id, action, cost, collisions, 0, p.tokens, "RANK_B_COLLAPSE", turn=self.turn, rank=Rank.B)
        p.tokens -= cost
        for idx in action:
            self.bit_history[idx] += 1
            self.state[idx] ^= 1
        return StepResult(player_id, action, cost, collisions, 0, p.tokens, "CONTINUE", turn=self.turn, rank=None)

    def resolve_round(self):
        """ 💥 每回合結算：判定租金加速燈與 32 分燈 """
        # 1. 32分燈：有人總分超過一半就亮
        someone_reached_32 = any(p.total_points >= 32 for p in self.players.values())
        self.global_light_2 = 1.0 if someone_reached_32 else 0.0

        # 2. 租金與加速燈判定
        any_rent_accelerated = False
        for p_id, p in self.players.items():
            if p.active:
                current_rent = int(np.sum(self.state == p.goal))
                if current_rent > p.last_round_rent:
                    any_rent_accelerated = True # 💥 有人這回合收的比上回合多！
                
                p.total_points += current_rent
                p.last_round_rent = current_rent # 更新記憶
                
                if p.total_points >= self.GOAL_SCORE:
                    p.active = False
                    p.rank = Rank.S if self.turn <= 20 else Rank.A
        
        self.global_light_1 = 1.0 if any_rent_accelerated else 0.0
        
        self.bit_history = np.zeros(6) 
        for p in self.players.values(): 
            if p.active: p.tokens += self.INCOME

    def _ab_ba_generator(self):
        while True:
            for p_id in ["A", "B", "B", "A"]: yield p_id

    def _to_dict(self, sr: StepResult) -> dict:
        p = self.players[sr.player]
        status = "GOAL_REACHED" if p.rank in [Rank.S, Rank.A] else sr.status
        return {"player": sr.player, "cost": sr.cost, "status": status, 
                "rank": p.rank, "tokens_after": sr.tokens_after, "turn": sr.turn}
