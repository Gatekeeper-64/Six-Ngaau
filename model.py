# ==========================================
# 2. 神經網路大腦 (23維輸入)
# ==========================================
import torch
import torch.nn as nn
from itertools import combinations

ACTIONS = list(combinations(range(6), 2))

class PPOAgent(nn.Module):
    def __init__(self):
        super(PPOAgent, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(23, 128),  # 💥 配合觀測值改為 23 維
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.policy = nn.Linear(128, len(ACTIONS))
        self.value = nn.Linear(128, 1)

    def forward(self, x):
        x = self.shared(x)
        return self.policy(x), self.value(x)
