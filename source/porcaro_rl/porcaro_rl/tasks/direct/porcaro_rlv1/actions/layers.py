# source/porcaro_rl/porcaro_rl/tasks/direct/porcaro_rlv1/actions/layers.py
import torch
import torch.nn as nn
from typing import List

class SimpleMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_units: List[int], activation: str = "relu"):
        super().__init__()
        layers = []
        in_d = input_dim
        for h in hidden_units:
            layers.append(nn.Linear(in_d, h))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            in_d = h
        layers.append(nn.Linear(in_d, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class PressureNet(SimpleMLP):
    """NN1 Wrapper"""
    pass

class ForceNet(SimpleMLP):
    """NN2 Wrapper"""
    pass