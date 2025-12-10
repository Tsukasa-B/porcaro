# porcaro_rl/actions/base.py
from __future__ import annotations
import abc
import torch
from typing import Protocol

class RobotLike(Protocol):
    def set_joint_velocity_target(self, cmd, joint_ids: list[int] | None = None) -> None: ...
    def set_joint_effort_target(self, cmf) -> None: ...
    # トルク化する時は set_joint_effort_target を追加で想定

class ActionController(abc.ABC):
    """アクション処理のインターフェース。env からはこれだけ呼ぶ。"""
    @abc.abstractmethod
    def apply(self, *,  # キーワード専用で可読性UP
              actions: torch.Tensor,          # (N,2)
              q: torch.Tensor,                # (N,DOF)
              joint_ids: tuple[int, int],     # (wrist_id, grip_id)
              robot: RobotLike) -> None:
        ...