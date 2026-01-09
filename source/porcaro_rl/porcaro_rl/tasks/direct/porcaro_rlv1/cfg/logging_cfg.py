# source/porcaro_rl/porcaro_rl/tasks/direct/porcaro_rlv1/cfg/logging_cfg.py
from __future__ import annotations
from isaaclab.utils import configclass

@configclass
class LoggingCfg:
    """シミュレーションデータ（物理状態・アクション等）のロギング設定"""
    enabled: bool = True
    filepath: str = "simulation_log.csv"

@configclass
class RewardLoggingCfg:
    """報酬計算の詳細ロギング設定"""
    enabled: bool = True
    filepath: str = "reward_log.csv"