# cfg/logging_cfg.py
from __future__ import annotations
from isaaclab.utils import configclass

@configclass
class LoggingCfg:
    """データロギングの設定"""
    enabled: bool = True  # ロギングを有効にするか
    filepath: str = "simulation_log.csv" # 保存先ファイル名
    
    # CSVのヘッダー（列名）
    # ※ この順序と、env.py で渡すデータの順序を厳密に一致させる必要があります
    headers: list[str] = [
        "time_s",
        "q_wrist", "q_grip",
        "qd_wrist", "qd_grip",
        "stick_force_x", "stick_force_y", "stick_force_z",
        "P_cmd_DF", "P_cmd_F", "P_cmd_G",
        "P_out_DF", "P_out_F", "P_out_G",
        "tau_w", "tau_g",
        "f1_force",
    ]

@configclass
class RewardLoggingCfg:
    """ステップごとの報酬ロギング設定"""
    enabled: bool = True
    filepath: str = "reward_log.csv"
    # ヘッダー (時間とステップ報酬)
    headers: list[str] = ["time_s", "step_reward"]