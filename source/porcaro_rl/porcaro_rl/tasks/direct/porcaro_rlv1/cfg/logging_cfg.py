# cfg/logging_cfg.py
from __future__ import annotations
from isaaclab.utils import configclass

@configclass
class LoggingCfg:
    """データロギングの設定"""
    enabled: bool = False
    filepath: str = "simulation_log.csv"
    
    # CSVヘッダー定義
    # LoggingManagerが出力するリストの順序と完全に一致させています
    headers: list[str] = [
        # --- Time ---
        "time_s",
        
        # --- Actions (Modeによって意味が変わるため汎用名にする) ---
        "action_0", # EP: theta_eq / Pressure: P_cmd_DF
        "action_1", # EP: K_wrist / Pressure: P_cmd_F
        "action_2", # EP: K_grip  / Pressure: P_cmd_G
        
        # --- Joint States (Degree) ---
        "q_wrist_deg",  "q_grip_deg",
        "qd_wrist_deg", "qd_grip_deg",
        
        # --- Contact Forces (Z-axis only) ---
        "force_z",      # 瞬時値 (Newtons)
        "f1_score",     # 打撃判定されたピーク値
        
        # --- Internal Pressures (MPa) ---
        "P_cmd_DF", "P_cmd_F", "P_cmd_G", # 指令値
        "P_out_DF", "P_out_F", "P_out_G", # 遅延後の実際値
        
        # --- Torques (Nm) ---
        "tau_wrist", "tau_grip",
    ]

@configclass
class RewardLoggingCfg:
    """ステップごとの報酬ロギング設定"""
    enabled: bool = False
    filepath: str = "reward_log.csv"
    headers: list[str] = ["time_s", "step_reward"]