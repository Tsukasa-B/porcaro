# logging/logging_manager.py
from __future__ import annotations
import torch
from typing import Sequence, List, Any

from ..cfg.logging_cfg import LoggingCfg, RewardLoggingCfg
from .datalogger import DataLogger

class LoggingManager:
    """
    シミュレーションのデータロギングを管理するクラス。
    """
    def __init__(self, 
                 cfg: LoggingCfg, 
                 reward_cfg: RewardLoggingCfg,
                 dof_idx: Sequence[int], 
                 num_envs: int, 
                 device: str | torch.device,
                 dt: float,
                 decimation: int,
                 ):
        
        self.cfg = cfg
        self.reward_cfg = reward_cfg
        self.dof_idx = dof_idx
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.dt = dt
        self.decimation = decimation
        
        self.logger: DataLogger | None = None
        self.reward_logger: DataLogger | None = None
        
        # Cfg に基づいてロガーを初期化
        if cfg.enabled:
            print(f"[LoggingManager] ロギング有効 (Simulation Data)")
            self.logger = DataLogger(filepath=cfg.filepath, headers=cfg.headers)

        if reward_cfg.enabled:
            print(f"[LoggingManager] ロギング有効 (Reward Data)")
            self.reward_logger = DataLogger(filepath=reward_cfg.filepath, headers=reward_cfg.headers)

        # 内部変数
        self.current_time_s = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.log_step_counter: int = 0
        self.log_save_interval: int = 2000 
        self.step_data_buffer: List[Any] | None = None

    def force_enable(self):
        """Playスクリプトなどから強制的に有効化する場合"""
        if self.logger is None:
            print(f"[LoggingManager] ロギングを強制有効化します")
            self.logger = DataLogger(filepath=self.cfg.filepath, headers=self.cfg.headers)

    def force_enable_reward_log(self):
        """Playスクリプトなどから強制的に有効化する場合"""
        if self.reward_logger is None:
            print(f"[LoggingManager] 報酬ロギングを強制有効化します")
            self.reward_logger = DataLogger(filepath=self.reward_cfg.filepath, headers=self.reward_cfg.headers)

    def update_time(self, dt_ctrl: float):
        """時刻更新 (これは学習に必要なため常に実行)"""
        self.current_time_s.add_(dt_ctrl)

    def reset_idx(self, env_ids: Sequence[int]):
        """リセット"""
        self.current_time_s[env_ids] = 0.0
    
    def buffer_step_data(self, q_full, qd_full, telemetry, actions):
        """
        データを一時バッファに保存。
        ★修正: ロガーが無効なら即リターンし、リスト作成やメモリアクセスを回避
        """
        if self.logger is None:
            return

        env_idx = 0 
        try:
            # テンソル操作とリスト作成のコストはロガー有効時のみ払う
            time_s_tensor = self.current_time_s[env_idx].reshape(-1)
            act = actions[env_idx].reshape(-1)
            q = q_full[env_idx, self.dof_idx].reshape(-1)
            qd = qd_full[env_idx, self.dof_idx].reshape(-1)
            
            if telemetry:
                p_cmd = telemetry["P_cmd"][env_idx].reshape(-1)
                p_out = telemetry["P_out"][env_idx].reshape(-1)
                tau_w = telemetry["tau_w"][env_idx].reshape(-1)
                tau_g = telemetry["tau_g"][env_idx].reshape(-1)
            else:
                p_cmd = torch.zeros(3, device=self.device)
                p_out = torch.zeros(3, device=self.device)
                tau_w = torch.zeros(1, device=self.device)
                tau_g = torch.zeros(1, device=self.device)

            self.step_data_buffer = [
                time_s_tensor, act, q, qd, p_cmd, p_out, tau_w, tau_g
            ]
        except Exception:
            self.step_data_buffer = None

    def finalize_log_step(self, peak_force, f1_force, step_reward):
        """
        バッファデータと最新データを組み合わせてログ書き込み
        ★修正: 両方のロガーが無効なら即リターン
        """
        if self.logger is None and self.reward_logger is None:
            return  # << ここで完全に処理を打ち切る

        env_idx = 0

        # --- 1. シミュレーションログ ---
        if self.logger is not None and self.step_data_buffer is not None:
            try:
                rows_to_log: list[list[float]] = []
                base_data = self.step_data_buffer
                time_s_end = base_data[0]
                f1 = f1_force[env_idx].reshape(-1)
                
                # CPUへの転送(tolist)を含む重いループ処理
                for i in range(self.decimation):
                    history_idx = (self.decimation - 1) - i
                    time_s_curr = time_s_end - (self.decimation - i) * self.dt
                    force = peak_force[env_idx, history_idx].reshape(-1)
                    
                    row_tensors = list(base_data)
                    row_tensors[0] = time_s_curr.reshape(-1)
                    row_tensors.insert(4, force)
                    row_tensors.append(f1)
                
                    # 連結してリスト化 (最も重い処理)
                    data_row = torch.cat(row_tensors).cpu().tolist()
                    rows_to_log.append(data_row)

                if rows_to_log:
                    self.logger.add_data_batch(rows_to_log)

            except Exception as e:
                print(f"[WARN] Log error: {e}")
            
            self.step_data_buffer = None

        # --- 2. 報酬ログ ---
        if self.reward_logger is not None:
            try:
                time_s = self.current_time_s[env_idx].cpu().item()
                reward_val = step_reward[env_idx].cpu().item()
                self.reward_logger.add_step([time_s, reward_val])
            except Exception:
                pass

        # --- 3. 定期保存 ---
        # ここもロガー有効時のみ実行するように変更
        self.log_step_counter += 1
        if (self.log_step_counter % self.log_save_interval) == 0:
            # print も save も、ロガーが存在するときだけ行う
            if self.logger:
                print(f"[DataLogger] 定期保存 (Sim) ...")
                self.logger.save()
            if self.reward_logger:
                # 報酬ログだけの時はうるさいのでprintしないか、控えめに
                self.reward_logger.save()

    def save_on_exit(self):
        """終了時の保存"""
        if self.logger:
            print("[INFO] Saving sim logs...")
            self.logger.save()
        if self.reward_logger:
            print("[INFO] Saving reward logs...")
            self.reward_logger.save()