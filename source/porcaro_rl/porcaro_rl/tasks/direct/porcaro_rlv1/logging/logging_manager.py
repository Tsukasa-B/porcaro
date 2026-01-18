# source/porcaro_rl/porcaro_rl/tasks/direct/porcaro_rlv1/logging/logging_manager.py
from __future__ import annotations
import torch
import numpy as np
from typing import TYPE_CHECKING, Any, List, Sequence
from .datalogger import DataLogger

if TYPE_CHECKING:
    from ..porcaro_rl_env import PorcaroRLEnv

class LoggingManager:
    def __init__(self, 
                 env: PorcaroRLEnv, 
                 dt: float, 
                 log_filepath: str | None = None,
                 enable_logging: bool = False):
        
        self.env = env
        self.device = env.device
        self.dt = dt # 物理ステップ (sim.dt)
        self.num_envs = env.num_envs
        self.enable_logging = enable_logging
        
        # 時刻管理 (Tensorではなくシンプルなfloat/numpyで管理して確実性を高める)
        # 各環境ごとの時刻を持つが、通常は同期しているため代表値で管理も可能だが、
        # Sim-to-Realの厳密性のためにTensorで保持する。
        self.current_time_s = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)

        # ★修正: 物理ステップごとの生データを貯める一時バッファ (Decimationループごとにクリアされる)
        self.step_data_buffer: list[list[float]] = []

        self.logger: DataLogger | None = None
        if enable_logging and log_filepath is not None:
            # ヘッダー定義
            headers = [
                "time_s",
                # Action (3)
                "action_0", "action_1", "action_2",
                # Joint State (4)
                "q_wrist_deg", "q_grip_deg",
                "qd_wrist_deg", "qd_grip_deg",
                # Force (1) & Score (1)
                "force_z", "f1_score",
                # Rhythm Info (2)
                "target_force", "target_bpm",
                # Telemetry (Command Pressure etc)
                "P_cmd_DF", "P_cmd_F", "P_cmd_G",
                "P_out_DF", "P_out_F", "P_out_G",
                # Rewards (1)
                "step_reward"
            ]
            self.logger = DataLogger(filepath=log_filepath, headers=headers)
            print(f"[LoggingManager] Logging enabled -> {log_filepath}")

    def reset_idx(self, env_ids: torch.Tensor):
        """指定された環境の時刻をリセット"""
        if self.enable_logging:
            self.current_time_s[env_ids] = 0.0
            # バッファは一括管理なのでここではクリアしない（stepの先頭でクリアされる）

    def update_time(self, dt_step: float):
        """物理ステップ時間を進める"""
        if self.enable_logging:
            self.current_time_s += dt_step

    def buffer_step_data(self, 
                         q_full: torch.Tensor, 
                         qd_full: torch.Tensor, 
                         telemetry: dict, 
                         actions: torch.Tensor,
                         current_sim_time: float, # デバッグ用(未使用)
                         target_force: float = 0.0,
                         target_bpm: float = 0.0):
        """
        物理サブステップごとのデータをバッファに追加する。
        ここでのデータはまだファイルには書き込まれない。
        """
        if not self.enable_logging:
            return

        # 環境0 (代表) のデータを取得
        env_idx = 0
        
        # 1. 時刻
        t = self.current_time_s[env_idx].item()
        
        # 2. アクション
        act = actions[env_idx].detach().cpu().numpy().tolist() # [3]
        
        # 3. 関節状態 (Rad -> Deg)
        # dof_idx は env側で管理されているが、ここでは簡易的に wrist(0), grip(1) と仮定
        # 必要なら self.env.dof_idx を参照
        wrist_deg = np.degrees(q_full[env_idx, self.env.dof_idx[0]].item())
        grip_deg  = np.degrees(q_full[env_idx, self.env.dof_idx[1]].item())
        wrist_vel_deg = np.degrees(qd_full[env_idx, self.env.dof_idx[0]].item())
        grip_vel_deg  = np.degrees(qd_full[env_idx, self.env.dof_idx[1]].item())

        # 4. テレメトリ (P_cmd, P_out)
        def get_tel(key, idx):
            val = telemetry.get(key, None)
            if val is not None:
                return val[env_idx, idx].item()
            return 0.0

        p_cmd = [get_tel("P_cmd", i) for i in range(3)]
        p_out = [get_tel("P_out", i) for i in range(3)]

        # 5. 行の構築
        # ForceとReward, F1はまだ確定していないのでプレースホルダ(0.0)を入れる
        # 構造: [Time, Act(3), Q(2), Qd(2), Force(1), F1(1), Tgt(2), P_cmd(3), P_out(3), Rew(1)]
        row = [
            t,
            *act,
            wrist_deg, grip_deg,
            wrist_vel_deg, grip_vel_deg,
            0.0, 0.0, # Force, F1 (Placeholder)
            target_force, target_bpm,
            *p_cmd,
            *p_out,
            0.0 # Step Reward (Placeholder)
        ]
        
        # バッファに追加
        self.step_data_buffer.append(row)

    def finalize_log_step(self, peak_force: torch.Tensor, f1_force: torch.Tensor, step_reward: torch.Tensor):
        """
        1回のRLステップ（Decimationループ終了後）に呼ばれる。
        バッファ内のプレースホルダを実際のForce/Rewardで埋め、Loggerに送る。
        
        Args:
            peak_force: (num_envs, decimation, 3) または (num_envs, 3) 
                        物理ステップごとの履歴、または集約値。
                        ★修正: ここでは (N, Decimation, 3) の履歴を受け取ることを前提とする。
            f1_force:   (num_envs,) F1スコア
            step_reward:(num_envs,) 報酬
        """
        if not self.enable_logging or self.logger is None:
            # バッファをクリアして終了
            self.step_data_buffer.clear()
            return

        env_idx = 0
        num_buffered = len(self.step_data_buffer)
        
        # Force履歴の取得 (CPUへ)
        # peak_force は [N, Decimation, 3] のはず
        forces_cpu = peak_force[env_idx].detach().cpu() # [Decimation, 3]
        
        # バッファサイズと履歴サイズが一致しているか確認
        # (sim.stepのエラーなどでズレる場合への安全策)
        limit = min(num_buffered, forces_cpu.shape[0])

        reward_val = step_reward[env_idx].item()
        f1_val = f1_force[env_idx].item()

        rows_to_write = []

        for i in range(limit):
            row = self.step_data_buffer[i]
            
            # Force (Z軸) を埋める
            # センサーデータは [Decimation, 3] なので i 番目のデータを使う
            f_z = forces_cpu[i, 2].item()
            
            # プレースホルダのインデックスを特定して上書き
            # [8]: Force, [9]: F1, [-1]: Reward
            row[8] = f_z
            row[9] = f1_val     # F1は全行に同じ値を入れる（エピソード評価のため）
            row[-1] = reward_val # 報酬も全行に同じ値を入れる
            
            rows_to_write.append(row)

        # まとめて書き込み
        if rows_to_write:
            self.logger.add_data_batch(rows_to_write)
        
        # 次のRLステップのためにバッファをクリア
        self.step_data_buffer.clear()

    def save_on_exit(self):
        if self.logger:
            self.logger.save()
            print(f"[LoggingManager] Logs saved to {self.logger.filepath}")