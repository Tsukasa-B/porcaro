# source/porcaro_rl/porcaro_rl/tasks/direct/porcaro_rlv1/rewards/reward.py

from __future__ import annotations
import torch
from typing import Sequence
from ..cfg.rewards_cfg import RewardsCfg

class RewardManager:
    def __init__(self, cfg: RewardsCfg, num_envs: int, device: str | torch.device, dt: float):
        # ★引数に dt を追加しました (時刻計算用)
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.dt = dt  # 物理ステップ時間 (sim.dt)
        
        # 設定値のロード
        self.hit_threshold = float(cfg.hit_threshold_force)
        self.target_force = float(cfg.target_force_fd)
        self.sigma_f = float(cfg.sigma_f)
        self.w1 = float(cfg.weight_w1_force)
        self.w2 = float(cfg.weight_w2_hit_count)
        
        # --- リズム用 ---
        self.bpm = float(cfg.target_bpm)
        self.beat_interval = 60.0 / self.bpm if self.bpm > 0 else 1.0
        self.sigma_t = float(cfg.sigma_t)
        self.w3 = float(cfg.weight_w3_timing)

        # バッファ類 (既存と同じ)
        self.hit_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.first_hit_force = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.hit_state = torch.zeros(self.num_envs, device=self.device, dtype=torch.int8)
        self.current_peak_force = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.terminated_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.truncated_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

    def reset_idx(self, env_ids: torch.Tensor | Sequence[int]):
        self.hit_count[env_ids] = 0
        self.first_hit_force[env_ids] = 0.0
        self.hit_state[env_ids] = 0
        self.current_peak_force[env_ids] = 0.0
        self.terminated_buf[env_ids] = False
        self.truncated_buf[env_ids] = False

    def _process_hit_event(self, env_ids: torch.Tensor, peak_force: torch.Tensor, 
                           hit_time: torch.Tensor, # <--- 時刻を受け取るように変更
                           rewards: torch.Tensor):
        if len(env_ids) == 0: return

        self.hit_count[env_ids] += 1
        is_first_hit_mask = (self.hit_count[env_ids] == 1)
        first_hit_ids = env_ids[is_first_hit_mask]
        
        if len(first_hit_ids) > 0:
            # --- 1. 力の正確性 (r_F1) ---
            f1 = peak_force[is_first_hit_mask]
            self.first_hit_force[first_hit_ids] = f1
            r_f1 = torch.exp(-0.5 * torch.square((f1 - self.target_force) / self.sigma_f))
            
            # --- 2. タイミングの正確性 (r_time) ---
            # そのヒット時刻 t に最も近い「正解の拍」を探す
            # nearest_beat = round(t / interval) * interval
            t_hit = hit_time[is_first_hit_mask]
            nearest_beat_time = torch.round(t_hit / self.beat_interval) * self.beat_interval
            time_diff = torch.abs(t_hit - nearest_beat_time)
            
            r_time = torch.exp(-0.5 * torch.square(time_diff / self.sigma_t))

            # 報酬の合算
            total_r = (self.w1 * r_f1) + (self.w3 * r_time)
            rewards[first_hit_ids] += total_r

    def compute_reward_and_dones(self, 
                                 force_z_history: torch.Tensor,
                                 episode_length_buf: torch.Tensor,
                                 max_episode_length: int,
                                 current_time_s: torch.Tensor # <--- 現在時刻を受け取る
                                 ) -> torch.Tensor:
        
        rewards = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        decimation = force_z_history.shape[1]

        # 履歴を古い順に処理 (インデックス: decimation-1 -> 0)
        # 履歴[0]が最新(現在時刻)。履歴[i]の時刻は (現在 - i*dt)
        for i in range(decimation - 1, -1, -1):
            
            # このサンプルの時刻を計算
            # current_time_s は [num_envs]
            step_time = current_time_s - (i * self.dt)
            
            current_force = force_z_history[:, i].reshape(-1)
            is_above_threshold = (current_force >= self.hit_threshold)
            
            # ステートマシン (IDLE, RISING, FALLING)
            state_idle = (self.hit_state == 0)
            state_rising = (self.hit_state == 1)
            state_falling = (self.hit_state == 2)
            
            # (A) IDLE -> RISING
            rising_edge_mask = state_idle & is_above_threshold
            rising_edge_ids = torch.where(rising_edge_mask)[0]
            if len(rising_edge_ids) > 0:
                self.hit_state[rising_edge_ids] = 1 
                self.current_peak_force[rising_edge_ids] = current_force[rising_edge_ids]

            # (B) RISING -> (Update Peak or FALLING)
            rising_state_mask = state_rising & is_above_threshold
            rising_state_ids = torch.where(rising_state_mask)[0]
            if len(rising_state_ids) > 0:
                f_sub = current_force[rising_state_ids]
                p_sub = self.current_peak_force[rising_state_ids]
                
                up_mask = (f_sub >= p_sub)
                self.current_peak_force[rising_state_ids[up_mask]] = f_sub[up_mask]
                
                down_mask = (f_sub < p_sub)
                self.hit_state[rising_state_ids[down_mask]] = 2 # FALLING

            # (C) RISING -> IDLE (Short pulse)
            rising_end_mask = state_rising & (~is_above_threshold)
            ids = torch.where(rising_end_mask)[0]
            if len(ids) > 0:
                self.hit_state[ids] = 0
                # ヒット確定処理: 時刻(step_time)を渡す
                self._process_hit_event(ids, self.current_peak_force[ids], step_time[ids], rewards)
                self.current_peak_force[ids] = 0.0

            # (D) FALLING -> RISING (Double hit)
            falling_up_mask = state_falling & is_above_threshold & (current_force >= self.current_peak_force)
            ids = torch.where(falling_up_mask)[0]
            if len(ids) > 0:
                self.hit_state[ids] = 1 # RISING again
                # 前の山の確定処理
                self._process_hit_event(ids, self.current_peak_force[ids], step_time[ids], rewards)
                self.current_peak_force[ids] = current_force[ids] # 新しい山の開始

            # (E) FALLING -> IDLE (End of hit)
            falling_end_mask = state_falling & (~is_above_threshold)
            ids = torch.where(falling_end_mask)[0]
            if len(ids) > 0:
                self.hit_state[ids] = 0
                self._process_hit_event(ids, self.current_peak_force[ids], step_time[ids], rewards)
                self.current_peak_force[ids] = 0.0

        # 終了判定
        #self.terminated_buf = (self.hit_count >= 2)
        # 転倒や関節制限違反がない限り、エピソード時間(8秒)いっぱいまで叩かせます。
        self.terminated_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.truncated_buf = (episode_length_buf >= max_episode_length)

        # タイムアウト時の生存ボーナス的な処理
        truncated_ids = torch.where(self.truncated_buf)[0]
        if len(truncated_ids) > 0:
            success_ids = truncated_ids[self.hit_count[truncated_ids] == 1]
            if len(success_ids) > 0:
                rewards[success_ids] += self.w2

        return rewards

    def get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.terminated_buf, self.truncated_buf
    
    def get_first_hit_force(self) -> torch.Tensor:
        return self.first_hit_force