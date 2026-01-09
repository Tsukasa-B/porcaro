# source/porcaro_rl/porcaro_rl/tasks/direct/porcaro_rlv1/rewards/reward.py
from __future__ import annotations
import torch
from typing import Sequence

from ..cfg.rewards_cfg import RewardsCfg

class RewardManager:
    """
    リズム演奏用の報酬計算クラス。
    """
    def __init__(self, cfg: RewardsCfg, num_envs: int, device: str | torch.device):
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = torch.device(device)
        
        # --- 設定値の読み込み（新旧互換性対応）---
        
        # ヒット判定の閾値
        # 新しい設定(force_threshold_rest)があれば使い、なければデフォルト1.0
        self.hit_threshold = float(getattr(cfg, "force_threshold_rest", getattr(cfg, "hit_threshold_force", 1.0)))
        
        # 目標打撃力
        self.target_force = float(cfg.target_force_fd)
        
        # BPM設定
        # 新しい設定はbpm_min/maxだが、RewardManagerの内部計算用に固定値(120)または代表値を設定
        # (実際の学習には RhythmGenerator が使われるため、ここでの値はログ用/参考用)
        if hasattr(cfg, "target_bpm"):
            self.target_bpm = float(cfg.target_bpm)
        elif hasattr(cfg, "bpm_max"):
            self.target_bpm = float(cfg.bpm_max) # 仮で最大値を入れておく
        else:
            self.target_bpm = 120.0
            
        self.beat_interval = 60.0 / self.target_bpm
        
        # ガウス関数のパラメータ (名前変更に対応: sigma_f -> sigma_force, sigma_t -> sigma_time)
        self.sigma_f = float(getattr(cfg, "sigma_force", getattr(cfg, "sigma_f", 10.0)))
        self.sigma_t = float(getattr(cfg, "sigma_time", getattr(cfg, "sigma_t", 0.05)))
        
        # 重み (古い重みパラメータは学習に使われないため、エラー回避用に0.0を設定)
        self.w_force = float(getattr(cfg, "weight_force", 0.0))
        self.w_timing = float(getattr(cfg, "weight_timing", 0.0))
        
        # ペナルティ用パラメータ
        self.w_penalty = float(getattr(cfg, "weight_contact_penalty", 0.0))
        self.safe_window = float(getattr(cfg, "penalty_safe_window", 0.1))
        self.w_double_hit = float(getattr(cfg, "weight_double_hit_penalty", 0.0))

        # --- 内部状態の初期化 ---
        self.hit_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        # アーミング（タメ）判定用のパラメータ
        self.arming_duration_threshold = 0.05
        
        # 内部状態バッファ
        self.air_time_counter = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.is_armed = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        
        # ステートマシン (0=IDLE, 1=RISING, 2=FALLING)
        self.hit_state = torch.zeros(self.num_envs, device=self.device, dtype=torch.int8)
        self.current_peak_force = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.current_peak_time = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)

        # 終了判定バッファ
        self.terminated_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.truncated_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        
        # 最新のヒットのピーク力を保持するバッファ
        self.last_hit_peak_force = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        # 最後に報酬を与えたビートのインデックス (-1で初期化)
        self.last_rewarded_beat = torch.full((self.num_envs,), -1, device=self.device, dtype=torch.long)

    def reset_idx(self, env_ids: torch.Tensor | Sequence[int]):
        """リセット処理"""
        self.hit_count[env_ids] = 0
        self.hit_state[env_ids] = 0
        self.current_peak_force[env_ids] = 0.0
        self.current_peak_time[env_ids] = 0.0
        self.terminated_buf[env_ids] = False
        self.truncated_buf[env_ids] = False
        self.last_hit_peak_force[env_ids] = 0.0
        self.air_time_counter[env_ids] = 0.0
        self.is_armed[env_ids] = True 
        self.last_rewarded_beat[env_ids] = -1

    def _process_hit_event(self, env_ids: torch.Tensor, peak_forces: torch.Tensor, peak_times: torch.Tensor, rewards: torch.Tensor):
        """
        打撃確定時の処理 (ログ記録用)
        ※ 学習用報酬はEnv側で計算しているため、ここでは内部状態の更新のみを行う
        """
        if len(env_ids) == 0:
            return

        # 1. ビート計算 (ログ用)
        current_beat_indices = torch.round(peak_times / self.beat_interval).long()
        is_new_beat_mask = (current_beat_indices != self.last_rewarded_beat[env_ids])
        
        # 2. ヒットカウント更新
        self.hit_count[env_ids] += 1 
        
        # 3. ビート履歴更新
        valid_ids = env_ids[is_new_beat_mask]
        if len(valid_ids) > 0:
            self.last_rewarded_beat[valid_ids] = current_beat_indices[is_new_beat_mask]

        # 4. ピーク力記録
        self.last_hit_peak_force[env_ids] = peak_forces

    def compute_reward_and_dones(self, 
                                 force_z_history: torch.Tensor,
                                 current_time_s: torch.Tensor,
                                 dt_step: float,
                                 episode_length_buf: torch.Tensor,
                                 max_episode_length: int
                                 ) -> torch.Tensor:
        """
        物理ステップごとの履歴を走査してヒット検知 & ログ更新
        """
        rewards = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        decimation = force_z_history.shape[1]

        for i in range(decimation - 1, -1, -1):
            f_val = force_z_history[:, i]
            t_val = current_time_s - (i + 1) * dt_step
            
            is_touching = (f_val >= self.hit_threshold)

            # アーミングロジック
            self.air_time_counter = torch.where(
                ~is_touching, 
                self.air_time_counter + dt_step, 
                torch.zeros_like(self.air_time_counter)
            )
            just_armed = (self.air_time_counter >= self.arming_duration_threshold)
            self.is_armed = self.is_armed | just_armed

            # ステートマシン更新
            # (A) IDLE -> RISING
            rising_edge = (self.hit_state == 0) & is_touching
            if rising_edge.any():
                ids = torch.where(rising_edge)[0]
                valid_start_ids = ids[self.is_armed[ids]]
                if len(valid_start_ids) > 0:
                    self.hit_state[valid_start_ids] = 1 # RISING
                    self.current_peak_force[valid_start_ids] = f_val[valid_start_ids]
                    self.current_peak_time[valid_start_ids] = t_val[valid_start_ids]
                    self.is_armed[valid_start_ids] = False
                    self.air_time_counter[valid_start_ids] = 0.0

            # (B) RISING -> Update/FALLING
            rising_mask = (self.hit_state == 1) & is_touching
            if rising_mask.any():
                ids = torch.where(rising_mask)[0]
                curr_f = f_val[ids]
                peak_f = self.current_peak_force[ids]
                
                up_mask = (curr_f >= peak_f)
                up_ids = ids[up_mask]
                self.current_peak_force[up_ids] = curr_f[up_mask]
                self.current_peak_time[up_ids] = t_val[up_ids]
                
                down_mask = (curr_f < peak_f)
                down_ids = ids[down_mask]
                self.hit_state[down_ids] = 2

            # (C) END EVENT
            falling_edge = ((self.hit_state == 1) | (self.hit_state == 2)) & (~is_touching)
            if falling_edge.any():
                ids = torch.where(falling_edge)[0]
                self.hit_state[ids] = 0 # IDLE
                
                # イベント確定処理
                self._process_hit_event(
                    ids, 
                    self.current_peak_force[ids], 
                    self.current_peak_time[ids], 
                    rewards
                )
                self.current_peak_force[ids] = 0.0
                self.current_peak_time[ids] = 0.0

        # 終了判定
        self.truncated_buf = (episode_length_buf >= max_episode_length)
        self.terminated_buf[:] = False 

        return rewards

    def get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.terminated_buf, self.truncated_buf
    
    def get_first_hit_force(self) -> torch.Tensor:
        """ログ用に、最新の打撃のピーク力を返す"""
        return self.last_hit_peak_force