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
        
        # 設定値の読み込み
        self.hit_threshold = float(getattr(cfg, "force_threshold_rest", getattr(cfg, "hit_threshold_force", 1.0)))
        self.target_force = float(cfg.target_force_fd)
        
        self.weight_match = cfg.weight_match
        self.weight_rest_penalty = cfg.weight_rest_penalty
        self.weight_efficiency = cfg.weight_efficiency
        
        self.weight_joint_limits = getattr(cfg, "weight_joint_limits", 0.0)
        # ★追加: 常時接触ペナルティ設定
        self.weight_contact_continuous = getattr(cfg, "weight_contact_continuous", 0.0)
        self.max_contact_duration_s = getattr(cfg, "max_contact_duration_s", 0.1)

        wrist_range_deg = getattr(cfg, "limit_wrist_range", (-80.0, 30.0))
        self.wrist_limit_min = torch.tensor(wrist_range_deg[0], device=self.device) * (3.14159 / 180.0)
        self.wrist_limit_max = torch.tensor(wrist_range_deg[1], device=self.device) * (3.14159 / 180.0)
        
        if hasattr(cfg, "target_bpm"):
            self.target_bpm = float(cfg.target_bpm)
        elif hasattr(cfg, "bpm_max"):
            self.target_bpm = float(cfg.bpm_max)
        else:
            self.target_bpm = 120.0

        # 内部ステート
        self.hit_state = torch.zeros(num_envs, dtype=torch.long, device=self.device) 
        self.current_peak_force = torch.zeros(num_envs, device=self.device)
        self.current_peak_time = torch.zeros(num_envs, device=self.device)
        # ★追加: 接触継続時間の計測用
        self.contact_duration = torch.zeros(num_envs, device=self.device)
        
        self.terminated_buf = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.truncated_buf = torch.zeros(num_envs, dtype=torch.bool, device=self.device)

        self.last_match_reward = torch.zeros(num_envs, device=self.device)
        self.last_rest_penalty = torch.zeros(num_envs, device=self.device)
        self.last_limit_penalty = torch.zeros(num_envs, device=self.device)

    def reset_idx(self, env_ids: torch.Tensor):
        self.hit_state[env_ids] = 0
        self.current_peak_force[env_ids] = 0.0
        self.current_peak_time[env_ids] = 0.0
        # ★追加: 接触時間をリセット
        self.contact_duration[env_ids] = 0.0
        self.terminated_buf[env_ids] = False
        self.truncated_buf[env_ids] = False
        
        self.last_match_reward[env_ids] = 0.0
        self.last_rest_penalty[env_ids] = 0.0
        self.last_limit_penalty[env_ids] = 0.0

    def compute_rewards(self, 
                        obs: torch.Tensor, 
                        actions: torch.Tensor, 
                        joint_pos: torch.Tensor, 
                        force_z: torch.Tensor, 
                        target_force_trace: torch.Tensor,
                        episode_length_buf: torch.Tensor,
                        max_episode_length: int,
                        current_time_s: torch.Tensor,
                        dt: float) -> torch.Tensor: # ★修正: dtを引数に追加(env.py側での渡しを確認)
        
        rewards = torch.zeros(self.num_envs, device=self.device)
        
        # --- 接触状態の判定 ---
        is_touching = (force_z > self.hit_threshold)

        # ★追加: 接触継続時間の更新
        self.contact_duration[is_touching] += dt
        self.contact_duration[~is_touching] = 0.0

        # 1. 休符ペナルティ
        is_rest_period = (target_force_trace < 1.0)
        rest_violation = is_rest_period & is_touching
        
        rest_penalty = torch.zeros_like(rewards)
        rest_penalty[rest_violation] = -1.0 * (force_z[rest_violation] / 10.0) * self.weight_rest_penalty
        rewards += rest_penalty
        self.last_rest_penalty = rest_penalty

        # 2. 手首角度制限ペナルティ
        wrist_pos = joint_pos[:, 0]
        violation_min = torch.clamp(self.wrist_limit_min - wrist_pos, min=0.0)
        violation_max = torch.clamp(wrist_pos - self.wrist_limit_max, min=0.0)
        total_violation = violation_min + violation_max
        limit_penalty = total_violation * self.weight_joint_limits
        rewards += limit_penalty
        self.last_limit_penalty = limit_penalty

        # ★追加: 3. 連続接触(押し付け)ペナルティ
        # max_contact_duration_s を超えて触り続けている場合、超過時間に比例して罰を与える
        continuous_penalty = torch.where(
            self.contact_duration > self.max_contact_duration_s,
            self.weight_contact_continuous * (self.contact_duration - self.max_contact_duration_s),
            torch.zeros_like(rewards)
        )
        rewards += continuous_penalty

        # 4. エネルギー効率
        rewards -= torch.sum(actions ** 2, dim=1) * self.weight_efficiency

        # 5. ヒットイベント (IDLE -> RISING -> FALLING -> IDLE)
        # IDLE -> RISING
        rising_edge = (self.hit_state == 0) & is_touching
        if rising_edge.any():
            ids = torch.where(rising_edge)[0]
            self.hit_state[ids] = 1 
            self.current_peak_force[ids] = force_z[ids]
            self.current_peak_time[ids] = current_time_s[ids]

        # UPDATE PEAK
        if (self.hit_state == 1).any():
            rising_mask = (self.hit_state == 1)
            ids = torch.where(rising_mask)[0]
            curr_f = force_z[ids]
            peak_f = self.current_peak_force[ids]
            
            up_mask = (curr_f >= peak_f)
            up_ids = ids[up_mask]
            self.current_peak_force[up_ids] = curr_f[up_mask]
            self.current_peak_time[up_ids] = current_time_s[up_ids]

        # END EVENT
        falling_edge = (self.hit_state > 0) & (~is_touching)
        if falling_edge.any():
            ids = torch.where(falling_edge)[0]
            self.hit_state[ids] = 0 # IDLE
            
            self._process_hit_event(
                ids, 
                self.current_peak_force[ids], 
                self.current_peak_time[ids], 
                rewards,
                target_force_trace 
            )
            self.current_peak_force[ids] = 0.0
            self.current_peak_time[ids] = 0.0

        self.truncated_buf = (episode_length_buf >= max_episode_length)
        self.terminated_buf[:] = False 

        return rewards

    def _process_hit_event(self, ids, peak_forces, peak_times, rewards_tensor, target_trace):
        target_vals = target_trace[ids]
        force_err = torch.abs(peak_forces - self.target_force) 
        
        score = torch.exp(-force_err / self.cfg.sigma_force)
        target_factor = (target_vals / self.target_force).clamp(min=0.0, max=1.0)
        
        match_reward = score * target_factor * self.weight_match
        
        rewards_tensor[ids] += match_reward
        self.last_match_reward[ids] = match_reward

    def get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.terminated_buf, self.truncated_buf