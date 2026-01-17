# source/porcaro_rl/porcaro_rl/tasks/direct/porcaro_rlv1/rewards/reward.py
from __future__ import annotations
import torch
from ..cfg.rewards_cfg import RewardsCfg

class RewardManager:
    def __init__(self, cfg: RewardsCfg, num_envs: int, device: str | torch.device):
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = torch.device(device)
        
        # 閾値や制限値のプリロード
        self.hit_threshold = getattr(cfg, "hit_threshold_force", 1.0)
        self.max_contact_duration = getattr(cfg, "max_contact_duration_s", 0.1)
        self.impact_window = getattr(cfg, "impact_window_s", 0.05)
        
        # ターゲット参照値（分母用）
        self.target_ref_val = float(cfg.target_force_fd)

        # 状態管理用バッファ
        self.hit_state = torch.zeros(num_envs, dtype=torch.long, device=self.device) 
        self.current_peak_force = torch.zeros(num_envs, device=self.device)
        self.contact_duration = torch.zeros(num_envs, device=self.device)
        
        # ★追加: ピーク発生時のタイミング適合度 (0.0 ~ 1.0)
        self.peak_timing_scale = torch.zeros(num_envs, device=self.device)

        # ログ用
        self.episode_sums = {
            "match": torch.zeros(num_envs, device=self.device),
            "rest_reward": torch.zeros(num_envs, device=self.device),
            "rest_penalty": torch.zeros(num_envs, device=self.device),
            "contact_limit": torch.zeros(num_envs, device=self.device),
            "joint_limit": torch.zeros(num_envs, device=self.device)
        }

    def reset_idx(self, env_ids: torch.Tensor):
        self.hit_state[env_ids] = 0
        self.current_peak_force[env_ids] = 0.0
        self.contact_duration[env_ids] = 0.0
        self.peak_timing_scale[env_ids] = 0.0 # リセット
        
        for key in self.episode_sums:
            self.episode_sums[key][env_ids] = 0.0

    def compute_rewards(self, 
                        actions: torch.Tensor, 
                        joint_pos: torch.Tensor, 
                        force_z: torch.Tensor, 
                        target_force_trace: torch.Tensor,
                        target_force_ref: torch.Tensor,
                        dt: float) -> dict[str, torch.Tensor]:
        
        terms = {}

        # 共通項目の計算
        is_touching = (force_z > self.hit_threshold)
        is_rest_period = (target_force_trace < 1.0) 

        self.contact_duration = torch.where(
            is_touching,
            self.contact_duration + dt,
            torch.zeros_like(self.contact_duration)
        )

        # ----------------------------------------------------
        # 1. 打撃 (Match) & タイミング評価
        # ----------------------------------------------------
        
        # A. Rising Edge
        rising = (self.hit_state == 0) & is_touching
        if rising.any():
            ids = torch.where(rising)[0]
            self.hit_state[ids] = 1 
            self.current_peak_force[ids] = force_z[ids]
            # 初期値として現在のタイミング評価を保存
            current_scale = (target_force_trace[ids] / self.target_ref_val).clamp(0.0, 1.0)
            self.peak_timing_scale[ids] = current_scale
        
        # B. Sustain (接触継続)
        sustain = (self.hit_state == 1) & is_touching
        if sustain.any():
            ids = torch.where(sustain)[0]
            
            in_window = (self.contact_duration[ids] <= self.impact_window)
            if in_window.any():
                upd_ids = ids[in_window]
                
                curr_force = force_z[upd_ids]
                prev_peak = self.current_peak_force[upd_ids]
                
                # ★重要変更: 力が更新されたら、その瞬間の「タイミング評価」も更新する
                # これにより「最大の力が出た瞬間」のターゲット値が採用される
                is_new_peak = (curr_force > prev_peak)
                if is_new_peak.any():
                    peak_ids = upd_ids[is_new_peak]
                    self.current_peak_force[peak_ids] = curr_force[is_new_peak]
                    
                    # その瞬間のターゲット波形の高さを取得 (Timing Score)
                    t_scale = (target_force_trace[peak_ids] / self.target_ref_val).clamp(0.0, 1.0)
                    self.peak_timing_scale[peak_ids] = t_scale
            
            # 押し付け判定
            is_pushing = (self.contact_duration[ids] > self.max_contact_duration)
            if is_pushing.any():
                push_ids = ids[is_pushing]
                self.hit_state[push_ids] = 2 
                self.current_peak_force[push_ids] = 0.0
                self.peak_timing_scale[push_ids] = 0.0

        # C. Falling Edge (報酬確定)
        falling = (self.hit_state != 0) & (~is_touching)
        hit_reward = torch.zeros(self.num_envs, device=self.device)
        
        if falling.any():
            ids = torch.where(falling)[0]
            valid_hits = (self.hit_state[ids] == 1)
            
            if valid_hits.any():
                valid_ids = ids[valid_hits]
                
                # 1. 力の一致度スコア (0.0 ~ 1.0)
                force_score = self._evaluate_hit(
                    peak_force=self.current_peak_force[valid_ids],
                    target_val=target_force_ref[valid_ids] 
                )
                
                # 2. タイミング係数 (0.0 ~ 1.0)
                # ピーク発生時にターゲットが盛り上がっていたか？
                timing_factor = self.peak_timing_scale[valid_ids]
                
                # 最終報酬 = 力スコア × タイミング係数
                hit_reward[valid_ids] = force_score * timing_factor

            self.hit_state[ids] = 0
            self.current_peak_force[ids] = 0.0
            self.peak_timing_scale[ids] = 0.0

        terms["match"] = hit_reward

        # ----------------------------------------------------
        # 他の項 (変更なし)
        # ----------------------------------------------------
        compliance = is_rest_period & (~is_touching)
        terms["rest_reward"] = torch.where(
            compliance,
            torch.ones(self.num_envs, device=self.device),
            torch.zeros(self.num_envs, device=self.device)
        )

        violation = is_rest_period & is_touching
        terms["rest_penalty"] = torch.where(
            violation,
            (force_z / 10.0), 
            torch.zeros(self.num_envs, device=self.device)
        )

        over_time = (self.contact_duration - self.max_contact_duration).clamp(min=0.0)
        terms["contact_limit"] = over_time

        if self.cfg.weight_joint_limits != 0.0:
            wrist_pos = joint_pos[:, 0]
            wrist_min = torch.tensor(self.cfg.limit_wrist_range[0], device=self.device).deg2rad()
            wrist_max = torch.tensor(self.cfg.limit_wrist_range[1], device=self.device).deg2rad()
            v_min = (wrist_min - wrist_pos).clamp(min=0.0)
            v_max = (wrist_pos - wrist_max).clamp(min=0.0)
            terms["joint_limit"] = (v_min + v_max)
        else:
            terms["joint_limit"] = torch.zeros(self.num_envs, device=self.device)

        total_reward = (
            terms["match"] * self.cfg.weight_match +
            terms["rest_reward"] * self.cfg.weight_rest +
            terms["rest_penalty"] * self.cfg.weight_rest_penalty +
            terms["contact_limit"] * self.cfg.weight_contact_continuous +
            terms["joint_limit"] * self.cfg.weight_joint_limits
        )

        return total_reward, terms

    def _evaluate_hit(self, peak_force, target_val):
        force_error = torch.abs(peak_force - target_val)
        score = torch.exp(-force_error / self.cfg.sigma_force)
        
        if self.cfg.scale_reward_by_force_magnitude:
            magnitude_factor = (target_val / self.target_ref_val).clamp(max=1.5)
            return score * magnitude_factor
        else:
            return score