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
        self.target_force_ref = float(cfg.target_force_fd)

        # 新規パラメータのロード
        self.impact_window = getattr(cfg, "impact_window_s", 0.05)
        
        # 状態管理用バッファ
        self.hit_state = torch.zeros(num_envs, dtype=torch.long, device=self.device) 
        self.current_peak_force = torch.zeros(num_envs, device=self.device)
        self.contact_duration = torch.zeros(num_envs, device=self.device)

        # ログ用に各項の値を保持する辞書
        self.episode_sums = {
            "match": torch.zeros(num_envs, device=self.device),
            "rest_penalty": torch.zeros(num_envs, device=self.device),
            "contact_limit": torch.zeros(num_envs, device=self.device),
            "joint_limit": torch.zeros(num_envs, device=self.device)
        }

    def reset_idx(self, env_ids: torch.Tensor):
        self.hit_state[env_ids] = 0
        self.current_peak_force[env_ids] = 0.0
        self.contact_duration[env_ids] = 0.0
        
        for key in self.episode_sums:
            self.episode_sums[key][env_ids] = 0.0

    def compute_rewards(self, 
                        actions: torch.Tensor, 
                        joint_pos: torch.Tensor, 
                        force_z: torch.Tensor, 
                        target_force_trace: torch.Tensor, # これは休符判定に使う
                        target_force_ref: torch.Tensor,   # ★追加: これを打撃評価に使う
                        dt: float) -> dict[str, torch.Tensor]:
        
        terms = {}

        # 1. 接触判定
        is_touching = (force_z > self.hit_threshold)
        
        # 2. 休符判定 (波形が小さい時は休符とみなす)
        is_rest_period = (target_force_trace < 1.0) 

        # 接触時間の更新
        self.contact_duration = torch.where(
            is_touching,
            self.contact_duration + dt,
            torch.zeros_like(self.contact_duration)
        )

        # ============================================================
        # r_1: イベント駆動マッチング報酬
        # ============================================================
        hit_reward = torch.zeros(self.num_envs, device=self.device)
        
        # Rising Edge
        rising = (self.hit_state == 0) & is_touching
        if rising.any():
            ids = torch.where(rising)[0]
            self.hit_state[ids] = 1
            self.current_peak_force[ids] = force_z[ids]
        
        # Sustain (ウィンドウ内のみピーク更新)
        sustain = (self.hit_state == 1) & is_touching
        if sustain.any():
            ids = torch.where(sustain)[0]
            in_window_mask = (self.contact_duration[ids] <= self.impact_window)
            if in_window_mask.any():
                update_ids = ids[in_window_mask]
                curr = force_z[update_ids]
                peak = self.current_peak_force[update_ids]
                self.current_peak_force[update_ids] = torch.max(curr, peak)
            
        # Falling Edge (離脱 -> 評価)
        falling = (self.hit_state == 1) & (~is_touching)
        if falling.any():
            ids = torch.where(falling)[0]
            self.hit_state[ids] = 0 
            
            # ★修正: ここで target_force_ref (固定値) と比較する
            hit_reward[ids] = self._evaluate_hit(
                peak_force=self.current_peak_force[ids],
                target_val=target_force_ref[ids] 
            )
            self.current_peak_force[ids] = 0.0

        terms["match"] = hit_reward

        # ============================================================
        # r_2: 休符ペナルティ
        # ============================================================
        violation = is_rest_period & is_touching
        terms["rest_penalty"] = torch.where(
            violation,
            -1.0 * (force_z / 10.0), 
            torch.zeros(self.num_envs, device=self.device)
        )

        # ============================================================
        # r_3: 連続接触ペナルティ
        # ============================================================
        # 一定時間以上触り続けている場合、超過時間に応じてペナルティ
        over_time = (self.contact_duration - self.cfg.max_contact_duration_s).clamp(min=0.0)
        terms["contact_limit"] = -1.0 * over_time

        # ============================================================
        # r_4: 関節制限 (Joint Limits)
        # ============================================================
        # 必要な場合のみ計算
        if self.cfg.weight_joint_limits != 0.0:
            wrist_pos = joint_pos[:, 0]
            wrist_min = torch.tensor(self.cfg.limit_wrist_range[0], device=self.device).deg2rad()
            wrist_max = torch.tensor(self.cfg.limit_wrist_range[1], device=self.device).deg2rad()
            
            v_min = (wrist_min - wrist_pos).clamp(min=0.0)
            v_max = (wrist_pos - wrist_max).clamp(min=0.0)
            terms["joint_limit"] = (v_min + v_max)
        else:
            terms["joint_limit"] = torch.zeros(self.num_envs, device=self.device)

        # --- 総報酬の計算 ---
        # 重み付け和 (R = w1*r1 + w2*r2 + ...)
        total_reward = (
            terms["match"] * self.cfg.weight_match +
            terms["rest_penalty"] * self.cfg.weight_rest_penalty +
            terms["contact_limit"] * self.cfg.weight_contact_continuous +
            terms["joint_limit"] * self.cfg.weight_joint_limits
        )

        return total_reward, terms

    def _evaluate_hit(self, peak_force, target_val):
        """指定されたターゲット値との誤差に基づいてスコアを計算"""
        force_error = torch.abs(peak_force - target_val)
        score = torch.exp(-force_error / self.cfg.sigma_force)
        
        # 正規化オプション
        if self.cfg.scale_reward_by_force_magnitude:
            magnitude_factor = (target_val / self.target_force_ref).clamp(max=1.5)
            return score * magnitude_factor
        else:
            return score