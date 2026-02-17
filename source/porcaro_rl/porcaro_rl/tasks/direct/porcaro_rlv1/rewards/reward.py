# source/porcaro_rl/porcaro_rl/tasks/direct/porcaro_rlv1/rewards/reward.py
from __future__ import annotations
import torch
from ..cfg.rewards_cfg import RewardsCfg

class RewardManager:
    def __init__(self, cfg: RewardsCfg, num_envs: int, device: str | torch.device):
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = torch.device(device)
        
        # --- 基本設定 ---
        self.hit_threshold = getattr(cfg, "hit_threshold_force", 1.0)
        self.target_ref_val = float(cfg.target_force_fd)
        
        # 基準BPM
        self.default_bpm = 120.0
        self.rest_threshold = 1.0 

        # --- 状態管理用バッファ ---
        self.hit_state = torch.zeros(num_envs, dtype=torch.long, device=self.device) 
        self.current_peak_force = torch.zeros(num_envs, device=self.device)
        self.contact_duration = torch.zeros(num_envs, device=self.device)
        
        # 【重要】接触期間中の「ターゲット波形の最大重なり度」を記憶するバッファ
        self.peak_timing_scale = torch.zeros(num_envs, device=self.device)

        # 最後にヒット判定を行い、報酬を与えた「エピソード内経過時間(秒)」を記録
        self.last_reward_time = torch.full((num_envs,), -1.0, device=self.device)
        
        self.prev_is_rest = torch.ones(num_envs, dtype=torch.bool, device=self.device)
        self.current_time_s = torch.zeros(num_envs, device=self.device)

        # --- ログ・集計用バッファ ---
        self.episode_sums = {
            "match": torch.zeros(num_envs, device=self.device),
            "rest_reward": torch.zeros(num_envs, device=self.device),
            "rest_penalty": torch.zeros(num_envs, device=self.device),
            "miss_penalty": torch.zeros(num_envs, device=self.device),
            "double_hit_penalty": torch.zeros(num_envs, device=self.device),
            "contact_limit": torch.zeros(num_envs, device=self.device),
            "joint_limit": torch.zeros(num_envs, device=self.device)
        }

    def reset_idx(self, env_ids: torch.Tensor):
        self.hit_state[env_ids] = 0
        self.current_peak_force[env_ids] = 0.0
        self.contact_duration[env_ids] = 0.0
        self.peak_timing_scale[env_ids] = 0.0
        
        self.last_reward_time[env_ids] = -1.0
        self.prev_is_rest[env_ids] = True
        self.current_time_s[env_ids] = 0.0
        
        for key in self.episode_sums:
            self.episode_sums[key][env_ids] = 0.0

    def compute_rewards(self, 
                        actions: torch.Tensor, 
                        joint_pos: torch.Tensor, 
                        force_z: torch.Tensor, 
                        target_force_trace: torch.Tensor,
                        target_force_ref: torch.Tensor,
                        dt: float,
                        current_bpm: torch.Tensor = None) -> dict[str, torch.Tensor]:
        
        self.current_time_s += dt

        # --- BPMスケーリングの準備 ---
        if current_bpm is None:
             safe_bpm = torch.full((self.num_envs,), self.default_bpm, device=self.device)
        else:
             safe_bpm = current_bpm.clamp(min=10.0)

        t_16th = 15.0 / safe_bpm
        dyn_max_contact   = t_16th * 0.50
        dyn_cooltime      = t_16th * 0.50
        dyn_impact_window = t_16th * 0.40
        dyn_miss_thresh   = t_16th * 2.00

        # スケーリング係数 (前回の修正を維持)
        match_scale_factor = torch.ones_like(safe_bpm)
        time_scale_factor = (safe_bpm / 120.0)

        terms = {}
        is_touching = (force_z > self.hit_threshold)
        is_rest_period = (target_force_trace < self.rest_threshold)

        # ----------------------------------------------------
        # 1. Miss Penalty
        # ----------------------------------------------------
        note_offset = (~self.prev_is_rest) & is_rest_period
        miss_penalty_term = torch.zeros(self.num_envs, device=self.device)
        if note_offset.any():
            ids = torch.where(note_offset)[0]
            time_diff = self.current_time_s[ids] - self.last_reward_time[ids]
            missed = (time_diff > dyn_miss_thresh[ids]) | (self.last_reward_time[ids] < 0.0)
            real_miss = missed & (self.hit_state[ids] == 0)
            if real_miss.any():
                miss_penalty_term[ids[real_miss]] = 1.0

        terms["miss_penalty"] = miss_penalty_term * match_scale_factor

        self.contact_duration = torch.where(
            is_touching,
            self.contact_duration + dt,
            torch.zeros_like(self.contact_duration)
        )

        # ----------------------------------------------------
        # 2. Match Reward (打撃判定ロジックの改善)
        # ----------------------------------------------------
        match_reward = torch.zeros(self.num_envs, device=self.device)
        double_hit_penalty_term = torch.zeros(self.num_envs, device=self.device)

        # A. Rising Edge (接触開始)
        rising = (self.hit_state == 0) & is_touching
        if rising.any():
            ids = torch.where(rising)[0]
            self.hit_state[ids] = 1 
            self.current_peak_force[ids] = force_z[ids]
            # 初期タイミングを記録
            self.peak_timing_scale[ids] = (target_force_trace[ids] / self.target_ref_val).clamp(0.0, 1.0)
        
        # B. Sustain (接触中: 最大値を追いかけ続ける)
        sustain = (self.hit_state == 1) & is_touching
        if sustain.any():
            ids = torch.where(sustain)[0]
            # 物理的な立ち上がり窓 (dyn_impact_window) 以内であれば更新
            in_window = (self.contact_duration[ids] <= dyn_impact_window[ids])
            if in_window.any():
                upd_ids = ids[in_window]
                # 【改善1】現在の重なり度を取得し、期間中の最大値を保持
                # これにより、PAMの遅れで波形がズレても「最高の結果」を評価できる
                current_scale = (target_force_trace[upd_ids] / self.target_ref_val).clamp(0.0, 1.0)
                self.peak_timing_scale[upd_ids] = torch.max(self.peak_timing_scale[upd_ids], current_scale)
                
                curr_force = force_z[upd_ids]
                is_new_peak = (curr_force > self.current_peak_force[upd_ids])
                if is_new_peak.any():
                    self.current_peak_force[upd_ids[is_new_peak]] = curr_force[is_new_peak]

        # C. Falling Edge (離脱: 報酬確定)
        falling = (self.hit_state != 0) & (~is_touching)
        if falling.any():
            ids = torch.where(falling)[0]
            valid_hits = (self.hit_state[ids] == 1) 
            if valid_hits.any():
                valid_ids = ids[valid_hits]
                
                # 【改善2】判定閾値を 0.1 -> 0.01 へ。
                # 接触期間中に一度でも波形の裾野(1%以上)に触れていれば、Hitとみなす。
                hit_in_note = (self.peak_timing_scale[valid_ids] > 0.01)
                
                time_since_last = self.current_time_s[valid_ids] - self.last_reward_time[valid_ids]
                thresholds = dyn_cooltime[valid_ids]
                is_cooled_down = (time_since_last > thresholds)

                rewardable_hits = hit_in_note & is_cooled_down
                
                success_ids = valid_ids[rewardable_hits]
                if len(success_ids) > 0:
                    force_score = self._evaluate_hit(
                        self.current_peak_force[success_ids], 
                        target_force_ref[success_ids]
                    )
                    
                    # 【改善3】ベース報酬（ボーナス点）の導入
                    # 「とにかくタイミングよく叩いた」ことに対して50点を保証し、残り50点で力の精度を測る
                    # 変更前: base_reward = force_score * timing_ok
                    base_reward = 0.5 + (0.5 * force_score)
                    
                    match_reward[success_ids] = base_reward * match_scale_factor[success_ids]
                    self.last_reward_time[success_ids] = self.current_time_s[success_ids]

                # ペナルティ判定
                too_fast_ids = valid_ids[hit_in_note & (~is_cooled_down)]
                if len(too_fast_ids) > 0:
                    double_hit_penalty_term[too_fast_ids] = 1.0 * match_scale_factor[too_fast_ids]
                rest_hit_ids = valid_ids[~hit_in_note]
                if len(rest_hit_ids) > 0:
                    double_hit_penalty_term[rest_hit_ids] = 1.0 * match_scale_factor[rest_hit_ids]

            # 状態リセット
            self.hit_state[ids] = 0
            self.current_peak_force[ids] = 0.0
            self.peak_timing_scale[ids] = 0.0

        terms["match"] = match_reward
        terms["double_hit_penalty"] = double_hit_penalty_term
        
        # ----------------------------------------------------
        # 3. Continuous Rewards
        # ----------------------------------------------------
        compliance = is_rest_period & (~is_touching)
        rest_base = torch.where(compliance, torch.ones(self.num_envs, device=self.device), torch.zeros(self.num_envs, device=self.device))
        terms["rest_reward"] = rest_base * time_scale_factor

        violation = is_rest_period & is_touching
        rest_pen_base = torch.where(violation, (force_z / 10.0), torch.zeros(self.num_envs, device=self.device))
        terms["rest_penalty"] = rest_pen_base * time_scale_factor

        over_time = (self.contact_duration - dyn_max_contact).clamp(min=0.0)
        terms["contact_limit"] = over_time * time_scale_factor
        
        terms["joint_limit"] = torch.zeros(self.num_envs, device=self.device)

        w_miss = getattr(self.cfg, "weight_miss", -10.0)
        w_double = getattr(self.cfg, "weight_double_hit", -5.0)

        total_reward = (
            terms["match"] * self.cfg.weight_match +
            terms["rest_reward"] * self.cfg.weight_rest +
            terms["rest_penalty"] * self.cfg.weight_rest_penalty +
            terms["contact_limit"] * self.cfg.weight_contact_continuous +
            terms["joint_limit"] * self.cfg.weight_joint_limits +
            terms["miss_penalty"] * w_miss +
            terms["double_hit_penalty"] * w_double
        )
        
        self.prev_is_rest = is_rest_period.clone()
        return total_reward, terms

    def _evaluate_hit(self, peak_force, target_val):
        force_error = torch.abs(peak_force - target_val)
        score = torch.exp(-force_error / self.cfg.sigma_force)
        
        if self.cfg.scale_reward_by_force_magnitude:
            magnitude_factor = (target_val / self.target_ref_val).clamp(max=1.5)
            return score * magnitude_factor
        else:
            return score