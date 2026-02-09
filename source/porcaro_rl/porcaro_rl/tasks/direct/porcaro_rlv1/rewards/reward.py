# source/porcaro_rl/porcaro_rl/tasks/direct/porcaro_rlv1/rewards/reward.py
from __future__ import annotations
import torch
from ..cfg.rewards_cfg import RewardsCfg

class RewardManager:
    def __init__(self, cfg: RewardsCfg, num_envs: int, device: str | torch.device):
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = torch.device(device)
        
        # --- 閾値設定 ---
        self.hit_threshold = getattr(cfg, "hit_threshold_force", 1.0)
        self.max_contact_duration = getattr(cfg, "max_contact_duration_s", 0.1)
        self.impact_window = getattr(cfg, "impact_window_s", 0.05)
        
        # ターゲット波形がこれを下回ったら「休符」、上回ったら「音符」とみなす閾値
        self.rest_threshold = 1.0 

        # ターゲット参照値（分母用）
        self.target_ref_val = float(cfg.target_force_fd)

        # --- 状態管理用バッファ ---
        # 0: None, 1: Rising/Sustain, 2: Pushing(無効)
        self.hit_state = torch.zeros(num_envs, dtype=torch.long, device=self.device) 
        self.current_peak_force = torch.zeros(num_envs, device=self.device)
        self.contact_duration = torch.zeros(num_envs, device=self.device)
        self.peak_timing_scale = torch.zeros(num_envs, device=self.device)

        # ★追加: 現在の音符区間ですでに報酬を得たかどうかのフラグ
        self.hit_success_in_window = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        # ★追加: 前回のステップが休符だったかどうか（立ち上がり/立ち下がり検出用）
        self.prev_is_rest = torch.ones(num_envs, dtype=torch.bool, device=self.device)

        # --- ログ・集計用バッファ ---
        # 既存 + 新規ペナルティ項
        self.episode_sums = {
            "match": torch.zeros(num_envs, device=self.device),
            "rest_reward": torch.zeros(num_envs, device=self.device),
            "rest_penalty": torch.zeros(num_envs, device=self.device),
            "miss_penalty": torch.zeros(num_envs, device=self.device),       # 新規: 叩き損ね
            "double_hit_penalty": torch.zeros(num_envs, device=self.device), # 新規: 叩きすぎ
            "contact_limit": torch.zeros(num_envs, device=self.device),
            "joint_limit": torch.zeros(num_envs, device=self.device)
        }

    def reset_idx(self, env_ids: torch.Tensor):
        self.hit_state[env_ids] = 0
        self.current_peak_force[env_ids] = 0.0
        self.contact_duration[env_ids] = 0.0
        self.peak_timing_scale[env_ids] = 0.0
        
        # 新規フラグのリセット
        self.hit_success_in_window[env_ids] = False
        self.prev_is_rest[env_ids] = True
        
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

        # ----------------------------------------------------
        # 1. 区間判定 (Note vs Rest) & イベント検出
        # ----------------------------------------------------
        is_touching = (force_z > self.hit_threshold)
        
        # ターゲット強度による区間判定
        is_rest_period = (target_force_trace < self.rest_threshold)
        is_note_period = (~is_rest_period)

        # A. Note Onset (休符 -> 音符): 新しい打撃チャンスの開始
        # 前ステップが休符かつ、現在が音符なら「ウィンドウ開始」
        note_onset = self.prev_is_rest & is_note_period
        if note_onset.any():
            self.hit_success_in_window[note_onset] = False # フラグをリセット（まだ叩いてない）

        # B. Note Offset (音符 -> 休符): 打撃チャンスの終了確認
        # 音符が終わったのにまだ成功フラグがFalseなら「叩き損ね (Miss)」
        note_offset = (~self.prev_is_rest) & is_rest_period
        miss_penalty_term = torch.zeros(self.num_envs, device=self.device)
        
        if note_offset.any():
            ids = torch.where(note_offset)[0]
            missed_indices = ids[~self.hit_success_in_window[ids]]
            if len(missed_indices) > 0:
                # ペナルティ項として1.0を計上 (重みは後で掛ける)
                miss_penalty_term[missed_indices] = 1.0

        terms["miss_penalty"] = miss_penalty_term

        # 接触時間の更新
        self.contact_duration = torch.where(
            is_touching,
            self.contact_duration + dt,
            torch.zeros_like(self.contact_duration)
        )

        # ----------------------------------------------------
        # 2. 打撃検出 (Match) & 重複打撃判定
        # ----------------------------------------------------
        match_reward = torch.zeros(self.num_envs, device=self.device)
        double_hit_penalty_term = torch.zeros(self.num_envs, device=self.device)

        # Rising Edge (接触開始)
        rising = (self.hit_state == 0) & is_touching
        if rising.any():
            ids = torch.where(rising)[0]
            self.hit_state[ids] = 1 
            self.current_peak_force[ids] = force_z[ids]
            current_scale = (target_force_trace[ids] / self.target_ref_val).clamp(0.0, 1.0)
            self.peak_timing_scale[ids] = current_scale
        
        # Sustain (接触継続・ピーク更新)
        sustain = (self.hit_state == 1) & is_touching
        if sustain.any():
            ids = torch.where(sustain)[0]
            in_window = (self.contact_duration[ids] <= self.impact_window)
            
            if in_window.any():
                upd_ids = ids[in_window]
                curr_force = force_z[upd_ids]
                prev_peak = self.current_peak_force[upd_ids]
                
                is_new_peak = (curr_force > prev_peak)
                if is_new_peak.any():
                    peak_ids = upd_ids[is_new_peak]
                    self.current_peak_force[peak_ids] = curr_force[is_new_peak]
                    # ピーク更新時のターゲット値をタイミングスコアとして記録
                    t_scale = (target_force_trace[peak_ids] / self.target_ref_val).clamp(0.0, 1.0)
                    self.peak_timing_scale[peak_ids] = t_scale

        # Falling Edge (離脱・報酬確定)
        falling = (self.hit_state != 0) & (~is_touching)
        if falling.any():
            ids = torch.where(falling)[0]
            valid_hits = (self.hit_state[ids] == 1) # Pushing等の無効ステートでないか
            
            if valid_hits.any():
                valid_ids = ids[valid_hits]
                
                # --- ケース分け: 音符区間内の打撃か、そうでないか ---
                # 判定には「離脱した瞬間」の区間情報(is_note_period)を使用
                hit_in_note = is_note_period[valid_ids]
                
                # 1. 音符区間でのヒット
                note_hit_ids = valid_ids[hit_in_note]
                if len(note_hit_ids) > 0:
                    # すでに成功済みかチェック
                    already_success = self.hit_success_in_window[note_hit_ids]
                    
                    # (A) 初回成功 -> Match報酬付与
                    first_hit_ids = note_hit_ids[~already_success]
                    if len(first_hit_ids) > 0:
                        force_score = self._evaluate_hit(
                            self.current_peak_force[first_hit_ids], 
                            target_force_ref[first_hit_ids]
                        )
                        # タイミング緩和: ターゲット波形が立ち上がっていれば(>5%)OK
                        timing_ok = (self.peak_timing_scale[first_hit_ids] > 0.05).float()
                        match_reward[first_hit_ids] = force_score * timing_ok
                        
                        # 成功フラグを立てる (同じ区間での2回目を防ぐ)
                        self.hit_success_in_window[first_hit_ids] = True

                    # (B) 2回目以降 -> 重複打撃(Chattering)ペナルティ
                    dup_hit_ids = note_hit_ids[already_success]
                    if len(dup_hit_ids) > 0:
                        double_hit_penalty_term[dup_hit_ids] = 1.0

                # 2. 休符区間でのヒット (余計な打撃)
                # 休符中の打撃も「重複打撃」と同様にペナルティ扱いとする
                rest_hit_ids = valid_ids[~hit_in_note]
                if len(rest_hit_ids) > 0:
                    double_hit_penalty_term[rest_hit_ids] = 1.0

            # 状態のリセット
            self.hit_state[ids] = 0
            self.current_peak_force[ids] = 0.0
            self.peak_timing_scale[ids] = 0.0

        terms["match"] = match_reward
        terms["double_hit_penalty"] = double_hit_penalty_term

        # ----------------------------------------------------
        # 3. その他の項 (Rest, Limit)
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

        # ----------------------------------------------------
        # 4. 報酬合計
        # ----------------------------------------------------
        # 重み係数は cfg に存在しない場合に備えて getattr でデフォルト値を設定
        # miss_penalty と double_hit_penalty の重みは負の値（罰則）を設定
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
        
        # 次回ループ用に状態更新
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