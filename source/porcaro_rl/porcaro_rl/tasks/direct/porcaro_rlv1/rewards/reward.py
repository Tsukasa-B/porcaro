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
        
        self.rest_threshold = 1.0 
        self.target_ref_val = float(cfg.target_force_fd)

        # --- 状態管理用バッファ ---
        self.hit_state = torch.zeros(num_envs, dtype=torch.long, device=self.device) 
        self.current_peak_force = torch.zeros(num_envs, device=self.device)
        self.contact_duration = torch.zeros(num_envs, device=self.device)
        self.peak_timing_scale = torch.zeros(num_envs, device=self.device)

        # 【変更点1】 フラグ管理を「時刻」管理に変更
        # 最後にヒット判定を行い、報酬を与えた「エピソード内経過時間(秒)」を記録
        # これにより、「前のヒットから何秒経ったか」で判定できるようになる
        self.last_reward_time = torch.full((num_envs,), -1.0, device=self.device)
        
        # ダブルストロークを許可する最小間隔 (秒)
        # BPM160の16分音符 = 約93ms。これより短く設定する。
        # ここでは「60ms (0.06s)」経過していれば、同じ音符区間内でも2発目を認める
        self.double_hit_cooltime = 0.06

        # 前回のステップが休符だったかどうか
        self.prev_is_rest = torch.ones(num_envs, dtype=torch.bool, device=self.device)

        # エピソード時間を管理するカウンタ (dt積算用)
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
        
        # リセット
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
                        dt: float) -> dict[str, torch.Tensor]:
        
        # 時間を進める
        self.current_time_s += dt

        terms = {}
        is_touching = (force_z > self.hit_threshold)
        is_rest_period = (target_force_trace < self.rest_threshold)
        is_note_period = (~is_rest_period)

        # ----------------------------------------------------
        # 1. Miss Penalty の判定
        # ----------------------------------------------------
        # 音符が終わった瞬間に、「この音符区間で一度も報酬を貰ってない」ならMiss
        # 判定基準: (現在時刻 - 最後に報酬貰った時刻) が (音符の長さ + α) 以上開いているか？
        # ただ、簡易的にやるなら「音符終了エッジ」で判定するのが安全
        note_offset = (~self.prev_is_rest) & is_rest_period
        
        miss_penalty_term = torch.zeros(self.num_envs, device=self.device)
        if note_offset.any():
            ids = torch.where(note_offset)[0]
            
            # 最後に報酬を得たのが「ついさっき(今の音符区間内)」であればOK
            # 今の音符が始まったと推定される時刻より後か？
            # 簡易判定: last_reward_time が -1 (未達成) か、古すぎる(0.2秒以上前)ならMiss
            # ※ここは厳密にしすぎると難しいので、BPM160の1拍分(0.375s)くらい余裕を見る
            time_diff = self.current_time_s[ids] - self.last_reward_time[ids]
            
            # 0.25秒以上ヒットがなければ「叩き損ね」とみなす
            missed = (time_diff > 0.25) | (self.last_reward_time[ids] < 0.0)
            
            if missed.any():
                miss_penalty_term[ids[missed]] = 1.0

        terms["miss_penalty"] = miss_penalty_term

        # 接触時間の更新
        self.contact_duration = torch.where(
            is_touching,
            self.contact_duration + dt,
            torch.zeros_like(self.contact_duration)
        )

        # ----------------------------------------------------
        # 2. 打撃検出 (Match) & クールタイム付き連打判定
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
        
        # Sustain (ピーク更新)
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
                    t_scale = (target_force_trace[peak_ids] / self.target_ref_val).clamp(0.0, 1.0)
                    self.peak_timing_scale[peak_ids] = t_scale

        # Falling Edge (離脱・報酬確定)
        falling = (self.hit_state != 0) & (~is_touching)
        if falling.any():
            ids = torch.where(falling)[0]
            valid_hits = (self.hit_state[ids] == 1) 
            
            if valid_hits.any():
                valid_ids = ids[valid_hits]
                
                # --- ケース分け ---
                # 1. 音符区間でのヒットか？
                hit_in_note = is_note_period[valid_ids]
                
                # 2. 前回のヒットから十分時間が経っているか？ (クールタイム判定)
                time_since_last = self.current_time_s[valid_ids] - self.last_reward_time[valid_ids]
                is_cooled_down = (time_since_last > self.double_hit_cooltime)

                # ==================================================
                # ★ここが修正の肝: 
                # 「音符区間内」かつ「クールタイム経過済み」なら
                # 何回目だろうと報酬を与える！
                # ==================================================
                rewardable_hits = hit_in_note & is_cooled_down
                
                # A. 報酬付与 (Match)
                success_ids = valid_ids[rewardable_hits]
                if len(success_ids) > 0:
                    force_score = self._evaluate_hit(
                        self.current_peak_force[success_ids], 
                        target_force_ref[success_ids]
                    )
                    timing_ok = (self.peak_timing_scale[success_ids] > 0.05).float()
                    match_reward[success_ids] = force_score * timing_ok
                    
                    # ★成功時刻を更新 (これで次のヒットまでクールタイムが発生)
                    self.last_reward_time[success_ids] = self.current_time_s[success_ids]

                # B. ペナルティ (Double Hit / Chattering)
                # 「音符区間内」だが「クールタイムが終わっていない（連打早すぎ）」場合のみ罰する
                too_fast_ids = valid_ids[hit_in_note & (~is_cooled_down)]
                if len(too_fast_ids) > 0:
                    double_hit_penalty_term[too_fast_ids] = 1.0
                
                # C. 休符区間でのヒット (Rest Penalty)
                # 休符中のヒットもここに入れても良いが、今回はDoubleHitとして扱う
                rest_hit_ids = valid_ids[~hit_in_note]
                if len(rest_hit_ids) > 0:
                    double_hit_penalty_term[rest_hit_ids] = 1.0

            # 状態のリセット
            self.hit_state[ids] = 0
            self.current_peak_force[ids] = 0.0
            self.peak_timing_scale[ids] = 0.0

        terms["match"] = match_reward
        terms["double_hit_penalty"] = double_hit_penalty_term
        terms["miss_penalty"] = miss_penalty_term # 注: 上部で計算済み

        # ----------------------------------------------------
        # 3. その他の項 (Rest, Limit) - 変更なし
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
            # (省略: 変更なし)
            terms["joint_limit"] = torch.zeros(self.num_envs, device=self.device) 
        else:
            terms["joint_limit"] = torch.zeros(self.num_envs, device=self.device)

        # ----------------------------------------------------
        # 4. 報酬合計
        # ----------------------------------------------------
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