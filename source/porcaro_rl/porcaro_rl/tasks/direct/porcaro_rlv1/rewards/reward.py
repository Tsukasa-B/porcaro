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
                        target_force_trace: torch.Tensor,
                        dt: float) -> dict[str, torch.Tensor]:
        """
        各項 (r_1, r_2...) を個別に計算して返す
        """
        # 報酬項を格納する辞書
        terms = {}

        # --- 共通の状況判定 ---
        is_touching = (force_z > self.hit_threshold)
        is_rest_period = (target_force_trace < 1.0) # ターゲットがほぼ0なら休符

        # 接触時間の更新
        self.contact_duration = torch.where(
            is_touching,
            self.contact_duration + dt,
            torch.zeros_like(self.contact_duration)
        )

        # ============================================================
        # r_1: イベント駆動マッチング報酬 (Event-Driven Hit Reward)
        # ============================================================
        # 計算ロジック:
        # 叩き終わって離れた瞬間(Falling Edge)に、その打撃のピーク力とターゲットを比較する
        hit_reward = torch.zeros(self.num_envs, device=self.device)
        
        # 1. Rising Edge (接触開始) -> 計測開始
        rising = (self.hit_state == 0) & is_touching
        if rising.any():
            ids = torch.where(rising)[0]
            self.hit_state[ids] = 1
            self.current_peak_force[ids] = force_z[ids]
        
        # 2. Sustain (接触中) -> ピーク更新
        sustain = (self.hit_state == 1) & is_touching
        if sustain.any():
            ids = torch.where(sustain)[0]
            curr = force_z[ids]
            peak = self.current_peak_force[ids]
            self.current_peak_force[ids] = torch.max(curr, peak)
            
        # 3. Falling Edge (離脱) -> 報酬確定
        falling = (self.hit_state == 1) & (~is_touching)
        if falling.any():
            ids = torch.where(falling)[0]
            self.hit_state[ids] = 0 # IDLEへ
            
            # ここで打撃の評価を行う
            hit_reward[ids] = self._evaluate_hit(
                peak_force=self.current_peak_force[ids],
                target_val=target_force_trace[ids]
            )
            # 次回用にリセット
            self.current_peak_force[ids] = 0.0

        terms["match"] = hit_reward

        # ============================================================
        # r_2: 休符ペナルティ (Rest Violation)
        # ============================================================
        # 休符指示なのに触っている場合
        # 強度(force_z)に比例させるか、単に定数ペナルティにするか
        # ここでは「触ってしまった強さ」に比例させて、軽く触れただけなら許す設計
        violation = is_rest_period & is_touching
        terms["rest_penalty"] = torch.where(
            violation,
            -1.0 * (force_z / 10.0), # 10Nで -1.0 (正規化の一環)
            torch.zeros(self.num_envs, device=self.device)
        )

        # ============================================================
        # r_3: 連続接触(押し付け)ペナルティ (Contact Duration)
        # ============================================================
        is_over = (self.contact_duration > self.cfg.max_contact_duration_s)
        over_time = (self.contact_duration - self.cfg.max_contact_duration_s).clamp(min=0.0)
        terms["contact_limit"] = -1.0 * over_time # 秒数そのものをペナルティ係数に

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
        """
        1回の打撃イベントを評価し、0.0〜1.0の正規化されたスコアを返す。
        """
        # 誤差の絶対値
        force_error = torch.abs(peak_force - self.target_force_ref)
        
        # ガウス分布スコア (誤差0で1.0)
        # exp(-error / sigma)
        score = torch.exp(-force_error / self.cfg.sigma_force)
        
        # ★ここがポイント: 正規化の工夫★
        if self.cfg.scale_reward_by_force_magnitude:
            # 従来: ターゲットが大きいほど報酬もデカくなる (20N指示なら満点、5N指示なら0.25点など)
            # これだと「強い音」ばかり学習したがる
            magnitude_factor = (target_val / self.target_force_ref).clamp(max=1.5)
            return score * magnitude_factor
        else:
            # 推奨: ターゲットが何Nであれ、その通りに叩けたら満点(1.0)
            # これにより、弱奏(5N)も強奏(20N)も等しく価値があることになる
            # ただし「休符(target<1.0)」の誤検知を防ぐため、ターゲットが極端に小さい時はマスクする
            is_valid_note = (target_val >= 1.0).float()
            return score * is_valid_note