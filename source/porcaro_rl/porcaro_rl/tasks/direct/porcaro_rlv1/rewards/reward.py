# rewards/reward.py
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
        
        # 設定値のキャッシュ
        self.hit_threshold = float(cfg.hit_threshold_force)
        self.target_force = float(cfg.target_force_fd)
        self.target_bpm = float(cfg.target_bpm)
        self.beat_interval = 60.0 / self.target_bpm
        
        # ガウス関数のパラメータ
        self.sigma_f = float(cfg.sigma_f)
        self.sigma_t = float(cfg.sigma_t)
        
        # 重み
        self.w_force = float(cfg.weight_force)
        self.w_timing = float(cfg.weight_timing)

        # 内部状態
        self.hit_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        
        # ステートマシン (0=IDLE, 1=RISING, 2=FALLING)
        self.hit_state = torch.zeros(self.num_envs, device=self.device, dtype=torch.int8)
        self.current_peak_force = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        # ピーク時の時刻を記録するバッファ
        self.current_peak_time = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)

        # 終了判定バッファ
        self.terminated_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.truncated_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        # ▼▼▼ 追加: 最新のヒットのピーク力を保持するバッファ ▼▼▼
        self.last_hit_peak_force = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)

    def reset_idx(self, env_ids: torch.Tensor | Sequence[int]):
        """リセット処理"""
        self.hit_count[env_ids] = 0
        self.hit_state[env_ids] = 0
        self.current_peak_force[env_ids] = 0.0
        self.current_peak_time[env_ids] = 0.0
        self.terminated_buf[env_ids] = False
        self.truncated_buf[env_ids] = False
        self.last_hit_peak_force[env_ids] = 0.0 # リセット時は0に戻す

    def _process_hit_event(self, env_ids: torch.Tensor, peak_forces: torch.Tensor, peak_times: torch.Tensor, rewards: torch.Tensor):
        """
        打撃確定時の報酬計算 (Force + Timing)
        """
        if len(env_ids) == 0:
            return

        # 1. カウント更新
        self.hit_count[env_ids] += 1
        
        # 2. 力の報酬 (r_F)
        # target_force との誤差
        force_error = peak_forces - self.target_force
        r_force = torch.exp(-0.5 * torch.square(force_error / self.sigma_f))
        
        # 3. タイミングの報酬 (r_T)
        # 「最も近いビート」とのズレを計算
        # nearest_beat = round(t / interval) * interval
        # ※ round は .5 で偶数丸めになることがあるが、リズム用途なら許容範囲
        beat_indices = torch.round(peak_times / self.beat_interval)
        nearest_beat_times = beat_indices * self.beat_interval
        
        timing_error = peak_times - nearest_beat_times
        r_timing = torch.exp(-0.5 * torch.square(timing_error / self.sigma_t))
        
        # 4. 報酬の合算
        # hitした瞬間にドカンと報酬を与える
        total_reward = (self.w_force * r_force) + (self.w_timing * r_timing)
        rewards[env_ids] += total_reward

        # ▼▼▼ 追加: ピーク値を記録（次のヒットまで保持される） ▼▼▼
        self.last_hit_peak_force[env_ids] = peak_forces

    def compute_reward_and_dones(self, 
                                 force_z_history: torch.Tensor, # (num_envs, decimation)
                                 current_time_s: torch.Tensor,  # (num_envs,) 現在時刻
                                 dt_step: float,                # 物理ステップ時間 (1/200s)
                                 episode_length_buf: torch.Tensor,
                                 max_episode_length: int
                                 ) -> torch.Tensor:
        """
        物理ステップごとの履歴を走査してヒット検知 & 報酬計算
        """
        rewards = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        decimation = force_z_history.shape[1]

        # 履歴を古い順 ([decimation-1] -> [0]) に処理
        # 時刻も遡って計算する必要がある
        
        # current_time_s は「この制御ステップの終了時刻」を指す
        # history[0] の時刻 = current_time_s - dt_step
        # history[i] の時刻 = current_time_s - (i + 1) * dt_step
        
        for i in range(decimation - 1, -1, -1):
            # 該当ステップの力
            f_val = force_z_history[:, i]
            # 該当ステップの時刻
            # i=0 (最新) -> time - 1*dt
            # i=decimation-1 (最古) -> time - decimation*dt
            t_val = current_time_s - (i + 1) * dt_step
            
            is_above = (f_val >= self.hit_threshold)
            
            # ステート遷移ロジック
            # (IDLE -> RISING, RISING -> FALLING 等は前回のコードと同様)
            # ただし、ピーク更新時に「その時刻」も記録する点が異なる
            
            # --- 状態マスク作成 ---
            state_idle = (self.hit_state == 0)
            state_rising = (self.hit_state == 1)
            state_falling = (self.hit_state == 2)

            # (A) IDLE -> RISING
            rising_edge = state_idle & is_above
            if rising_edge.any():
                ids = torch.where(rising_edge)[0]
                self.hit_state[ids] = 1
                self.current_peak_force[ids] = f_val[ids]
                self.current_peak_time[ids] = t_val[ids] # 時刻記録

            # (B) RISING -> (Update Peak or FALLING)
            rising_mask = state_rising & is_above
            if rising_mask.any():
                ids = torch.where(rising_mask)[0]
                curr_f = f_val[ids]
                peak_f = self.current_peak_force[ids]
                
                # まだ上がっている
                up_mask = (curr_f >= peak_f)
                up_ids = ids[up_mask]
                self.current_peak_force[up_ids] = curr_f[up_mask]
                self.current_peak_time[up_ids] = t_val[up_ids] # 時刻更新
                
                # 下がり始めた -> FALLING
                down_mask = (curr_f < peak_f)
                down_ids = ids[down_mask]
                self.hit_state[down_ids] = 2

            # (C) RISING/FALLING -> IDLE (イベント終了＝打撃確定)
            # RISINGから直接落ちるケースと、FALLINGから落ちるケース
            # どちらも「閾値を割った」瞬間にイベント確定とする
            falling_edge = (state_rising | state_falling) & (~is_above)
            if falling_edge.any():
                ids = torch.where(falling_edge)[0]
                self.hit_state[ids] = 0 # IDLEへ
                
                # ★ここで報酬計算★
                self._process_hit_event(
                    ids, 
                    self.current_peak_force[ids], 
                    self.current_peak_time[ids], 
                    rewards
                )
                
                # バッファクリア
                self.current_peak_force[ids] = 0.0
                self.current_peak_time[ids] = 0.0

            # (D) FALLING -> RISING (ダブルヒット: リバウンド中に再上昇)
            # これは高度な制御だが、一旦「新しい打撃」として扱うか、
            # あるいは「1つの打撃の続き」とするか。
            # ここではシンプルに「前の打撃を確定させて、新しい打撃を開始」する
            re_rising = state_falling & is_above
            if re_rising.any():
                ids = torch.where(re_rising)[0]
                curr_f = f_val[ids]
                peak_f = self.current_peak_force[ids]
                
                # ピークを超えて再上昇した場合のみ処理
                # (ノイズで微増しただけなら無視したいが、簡易的に上昇なら即検知)
                rising_again = (curr_f > peak_f) # 単純比較
                ra_ids = ids[rising_again]
                
                if len(ra_ids) > 0:
                    # 前の山を確定させる
                    self._process_hit_event(
                        ra_ids, 
                        self.current_peak_force[ra_ids], 
                        self.current_peak_time[ra_ids], 
                        rewards
                    )
                    # 新しい山の開始
                    self.hit_state[ra_ids] = 1 # RISING
                    self.current_peak_force[ra_ids] = curr_f[rising_again]
                    self.current_peak_time[ra_ids] = t_val[ra_ids]

        # --- 終了判定 ---
        # 時間切れ (Truncated) のみ判定し、打撃回数による終了 (Terminated) はしない
        self.truncated_buf = (episode_length_buf >= max_episode_length)
        self.terminated_buf[:] = False # 常にFalse (転倒などがない限り)

        return rewards

    def get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.terminated_buf, self.truncated_buf
    
    def get_first_hit_force(self) -> torch.Tensor:
        """ログ用に、最新の打撃のピーク力を返す"""
        # ▼▼▼ 修正: 保持しておいたバッファを返す ▼▼▼
        return self.last_hit_peak_force