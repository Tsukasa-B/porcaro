# rewards.py
from __future__ import annotations
import torch
from typing import Sequence # reset_idx のために追加

# --- 必要なモジュールをインポート ---
from ..cfg.rewards_cfg import RewardsCfg

class RewardManager:
    """
    報酬計算と終了判定を管理するクラス。
    (env._get_rewards / env._get_dones から呼び出される)
    """
    def __init__(self, cfg: RewardsCfg, num_envs: int, device: str | torch.device):
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = torch.device(device)
        
        # --- 論文設定値 ---
        self.hit_threshold = float(cfg.hit_threshold_force)
        self.target_force = float(cfg.target_force_fd)
        self.sigma_f = float(cfg.sigma_f)
        self.w1 = float(cfg.weight_w1_force)
        self.w2 = float(cfg.weight_w2_hit_count)

        # --- 内部状態バッファ ---
        self.hit_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.first_hit_force = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)

        # --- (変更) ステートマシン用の状態バッファ ---
        # 0=IDLE (閾値未満), 1=RISING (閾値以上・上昇中), 2=FALLING (閾値以上・下降中)
        self.hit_state = torch.zeros(self.num_envs, device=self.device, dtype=torch.int8)
        # 現在の「山」のピーク値を一時的に保持するバッファ
        self.current_peak_force = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)

        # --- 終了判定バッファ ---
        self.terminated_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.truncated_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

    def reset_idx(self, env_ids: torch.Tensor | Sequence[int]):
        """指定された環境の内部状態をリセット"""
        self.hit_count[env_ids] = 0
        self.first_hit_force[env_ids] = 0.0
        # (変更) 状態バッファもリセット
        self.hit_state[env_ids] = 0 # IDLE
        self.current_peak_force[env_ids] = 0.0
        self.terminated_buf[env_ids] = False
        self.truncated_buf[env_ids] = False

    def _process_hit_event(self, env_ids: torch.Tensor, peak_force: torch.Tensor, rewards: torch.Tensor):
        """
        (新規) ヒットイベント（カウントと報酬計算）を処理するヘルパー関数
        """
        if len(env_ids) == 0:
            return

        # (A) ヒットカウントを増やす
        self.hit_count[env_ids] += 1
        
        # (B) 1回目のヒットだったか？
        is_first_hit_mask = (self.hit_count[env_ids] == 1)
        first_hit_ids = env_ids[is_first_hit_mask]
        
        if len(first_hit_ids) > 0:
            # (C) F1 (ピーク力) を確定
            f1_forces = peak_force[is_first_hit_mask] # peak_force は env_ids のサブセット
            self.first_hit_force[first_hit_ids] = f1_forces
            
            # (D) 報酬 r_F1 の計算
            r_f1_values = torch.exp(-0.5 * torch.square((f1_forces - self.target_force) / self.sigma_f))
            rewards[first_hit_ids] += self.w1 * r_f1_values

    def compute_reward_and_dones(self, 
                                 # (変更) 引数を Z軸力「履歴」に変更
                                 force_z_history: torch.Tensor,
                                 episode_length_buf: torch.Tensor,
                                 max_episode_length: int
                                 ) -> torch.Tensor:
        """
        (変更) 物理ステップの履歴を1つずつ処理するステートマシン
        """
        
        # 0. ステップ報酬を初期化
        rewards = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        
        # decimation (履歴の長さ) を取得
        decimation = force_z_history.shape[1]

        # --- 1. 物理ステップの履歴を時系列順（古いものから）に処理 ---
        # 履歴は [0] が最新, [decimation-1] が最古
        for i in range(decimation - 1, -1, -1):
            
            current_force = force_z_history[:, i].reshape(-1) #.squeeze()
            is_above_threshold = (current_force >= self.hit_threshold)
            
            # 現在の状態を取得
            state_idle = (self.hit_state == 0)
            state_rising = (self.hit_state == 1)
            state_falling = (self.hit_state == 2)
            
            # --- ステートマシンの遷移 ---
            
            # (A) IDLE -> RISING (イベント開始)
            rising_edge_mask = state_idle & is_above_threshold
            rising_edge_ids = torch.where(rising_edge_mask)[0]
            if len(rising_edge_ids) > 0:
                self.hit_state[rising_edge_ids] = 1 # RISING
                self.current_peak_force[rising_edge_ids] = current_force[rising_edge_ids]

            # (B) RISING -> (ピーク更新 or FALLING)
            rising_state_mask = state_rising & is_above_threshold
            rising_state_ids = torch.where(rising_state_mask)[0]
            if len(rising_state_ids) > 0:
                current_force_subset = current_force[rising_state_ids].reshape(-1)
                peak_buffer_subset = self.current_peak_force[rising_state_ids].reshape(-1)
                
                # B-1: まだ上昇中 -> ピーク更新
                still_going_up_mask = (current_force_subset >= peak_buffer_subset)
                going_up_ids = rising_state_ids[still_going_up_mask]
                self.current_peak_force[going_up_ids] = current_force_subset[still_going_up_mask]
                
                # B-2: ピークを過ぎて下降開始 -> FALLING へ
                now_falling_mask = (current_force_subset < peak_buffer_subset)
                now_falling_ids = rising_state_ids[now_falling_mask]
                self.hit_state[now_falling_ids] = 2 # FALLING

            # (C) RISING -> IDLE (閾値以下に落下)
            rising_fell_off_mask = state_rising & (~is_above_threshold)
            rising_fell_off_ids = torch.where(rising_fell_off_mask)[0]
            if len(rising_fell_off_ids) > 0:
                self.hit_state[rising_fell_off_ids] = 0 # IDLE
                # ヒットイベント処理
                self._process_hit_event(rising_fell_off_ids, self.current_peak_force[rising_fell_off_ids], rewards)
                self.current_peak_force[rising_fell_off_ids] = 0.0

            # (D) FALLING -> (RISING or IDLE)
            falling_state_mask = state_falling & is_above_threshold
            falling_state_ids = torch.where(falling_state_mask)[0]
            if len(falling_state_ids) > 0:
                current_force_subset = current_force[falling_state_ids]
                peak_buffer_subset = self.current_peak_force[falling_state_ids]
                
                # D-1: 再び上昇開始 (ダブルヒット)
                rising_again_mask = (current_force_subset >= peak_buffer_subset)
                rising_again_ids = falling_state_ids[rising_again_mask]
                if len(rising_again_ids) > 0:
                    self.hit_state[rising_again_ids] = 1 # RISING
                    # (重要) 前の山のピークでヒットイベント処理
                    self._process_hit_event(rising_again_ids, self.current_peak_force[rising_again_ids], rewards)
                    # 新しい山のピークでバッファを初期化
                    self.current_peak_force[rising_again_ids] = current_force_subset[rising_again_mask]

                # D-2: (まだ下降中 -> 何もしない)
                
            # (E) FALLING -> IDLE (イベント終了)
            falling_fell_off_mask = state_falling & (~is_above_threshold)
            falling_fell_off_ids = torch.where(falling_fell_off_mask)[0]
            if len(falling_fell_off_ids) > 0:
                self.hit_state[falling_fell_off_ids] = 0 # IDLE
                # ヒットイベント処理
                self._process_hit_event(falling_fell_off_ids, self.current_peak_force[falling_fell_off_ids], rewards)
                self.current_peak_force[falling_fell_off_ids] = 0.0

        # --- 2. 終了判定 (ループの外) ---
        # (重要) 物理ステップの処理後、最終的な hit_count で判定
        self.terminated_buf = (self.hit_count >= 2)
        
        self.truncated_buf = (episode_length_buf >= max_episode_length)

        # --- 3. 報酬 r_h の計算 (タイムアウト時) ---
        truncated_ids = torch.where(self.truncated_buf)[0]
        if len(truncated_ids) > 0:
            hit_once_and_truncated_ids = truncated_ids[self.hit_count[truncated_ids] == 1]
            if len(hit_once_and_truncated_ids) > 0:
                rewards[hit_once_and_truncated_ids] += self.w2

        # env にステップ報酬を返す
        return rewards

    def get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """計算済みの終了判定バッファを返す"""
        return self.terminated_buf, self.truncated_buf
    
    def get_first_hit_force(self) -> torch.Tensor:
        """エピソードで記録された F1 (最初の打撃力) を返す"""
        return self.first_hit_force