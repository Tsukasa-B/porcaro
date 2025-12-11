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

        # ▼▼▼ 追加: ペナルティ用パラメータ ▼▼▼
        self.w_penalty = float(cfg.weight_contact_penalty)
        self.safe_window = float(cfg.penalty_safe_window)
        self.w_double_hit = float(cfg.weight_double_hit_penalty)
        # ---------------------------------------

        # 内部状態
        self.hit_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        # ▼▼▼ 修正: アーミング（タメ）判定用のパラメータ ▼▼▼
        # 最低これだけの時間、空中にいないと「次の打撃」として認めない
        # 120BPMの16分音符(0.125s)の半分くらいを目安に (0.05s = 50ms)
        self.arming_duration_threshold = 0.05
        
        # 内部状態
        self.air_time_counter = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.is_armed = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        
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
        # ▼▼▼ 追加: 最後に報酬を与えたビートのインデックス (-1で初期化) ▼▼▼
        self.last_rewarded_beat = torch.full((self.num_envs,), -1, device=self.device, dtype=torch.long)

    def reset_idx(self, env_ids: torch.Tensor | Sequence[int]):
        """リセット処理"""
        self.hit_count[env_ids] = 0
        self.hit_state[env_ids] = 0
        self.current_peak_force[env_ids] = 0.0
        self.current_peak_time[env_ids] = 0.0
        self.terminated_buf[env_ids] = False
        self.truncated_buf[env_ids] = False
        self.last_hit_peak_force[env_ids] = 0.0 # リセット時は0に戻す
        self.air_time_counter[env_ids] = 0.0
        self.is_armed[env_ids] = True # 最初は「装填済み」からスタートしてあげる（最初の1打のため）
        # ▼▼▼ 追加: ビート管理バッファのリセット ▼▼▼
        self.last_rewarded_beat[env_ids] = -1

    def _process_hit_event(self, env_ids: torch.Tensor, peak_forces: torch.Tensor, peak_times: torch.Tensor, rewards: torch.Tensor):
        """
        打撃確定時の報酬計算 (Force + Timing + Double Hit Penalty)
        """
        if len(env_ids) == 0:
            return

        # 1. 今叩いたのが「何番目のビート」を狙ったものか計算
        #    例: 0.48秒 -> 0.5秒間隔なら「1番目のビート」
        current_beat_indices = torch.round(peak_times / self.beat_interval).long()
        
        # 2. 重複チェック
        #    「記録されているビート番号」と「今回のビート番号」が違うなら新規 (True)
        #    同じなら重複 (False)
        is_new_beat_mask = (current_beat_indices != self.last_rewarded_beat[env_ids])
        
        # --- 報酬計算 (Force + Timing) ---
        
        # 力の評価 (Target Forceとの誤差)
        force_error = peak_forces - self.target_force
        r_force = torch.exp(-0.5 * torch.square(force_error / self.sigma_f))
        
        # タイミングの評価 (Target Timeとの誤差)
        nearest_beat_times = current_beat_indices * self.beat_interval
        timing_error = peak_times - nearest_beat_times
        r_timing = torch.exp(-0.5 * torch.square(timing_error / self.sigma_t))
        
        # ベース報酬 (掛け算ロジック: タイミングが合わないと力報酬もゼロ)
        base_score = (self.w_force * r_force) + self.w_timing
        hit_reward = base_score * r_timing
        
        # --- 最終的な加算値の決定 ---
        
        # A. 新規ヒットの場合 (is_new_beat_mask=True):
        #    正当な報酬を与える (hit_reward)
        #
        # B. 重複ヒットの場合 (is_new_beat_mask=False):
        #    報酬は与えず、ペナルティを与える (-w_double_hit)
        
        # マスクを float に変換 (True->1.0, False->0.0)
        mask_float = is_new_beat_mask.float()
        
        # 最終スコア = (報酬 * マスク) - (ペナルティ * 反転マスク)
        final_score = (hit_reward * mask_float) - (self.w_double_hit * (1.0 - mask_float))
        
        rewards[env_ids] += final_score
        
        # --- 状態更新 ---
        
        # ヒットカウントを加算
        self.hit_count[env_ids] += 1 
        
        # 新しく報酬を与えた環境だけ、ビート番号を更新して記録
        # (これにより、同じビート番号での次回の打撃は重複判定される)
        valid_ids = env_ids[is_new_beat_mask]
        if len(valid_ids) > 0:
            self.last_rewarded_beat[valid_ids] = current_beat_indices[is_new_beat_mask]

        # ログ用にピーク力を記録
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
            
            is_touching = (f_val >= self.hit_threshold)

            # --- ▼▼▼ アーミング（充填）ロジック ▼▼▼ ---
            # 触っていなければカウンターを増やす、触っていたらリセット
            # (ベクトル演算のため、whereを使って条件分岐)
            self.air_time_counter = torch.where(
                ~is_touching, 
                self.air_time_counter + dt_step, 
                torch.zeros_like(self.air_time_counter)
            )
            
            # 一定時間浮いていたら「装填 (ARMED)」
            just_armed = (self.air_time_counter >= self.arming_duration_threshold)
            self.is_armed = self.is_armed | just_armed
            # ---------------------------------------------

            # --- ▼▼▼ 追加: 接触ペナルティの計算 ▼▼▼ ---
            
            # 最も近いビート時刻との距離を計算
            # beat_index = round(t / interval)
            beat_indices = torch.round(t_val / self.beat_interval)
            nearest_beat_times = beat_indices * self.beat_interval
            time_diff = torch.abs(t_val - nearest_beat_times)
            
            # 「ウィンドウの外側」かつ「接触している」ならペナルティ
            is_illegal_contact = (time_diff > self.safe_window) & is_touching
            
            if is_illegal_contact.any():
                # 該当する環境IDに減点 (broadcast)
                rewards[is_illegal_contact] -= self.w_penalty
            
            # ---------------------------------------------
            
            # ステート遷移ロジック
            # (IDLE -> RISING, RISING -> FALLING 等は前回のコードと同様)
            # ただし、ピーク更新時に「その時刻」も記録する点が異なる
            
            # --- 状態マスク作成 ---
            state_idle = (self.hit_state == 0)
            state_rising = (self.hit_state == 1)
            state_falling = (self.hit_state == 2)

            # (A) IDLE -> RISING
            # ★ここで「装填されているか？」をチェック
            rising_edge = (self.hit_state == 0) & is_touching
            if rising_edge.any():
                ids = torch.where(rising_edge)[0]
                
                # 装填されている環境だけ RISING に移行する（＝有効打撃の候補にする）
                # 装填されていない（押し付け中の）環境は IDLE のまま無視する
                valid_start_ids = ids[self.is_armed[ids]]
                
                if len(valid_start_ids) > 0:
                    self.hit_state[valid_start_ids] = 1 # RISING
                    self.current_peak_force[valid_start_ids] = f_val[valid_start_ids]
                    self.current_peak_time[valid_start_ids] = t_val[valid_start_ids]
                    
                    # 打撃を開始したので「弾」を消費（未装填に戻す）
                    self.is_armed[valid_start_ids] = False
                    self.air_time_counter[valid_start_ids] = 0.0

            # (B) RISING -> (Update Peak or FALLING)
            rising_mask = (self.hit_state == 1) & is_touching
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
            falling_edge = ((self.hit_state == 1) | (self.hit_state == 2)) & (~is_touching)
            
            if falling_edge.any():
                ids = torch.where(falling_edge)[0]
                self.hit_state[ids] = 0 # IDLEへ
                
                # 報酬計算
                self._process_hit_event(
                    ids, 
                    self.current_peak_force[ids], 
                    self.current_peak_time[ids], 
                    rewards
                )
                # バッファクリア
                self.current_peak_force[ids] = 0.0
                self.current_peak_time[ids] = 0.0

            # # (D) FALLING -> RISING (ダブルヒット: リバウンド中に再上昇)
            # # これは高度な制御だが、一旦「新しい打撃」として扱うか、
            # # あるいは「1つの打撃の続き」とするか。
            # # ここではシンプルに「前の打撃を確定させて、新しい打撃を開始」する
            # re_rising = state_falling & is_above
            # if re_rising.any():
            #     ids = torch.where(re_rising)[0]
            #     curr_f = f_val[ids]
            #     peak_f = self.current_peak_force[ids]
                
            #     # ピークを超えて再上昇した場合のみ処理
            #     # (ノイズで微増しただけなら無視したいが、簡易的に上昇なら即検知)
            #     rising_again = (curr_f > peak_f) # 単純比較
            #     ra_ids = ids[rising_again]
                
            #     if len(ra_ids) > 0:
            #         # 前の山を確定させる
            #         self._process_hit_event(
            #             ra_ids, 
            #             self.current_peak_force[ra_ids], 
            #             self.current_peak_time[ra_ids], 
            #             rewards
            #         )
            #         # 新しい山の開始
            #         self.hit_state[ra_ids] = 1 # RISING
            #         self.current_peak_force[ra_ids] = curr_f[rising_again]
            #         self.current_peak_time[ra_ids] = t_val[ra_ids]

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