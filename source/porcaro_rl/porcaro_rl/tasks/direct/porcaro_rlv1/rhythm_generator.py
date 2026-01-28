# rhythm_generator.py
import torch
import torch.nn.functional as F
import numpy as np

class RhythmGenerator:
    """
    エピソードごとにターゲット軌道（譜面）を生成するクラス。
    
    【GPU完全最適化版】
    - Pythonのforループを完全に撤廃。
    - 波形生成に「1次元畳み込み (Conv1d)」を使用し、全環境分を一括計算。
    """
    def __init__(self, num_envs, device, dt, max_episode_length, 
                 bpm_range=(60, 160), prob_rest=0.2, prob_double=0.3,target_force=20.0):
        
        self.num_envs = num_envs
        self.device = device
        self.dt = dt
        self.max_steps = max_episode_length
        self.target_peak_force = target_force
        
        # --- 基本設定 ---
        self.base_bpm_min = bpm_range[0]
        self.base_bpm_max = bpm_range[1]
        self.base_prob_rest = prob_rest
        self.base_prob_double = prob_double
        
        # --- 状態管理 ---
        self.difficulty = 0.5
        self.test_mode = False
        self.test_pattern_type = None

        # --- データバッファ ---
        self.target_trajectories = torch.zeros(
            (num_envs, max_episode_length), device=device, dtype=torch.float32
        )
        self.current_bpms = torch.zeros(num_envs, device=device, dtype=torch.float32)
        
        # --- 高速化のための畳み込みカーネル作成 (ガウス波形) ---
        # width_sec=0.05 (約50ms) の波形をカーネルとして事前計算
        width_sec = 0.05
        sigma = width_sec / 2.0
        # カーネルの半径 (3シグマ程度カバーすれば十分)
        kernel_radius = int(width_sec / dt) 
        t_vals = torch.arange(-kernel_radius, kernel_radius + 1, device=device, dtype=torch.float32) * dt
        
        # ガウス関数
        kernel = self.target_peak_force * torch.exp(-0.5 * (t_vals / sigma) ** 2)
        # Conv1d用に形状を整える: [Out_channels, In_channels, Kernel_size] -> [1, 1, K]
        self.kernel = kernel.view(1, 1, -1)
        self.kernel_padding = kernel_radius # sameパディング用

    def set_difficulty(self, level: float):
        self.difficulty = np.clip(level, 0.0, 1.0)

    def set_test_pattern(self, pattern_type: str = None):
        if pattern_type:
            self.test_mode = True
            self.test_pattern_type = pattern_type
        else:
            self.test_mode = False

    def reset(self, env_ids):
        """
        指定された環境IDの譜面を再生成する。
        GPU並列演算（Conv1d）により、ループなしで一瞬で計算完了させる。
        """
        num_reset = len(env_ids)
        if num_reset == 0:
            return

        # ==========================================
        # 1. BPMの決定 (GPU並列)
        # ==========================================
        if self.test_mode:
            # テストモードは固定
            target_bpm = 160.0 if "160" in self.test_pattern_type else 120.0
            bpms = torch.full((num_reset,), target_bpm, device=self.device)
        else:
            # ランダムモード: 10刻みで選択
            # 範囲内のステップ数: (160 - 60) / 10 + 1 = 11通り
            num_choices = int((self.base_bpm_max - self.base_bpm_min) / 10) + 1
            choices = torch.randint(0, num_choices, (num_reset,), device=self.device)
            bpms = self.base_bpm_min + (choices.float() * 10.0)

        self.current_bpms[env_ids] = bpms

        # ==========================================
        # 2. スパイク信号の生成 (GPU並列)
        # ==========================================
        # 叩く瞬間に「1.0」が立つスパースな行列を作る
        spikes = torch.zeros((num_reset, self.max_steps), device=self.device)
        
        # 1拍あたりのステップ数 [num_reset]
        steps_per_beat = (60.0 / bpms / self.dt).long()
        
        # 必要なビート数 (最大BPMでも足りるように少し多めに回す)
        max_beats = int((self.max_steps * self.dt) / (60.0 / self.base_bpm_max)) + 5

        # 確率パラメータ (難易度依存)
        p_rest = self.base_prob_rest * (1.0 - self.difficulty * 0.6)
        p_double = self.base_prob_double * (0.5 + self.difficulty * 1.0)

        # 全環境並列でビートを埋めていくループ (ループ回数は環境数によらず30回程度)
        for k in range(max_beats):
            # 各環境における k番目の拍のステップ位置
            beat_indices = k * steps_per_beat
            
            # まだエピソード範囲内の環境のみ処理
            valid_mask = beat_indices < self.max_steps
            if not valid_mask.any():
                break

            # 乱数生成
            rand_vals = torch.rand(num_reset, device=self.device)
            
            if self.test_mode:
                # テスト用固定ロジック (ダブルストローク例)
                if "double" in self.test_pattern_type:
                    is_rest = torch.zeros_like(rand_vals, dtype=torch.bool)
                    is_double = torch.ones_like(rand_vals, dtype=torch.bool)
                else:
                    is_rest = torch.zeros_like(rand_vals, dtype=torch.bool)
                    is_double = torch.zeros_like(rand_vals, dtype=torch.bool)
            else:
                # 学習用ランダムロジック
                is_rest = rand_vals < p_rest
                is_double = (rand_vals >= p_rest) & (rand_vals < (p_rest + p_double))
                
            # --- シングルストローク ---
            mask_s = valid_mask & (~is_rest) & (~is_double)
            if mask_s.any():
                target_envs = torch.where(mask_s)[0]
                target_steps = beat_indices[mask_s]
                spikes[target_envs, target_steps] = 1.0

            # --- ダブルストローク ---
            mask_d = valid_mask & is_double
            if mask_d.any():
                target_envs = torch.where(mask_d)[0]
                idx1 = beat_indices[mask_d]
                
                # 1打目
                spikes[target_envs, idx1] = 1.0
                
                # 2打目 (0.25拍後)
                offset = (steps_per_beat[mask_d].float() * 0.25).long()
                idx2 = idx1 + offset
                
                # 2打目が範囲内かチェック
                valid_2 = idx2 < self.max_steps
                if valid_2.any():
                    safe_idx2 = idx2.clamp(max=self.max_steps-1)
                    spikes[target_envs, safe_idx2] = 1.0

        # ==========================================
        # 3. 畳み込みによる波形生成 (超高速)
        # ==========================================
        # spikes: [Batch, Length] -> [Batch, 1, Length]
        spikes_reshaped = spikes.unsqueeze(1)
        
        # ガウスカーネルと畳み込み (GPU演算)
        traj = F.conv1d(spikes_reshaped, self.kernel, padding=self.kernel_padding)
        
        # 形状調整 [Batch, Length]
        traj = traj.squeeze(1) 
        if traj.shape[1] > self.max_steps:
            traj = traj[:, :self.max_steps]
        elif traj.shape[1] < self.max_steps:
            traj = F.pad(traj, (0, self.max_steps - traj.shape[1]))

        # 結果をバッファに格納
        self.target_trajectories[env_ids] = traj

    def get_lookahead(self, current_time_step_indices, horizon_steps):
        """GPU並列で先読みデータを取得"""
        offsets = torch.arange(horizon_steps, device=self.device)
        indices = current_time_step_indices.unsqueeze(1) + offsets.unsqueeze(0)
        valid_mask = indices < self.max_steps
        safe_indices = indices.clamp(max=self.max_steps - 1)
        vals = torch.gather(self.target_trajectories, 1, safe_indices)
        return vals * valid_mask.float()

    def get_current_target(self, current_time_step_indices):
        """現在のターゲット値を取得"""
        indices = current_time_step_indices.unsqueeze(1).clamp(max=self.max_steps-1)
        return torch.gather(self.target_trajectories, 1, indices).squeeze(1)