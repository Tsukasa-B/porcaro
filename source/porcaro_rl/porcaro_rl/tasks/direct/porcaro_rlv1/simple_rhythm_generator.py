# simple_rhythm_generator.py
import torch
import torch.nn.functional as F
import numpy as np

class SimpleRhythmGenerator:
    """
    Sim-to-Realデバッグ用の単純なリズム生成クラス。
    固定パターン（Rudiments）を繰り返し生成し、基礎動作の習得を確認する。
    """
    def __init__(self, num_envs, device, dt, max_episode_length, 
                 mode="single", bpm=60.0, target_force=50.0):
        
        self.num_envs = num_envs
        self.device = device
        self.dt = dt
        self.max_steps = max_episode_length
        
        # --- 設定 ---
        self.mode = mode  # "single", "double", "steady"
        self.target_bpm = bpm
        self.target_peak_force = target_force
        
        # --- データバッファ ---
        self.target_trajectories = torch.zeros(
            (num_envs, max_episode_length), device=device, dtype=torch.float32
        )
        # 固定BPMで埋めておく
        self.current_bpms = torch.full((num_envs,), bpm, device=device, dtype=torch.float32)
        
        # --- 波形生成用カーネル ---
        width_sec = 0.05
        sigma = width_sec / 2.0
        kernel_radius = int(width_sec / dt) 
        t_vals = torch.arange(-kernel_radius, kernel_radius + 1, device=device, dtype=torch.float32) * dt
        
        # ★修正: Configから受け取った力を使う (ガウスの頂点が target_force になる)
        # kernel = 20.0 * ... (削除)
        self.kernel = self.target_peak_force * torch.exp(-0.5 * (t_vals / sigma) ** 2)
        self.kernel = self.kernel.view(1, 1, -1)
        self.kernel_padding = kernel_radius

    def set_difficulty(self, level: float):
        # シンプルモードでは難易度調整は行わない（API互換性のためのダミー）
        pass

    def reset(self, env_ids):
        """
        指定された環境IDに対して固定パターンの譜面を生成する。
        """
        num_reset = len(env_ids)
        if num_reset == 0:
            return

        # スパイク行列の初期化
        spikes = torch.zeros((num_reset, self.max_steps), device=self.device)
        
        # モードに応じたロジック
        if self.mode == "single":
            # 2秒に1回叩く (Sim-to-Realでの応答確認用)
            interval_sec = 2.0
            interval_steps = int(interval_sec / self.dt)
            
            # 0.5秒後から開始
            start_step = int(0.5 / self.dt)
            
            for t in range(start_step, self.max_steps, interval_steps):
                spikes[:, t] = 1.0

        elif self.mode == "double":
            # 2連打 (タタン……休み)
            # サイクル: 2秒
            cycle_sec = 2.0
            cycle_steps = int(cycle_sec / self.dt)
            
            # 2打の間隔: 0.2秒 (速めの連打)
            double_interval = int(0.2 / self.dt)
            start_step = int(0.5 / self.dt)

            for t in range(start_step, self.max_steps, cycle_steps):
                if t < self.max_steps:
                    spikes[:, t] = 1.0
                if t + double_interval < self.max_steps:
                    spikes[:, t + double_interval] = 1.0

        elif self.mode == "steady":
            # メトロノーム (一定BPMで叩き続ける)
            beat_interval_sec = 60.0 / self.target_bpm
            beat_steps = int(beat_interval_sec / self.dt)
            
            for t in range(int(0.5/self.dt), self.max_steps, beat_steps):
                spikes[:, t] = 1.0

        # --- 畳み込みによる波形生成 ---
        spikes_reshaped = spikes.unsqueeze(1) # [Batch, 1, Length]
        traj = F.conv1d(spikes_reshaped, self.kernel, padding=self.kernel_padding)
        traj = traj.squeeze(1)
        
        # サイズ調整
        if traj.shape[1] > self.max_steps:
            traj = traj[:, :self.max_steps]
        elif traj.shape[1] < self.max_steps:
            traj = F.pad(traj, (0, self.max_steps - traj.shape[1]))

        self.target_trajectories[env_ids] = traj
        
        # BPM情報は固定値を維持
        self.current_bpms[env_ids] = self.target_bpm

    def get_lookahead(self, current_time_step_indices, horizon_steps):
        """GPU並列で先読みデータを取得 (RhythmGeneratorと同等)"""
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