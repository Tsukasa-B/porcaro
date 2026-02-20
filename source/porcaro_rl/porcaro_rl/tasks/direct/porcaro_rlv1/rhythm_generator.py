# source/porcaro_rl/porcaro_rl/tasks/direct/porcaro_rlv1/rhythm_generator.py
import torch
import torch.nn.functional as F
import numpy as np

class RhythmGenerator:
    """
    強化学習および実機検証用の統合リズム生成クラス。
    """
    def __init__(self, num_envs, device, dt, max_episode_length, 
                 bpm_range=None, target_force=0.0):
        
        self.num_envs = num_envs
        self.device = device
        self.dt = dt
        self.max_steps = max_episode_length
        self.target_peak_force = target_force
        
        # --- BPM設定 ---
        self.bpm_options = torch.tensor([60.0, 80.0, 100.0, 120.0, 140.0, 160.0], device=device)

        # --- カリキュラム設定 ---
        self.curriculum_level = 0
        self.max_bpm_idx_per_level = [1, 3, 5]

        # --- ルーディメンツ定義 ---
        self.rudiments = {
            "single_4":  [0, 8],
            "single_8":  [0, 4, 8, 12],
            "double":    [0, 1, 4, 5, 8, 9, 12, 13],
            "rest":      []
        }
        self.pattern_keys = ["single_4", "single_8", "double", "rest"]

        # --- 内部状態 ---
        self.current_bpms = torch.zeros(num_envs, device=device, dtype=torch.float32)
        # ★追加: 現在のパターンID(0~3)を記憶するバッファ
        self.current_pattern_idxs = torch.zeros(num_envs, device=device, dtype=torch.long)
        
        self.target_trajectories = torch.zeros(
            (num_envs, max_episode_length), device=device, dtype=torch.float32
        )
        
        # --- テスト/検証モード設定 ---
        self.test_mode = False
        self.test_bpm = 120.0
        self.test_pattern = "single_8"

        # --- 波形生成用カーネル ---
        width_sec = 0.035
        sigma = width_sec / 2.0
        kernel_radius = int(width_sec / dt) 
        t_vals = torch.arange(-kernel_radius, kernel_radius + 1, device=device, dtype=torch.float32) * dt
        
        kernel = self.target_peak_force * torch.exp(-0.5 * (t_vals / sigma) ** 2)
        self.kernel = kernel.view(1, 1, -1)
        self.kernel_padding = kernel_radius

    def set_test_mode(self, enabled: bool, bpm: float = 120.0, pattern: str = "single_4"):
        self.test_mode = enabled
        self.test_bpm = bpm
        self.test_pattern = pattern
        if pattern not in self.rudiments and pattern != "random":
             if "double" in pattern: self.test_pattern = "double"
             elif "single" in pattern: self.test_pattern = "single_8"
             else: self.test_pattern = "single_4"

    def set_curriculum_level(self, level: int):
        self.curriculum_level = min(level, 2)

    def reset(self, env_ids):
        num_reset = len(env_ids)
        if num_reset == 0:
            return

        # ==========================================
        # 1. BPMとパターンの決定 (エピソード全体で固定)
        # ==========================================
        if self.test_mode:
            bpms = torch.full((num_reset,), self.test_bpm, device=self.device)
            # パターンの決定
            pat_idx = self.pattern_keys.index(self.test_pattern) if self.test_pattern in self.pattern_keys else 0
            ep_pat_idxs = torch.full((num_reset,), pat_idx, device=self.device, dtype=torch.long)
            ep_patterns = [self.pattern_keys[pat_idx]] * num_reset
        else:
            # BPM抽選
            max_idx = self.max_bpm_idx_per_level[self.curriculum_level]
            idxs = torch.randint(0, max_idx + 1, (num_reset,), device=self.device)
            bpms = self.bpm_options[idxs]

            # パターン抽選 (エピソード開始時に1回だけ)
            if self.curriculum_level == 0:
                probs = torch.tensor([0.4, 0.5, 0.0, 0.1], device=self.device)
            elif self.curriculum_level == 1:
                probs = torch.tensor([0.2, 0.4, 0.3, 0.1], device=self.device)
            else:
                probs = torch.tensor([0.2, 0.3, 0.4, 0.1], device=self.device)
            
            ep_pat_idxs = torch.multinomial(probs, num_reset, replacement=True)
            ep_patterns = [self.pattern_keys[i] for i in ep_pat_idxs.tolist()]

        self.current_bpms[env_ids] = bpms
        # ★追加: 選ばれたパターンIDを保存
        self.current_pattern_idxs[env_ids] = ep_pat_idxs

        # ==========================================
        # 2. グリッド計算
        # ==========================================
        steps_per_16th = (15.0 / bpms / self.dt).long()
        spikes = torch.zeros((num_reset, self.max_steps), device=self.device)

        # ==========================================
        # 3. 4小節分のパターン生成
        # ==========================================
        for bar_idx in range(4):
            bar_start_steps = bar_idx * 16 * steps_per_16th 
            
            if bar_idx == 0:
                # 1小節目は休符
                selected_patterns = ["rest"] * num_reset
            else:
                # ★修正: エピソード固定のパターンを使用する
                selected_patterns = ep_patterns

            unique_patterns = set(selected_patterns)
            
            for pat_name in unique_patterns:
                env_mask_list = [p == pat_name for p in selected_patterns]
                env_mask = torch.tensor(env_mask_list, device=self.device, dtype=torch.bool)
                
                if not env_mask.any():
                    continue
                
                offsets = self.rudiments.get(pat_name, [])
                if not offsets:
                    continue
                
                target_local_indices = torch.where(env_mask)[0]
                base = bar_start_steps[target_local_indices].unsqueeze(1)
                step_unit = steps_per_16th[target_local_indices].unsqueeze(1)
                off_tensor = torch.tensor(offsets, device=self.device).unsqueeze(0)
                hit_times = base + off_tensor * step_unit
                valid_hits = hit_times < self.max_steps
                
                for i in range(len(target_local_indices)):
                    local_env_idx = target_local_indices[i]
                    valid_times = hit_times[i][valid_hits[i]]
                    spikes[local_env_idx, valid_times] = 1.0

        # ==========================================
        # 4. 畳み込みによる波形生成
        # ==========================================
        spikes_reshaped = spikes.unsqueeze(1)
        traj = F.conv1d(spikes_reshaped, self.kernel, padding=self.kernel_padding)
        traj = traj.squeeze(1)
        
        if traj.shape[1] > self.max_steps:
            traj = traj[:, :self.max_steps]
        elif traj.shape[1] < self.max_steps:
            traj = F.pad(traj, (0, self.max_steps - traj.shape[1]))

        self.target_trajectories[env_ids] = traj

    def get_lookahead(self, current_time_step_indices, horizon_steps):
        offsets = torch.arange(horizon_steps, device=self.device)
        indices = current_time_step_indices.unsqueeze(1) + offsets.unsqueeze(0)
        safe_indices = indices.clamp(max=self.max_steps - 1)
        vals = torch.gather(self.target_trajectories, 1, safe_indices)
        valid_mask = (indices < self.max_steps).float()
        return vals * valid_mask

    def get_current_target(self, current_time_step_indices):
        indices = current_time_step_indices.unsqueeze(1).clamp(max=self.max_steps-1)
        return torch.gather(self.target_trajectories, 1, indices).squeeze(1)