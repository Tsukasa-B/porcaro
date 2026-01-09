# rhythm_generator.py
import torch
import numpy as np

class RhythmGenerator:
    """
    エピソードごとにランダムな譜面（ターゲット軌道）を生成し、
    エージェントに先読み情報（Look-ahead Buffer）を提供するクラス。
    """
    def __init__(self, num_envs, device, dt, max_episode_length, 
                 bpm_range=(60, 180), prob_rest=0.2, prob_double=0.3):
        
        self.num_envs = num_envs
        self.device = device
        self.dt = dt
        self.max_steps = max_episode_length
        self.bpm_range = bpm_range
        self.prob_rest = prob_rest     # 休符の確率
        self.prob_double = prob_double # ダブルストローク(16分)の確率
        
        # ターゲット軌道バッファ (num_envs, max_steps)
        # 値は [0.0 ~ 20.0] (N)
        
        # --- ▼▼▼ 修正箇所 ▼▼▼ ---
        # 誤: (num_envs, max_steps) -> NameError
        # 正: (num_envs, max_episode_length) または (num_envs, self.max_steps)
        self.target_trajectories = torch.zeros(
            (num_envs, max_episode_length), device=device, dtype=torch.float32
        )
        # ------------------------
        
        # 各環境の現在のBPM（観測に含めるため）
        self.current_bpms = torch.zeros(num_envs, device=device, dtype=torch.float32)

    def reset(self, env_ids):
        """指定された環境IDの譜面を再生成する"""
        if len(env_ids) == 0:
            return

        # ★修正: 型と形状を強制的に整える
        # .view(-1) で1次元化し、.long() で整数化してから CPU へ
        indices = env_ids.view(-1).to(dtype=torch.long, device="cpu").numpy()
        
        # current_bpms と target_trajectories を書き換える
        # (スライシング代入するために一時的にCPUで作るか、Tensor操作するか)
        # ここではループで処理して最後にまとめて転送する方法をとる
        
        for idx in indices:
            # 1. BPMをランダム決定
            bpm = np.random.uniform(*self.bpm_range)
            self.current_bpms[idx] = bpm
            
            # 1拍の時間 [sec]
            beat_interval = 60.0 / bpm
            steps_per_beat = int(beat_interval / self.dt)
            # ゼロ除算防止
            if steps_per_beat < 1: steps_per_beat = 1
            
            trajectory = np.zeros(self.max_steps, dtype=np.float32)
            
            # 2. 譜面生成ループ
            current_step = 0
            while current_step < self.max_steps:
                # ランダムにパターン決定
                rand_val = np.random.rand()
                
                if rand_val < self.prob_rest:
                    # --- 休符 (Rest) ---
                    # 1拍休み
                    current_step += steps_per_beat
                    
                elif rand_val < (self.prob_rest + self.prob_double):
                    # --- ダブルストローク (Double / 16th notes) ---
                    # 1拍を4分割した 1つ目と3つ目、あるいは 1つ目と2つ目(変則) に配置
                    # ここではシンプルに「タタン」 (16分2連打)
                    step_1 = current_step
                    step_2 = current_step + int(steps_per_beat * 0.25) # 0.25拍後
                    
                    self._add_hit(trajectory, step_1)
                    self._add_hit(trajectory, step_2)
                    
                    current_step += steps_per_beat
                    
                else:
                    # --- シングルストローク (Single / Quarter note) ---
                    self._add_hit(trajectory, current_step)
                    current_step += steps_per_beat

            # Tensorに転送
            self.target_trajectories[idx] = torch.tensor(trajectory, device=self.device)

    def _add_hit(self, trajectory, center_step, force=20.0, width_sec=0.05):
        """ガウス形状のターゲット波形を加算する"""
        if center_step >= len(trajectory):
            return
            
        width_steps = int(width_sec / self.dt)
        start = max(0, center_step - width_steps)
        end = min(len(trajectory), center_step + width_steps)
        
        # ループではなくスライスで入れたほうが高速だが、Numpyならループでも耐えられる
        for i in range(start, end):
            # ガウス分布形状
            t_diff = (i - center_step) * self.dt
            val = force * np.exp(-0.5 * (t_diff / (width_sec/2))**2)
            trajectory[i] = max(trajectory[i], val) # Max合成

    def get_lookahead(self, current_time_step_indices, horizon_steps):
        """
        現在時刻から horizon_steps 先までのターゲット配列を取得する
        Returns:
            observation: (num_envs, horizon_steps)
        """
        batch_size = len(current_time_step_indices)
        obs = torch.zeros((batch_size, horizon_steps), device=self.device)
        
        for i in range(batch_size):
            curr = current_time_step_indices[i].item()
            # エピソード長を超えないようにクリップ
            end = min(curr + horizon_steps, self.max_steps)
            valid_len = end - curr
            
            if valid_len > 0:
                obs[i, :valid_len] = self.target_trajectories[i, curr:end]
                
        return obs

    def get_current_target(self, current_time_step_indices):
        """現在の瞬間のターゲット値を取得 (報酬計算用)"""
        # gatherを使って一括取得
        # target_trajectories: (num_envs, max_steps)
        # indices: (num_envs, 1)
        indices = current_time_step_indices.unsqueeze(1).clamp(max=self.max_steps-1)
        vals = torch.gather(self.target_trajectories, 1, indices)
        return vals.squeeze(1)