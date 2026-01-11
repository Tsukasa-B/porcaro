# rhythm_generator.py
import torch
import numpy as np

class RhythmGenerator:
    """
    エピソードごとにターゲット軌道（譜面）を生成するクラス。
    
    【特徴】
    1. 音楽的なBPM選定: 60, 70, ..., 160 のように10刻みの整数値から選択。
    2. カリキュラム学習: set_difficulty() で難易度（BPM範囲、連打率）を調整可能。
    3. ベンチマーク評価: set_test_pattern() で性能評価用の固定パターンを出力。
    """
    def __init__(self, num_envs, device, dt, max_episode_length, 
                 bpm_range=(60, 160), prob_rest=0.2, prob_double=0.3):
        
        self.num_envs = num_envs
        self.device = device
        self.dt = dt
        self.max_steps = max_episode_length
        
        # --- 基本設定 ---
        # 学習時のBPM範囲 (10刻みで生成)
        self.base_bpm_min = bpm_range[0]
        self.base_bpm_max = bpm_range[1]
        
        # 基本確率 (難易度0.5想定)
        self.base_prob_rest = prob_rest
        self.base_prob_double = prob_double
        
        # --- 状態管理 ---
        self.difficulty = 0.5  # 0.0(Easy) ~ 1.0(Hard)
        self.test_mode = False # Trueなら固定パターンのみ生成
        self.test_pattern_type = None

        # --- データバッファ ---
        # ターゲット軌道: [num_envs, max_steps]
        self.target_trajectories = torch.zeros(
            (num_envs, max_episode_length), device=device, dtype=torch.float32
        )
        # 現在のBPM: [num_envs] (観測に使用)
        self.current_bpms = torch.zeros(num_envs, device=device, dtype=torch.float32)

    def set_difficulty(self, level: float):
        """
        学習進度に合わせて難易度を調整する (Curriculum Learning)。
        Args:
            level (float): 0.0 (Easy) ~ 1.0 (Hard)
        """
        self.difficulty = np.clip(level, 0.0, 1.0)

    def set_test_pattern(self, pattern_type: str = None):
        """
        評価・推論用に固定パターンモードへ切り替える。
        Args:
            pattern_type (str): "double_stroke_160", "single_stroke_120" など
        """
        if pattern_type:
            self.test_mode = True
            self.test_pattern_type = pattern_type
        else:
            self.test_mode = False

    def reset(self, env_ids):
        """指定された環境IDの譜面を再生成する"""
        if len(env_ids) == 0:
            return

        # CPUで計算して最後にTensor転送する
        indices = env_ids.view(-1).to(dtype=torch.long, device="cpu").numpy()
        
        for idx in indices:
            # モード分岐
            if self.test_mode:
                # A. テスト（評価）モード: 固定パターン
                bpm, trajectory = self._generate_test_pattern(self.test_pattern_type)
            else:
                # B. 学習（ランダム）モード: 10刻みBPM + ランダム譜面
                bpm, trajectory = self._generate_random_pattern()

            # 結果をバッファに格納
            self.current_bpms[idx] = bpm
            
            # Tensorへの転送（長さ合わせ）
            traj_len = len(trajectory)
            if traj_len >= self.max_steps:
                # 長すぎる場合はカット
                self.target_trajectories[idx] = torch.tensor(trajectory[:self.max_steps], device=self.device)
            else:
                # 短い場合はゼロ埋め
                padded = np.zeros(self.max_steps, dtype=np.float32)
                padded[:traj_len] = trajectory
                self.target_trajectories[idx] = torch.tensor(padded, device=self.device)

    def _generate_random_pattern(self):
        """学習用: 10刻みのBPMから選択し、ランダム譜面を生成"""
        
        # 1. BPMの決定 (10刻み)
        # np.arange(start, stop, step) -> stopは含まないので +10
        possible_bpms = np.arange(self.base_bpm_min, self.base_bpm_max + 10, 10)
        # 範囲外を除外
        possible_bpms = possible_bpms[possible_bpms <= self.base_bpm_max]
        
        # ランダムに1つ選ぶ (例: 60.0, 70.0, ..., 160.0)
        bpm = float(np.random.choice(possible_bpms))
        
        # 2. パラメータ設定 (難易度依存)
        # 難易度が上がると -> 休符が減り、連打が増える
        prob_rest = self.base_prob_rest * (1.0 - self.difficulty * 0.6) 
        prob_double = self.base_prob_double * (0.5 + self.difficulty * 1.0)

        # 3. 譜面生成ループ
        trajectory = np.zeros(self.max_steps, dtype=np.float32)
        beat_interval = 60.0 / bpm
        steps_per_beat = int(beat_interval / self.dt)
        if steps_per_beat < 1: steps_per_beat = 1

        current_step = 0
        while current_step < self.max_steps:
            rand_val = np.random.rand()
            
            if rand_val < prob_rest:
                # --- 休符 (Rest) ---
                current_step += steps_per_beat
                
            elif rand_val < (prob_rest + prob_double):
                # --- ダブルストローク (Double / 16分音符2連打) ---
                step_1 = current_step
                step_2 = current_step + int(steps_per_beat * 0.25) # 0.25拍後 = 16分裏
                
                self._add_hit(trajectory, step_1)
                self._add_hit(trajectory, step_2)
                
                current_step += steps_per_beat
            else:
                # --- シングルストローク (Single) ---
                self._add_hit(trajectory, current_step)
                current_step += steps_per_beat
                
        return bpm, trajectory

    def _generate_test_pattern(self, pattern_type):
        """評価用: IROS 2026 ベンチマークパターン生成"""
        trajectory = np.zeros(self.max_steps, dtype=np.float32)
        
        if pattern_type == "double_stroke_160":
            # === 限界性能テスト: BPM 160 ダブルストローク ===
            # パターン: [タタン] (休) (休) (休) ... 
            # 片手でのダブルストローク性能を見るため、断続的に連打させる
            
            bpm = 160.0
            beat_interval = 60.0 / bpm # 0.375 sec
            steps_per_beat = int(beat_interval / self.dt)
            steps_16th = int(steps_per_beat * 0.25)
            
            # 開始余白: 0.5秒
            current_step = int(0.5 / self.dt)
            
            while current_step < self.max_steps - steps_per_beat:
                # 1. ダブル (タタン)
                self._add_hit(trajectory, current_step)
                self._add_hit(trajectory, current_step + steps_16th)
                
                # 2. 次の連打まで間隔を空ける (例: 2拍休み)
                # これにより「連打後の回復」と「次の連打への準備」が見える
                current_step += steps_per_beat * 2
                
        else:
            # デフォルト: BPM 120 シングル連打 (基礎確認)
            bpm = 120.0
            beat_interval = 60.0 / bpm
            steps_per_beat = int(beat_interval / self.dt)
            for s in range(0, self.max_steps, steps_per_beat):
                self._add_hit(trajectory, s)
                
        return bpm, trajectory

    def _add_hit(self, trajectory, center_step, force=20.0, width_sec=0.05):
        """ガウス形状の打撃波形を加算 (Max合成)"""
        if center_step >= len(trajectory): return
        
        width_steps = int(width_sec / self.dt)
        start = max(0, center_step - width_steps)
        end = min(len(trajectory), center_step + width_steps)
        
        for i in range(start, end):
            t_diff = (i - center_step) * self.dt
            # ガウス関数: exp(-0.5 * (x / sigma)^2)
            # width_secを 2*sigma 相当とする
            val = force * np.exp(-0.5 * (t_diff / (width_sec/2))**2)
            trajectory[i] = max(trajectory[i], val)

    def get_lookahead(self, current_time_step_indices, horizon_steps):
        """現在時刻から horizon_steps 先までのターゲット配列を取得"""
        batch_size = len(current_time_step_indices)
        obs = torch.zeros((batch_size, horizon_steps), device=self.device)
        
        for i in range(batch_size):
            curr = current_time_step_indices[i].item()
            end = min(curr + horizon_steps, self.max_steps)
            valid_len = end - curr
            
            if valid_len > 0:
                obs[i, :valid_len] = self.target_trajectories[i, curr:end]
        return obs

    def get_current_target(self, current_time_step_indices):
        """現在のターゲット値を取得 (報酬計算用)"""
        indices = current_time_step_indices.unsqueeze(1).clamp(max=self.max_steps-1)
        vals = torch.gather(self.target_trajectories, 1, indices)
        return vals.squeeze(1)