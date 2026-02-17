# rhythm_generator.py
import torch
import torch.nn.functional as F
import numpy as np

class RhythmGenerator:
    """
    強化学習および実機検証用の統合リズム生成クラス。
    
    【主な機能】
    1. 4小節構造 (4-Bar Structure) に基づくルーディメンツ生成
    2. GPU並列演算 (Conv1d) による高速な波形生成
    3. 学習用ランダムモードと検証用固定モードのシームレスな切り替え
    
    【改良点 (Sim-to-Real戦略に基づく)】
    - BPMを連続値ではなく、キリの良い離散値 (60, 80...160) から選択
    - パターンを基礎的な「シングル」「ダブル」に限定し、片手での確実な習得を目指す
    【カリキュラム学習の実装】
    - Lv0: BPM60-80, Single打ちのみ
    - Lv1: BPM60-120, Double打ち解禁
    - Lv2: BPM60-160, 全パターン (高速対応)
    """
    def __init__(self, num_envs, device, dt, max_episode_length, 
                 bpm_range=None, target_force=0.0):
        
        self.num_envs = num_envs
        self.device = device
        self.dt = dt
        self.max_steps = max_episode_length
        self.target_peak_force = target_force
        
        # --- BPM設定 ---
        # 実機検証と比較しやすいよう、20刻みの離散値を採用
        self.bpm_options = torch.tensor([60.0, 80.0, 100.0, 120.0, 140.0, 160.0], device=device)

        # --- カリキュラム設定 ---
        self.curriculum_level = 0
        # レベルごとの許可される最大BPMインデックス
        self.max_bpm_idx_per_level = [1, 3, 5]


        # --- ルーディメンツ定義 (16分音符グリッド: 0~15) ---
        # 1小節(4拍) = 16個の16分音符スロット
        # 片手タスクとして物理的に無理がなく、かつ重要な基礎動作のみに絞る
        self.rudiments = {
            # 表打ち (4分音符) - 最も基礎的な動作
            "single_4":  [0, 8], # 最大約600
            # 8ビート (8分音符) - 連続動作の基本
            "single_8":  [0, 4, 8, 12], # 最大約1200
            # ダブルストローク (RRLL...) - バネ性を活かしたリバウンド動作の学習用
            # 片手で16分音符2つを叩く [0, 1] ...
            "double":    [0, 1, 4, 5, 8, 9, 12, 13], #最大約2400
            # 休符 - 待機姿勢と脱力の学習用
            "rest":      []
        }
        self.pattern_keys = ["single_4", "single_8", "double", "rest"]

        # --- 内部状態 ---
        self.current_bpms = torch.zeros(num_envs, device=device, dtype=torch.float32)
        self.target_trajectories = torch.zeros(
            (num_envs, max_episode_length), device=device, dtype=torch.float32
        )
        
        # --- テスト/検証モード設定 ---
        self.test_mode = False
        self.test_bpm = 120.0
        self.test_pattern = "single_8"

        # --- 波形生成用カーネル (ガウス関数) ---
        # width_sec=0.01 (約10ms)
        width_sec = 0.035
        sigma = width_sec / 2.0
        kernel_radius = int(width_sec / dt) 
        t_vals = torch.arange(-kernel_radius, kernel_radius + 1, device=device, dtype=torch.float32) * dt
        
        # 頂点が target_force になるように設定
        kernel = self.target_peak_force * torch.exp(-0.5 * (t_vals / sigma) ** 2)
        self.kernel = kernel.view(1, 1, -1)
        self.kernel_padding = kernel_radius

    def set_test_mode(self, enabled: bool, bpm: float = 120.0, pattern: str = "single_4"):
        """検証用モードの設定 (SimpleRhythmGeneratorの代替)"""
        self.test_mode = enabled
        self.test_bpm = bpm
        self.test_pattern = pattern
        # パターン名が辞書にない場合のフォールバック
        if pattern not in self.rudiments and pattern != "random":
             if "double" in pattern: self.test_pattern = "double"
             elif "single" in pattern: self.test_pattern = "single_8"
             else: self.test_pattern = "single_4"

    def set_curriculum_level(self, level: int):
        """難易度レベルを設定 (0:基礎 -> 2:完全ランダム)"""
        self.curriculum_level = min(level, 2)

    def reset(self, env_ids):
        """
        指定された環境IDの譜面を再生成する。
        4小節構成でパターンを生成。
        """
        num_reset = len(env_ids)
        if num_reset == 0:
            return

        # ==========================================
        # 1. BPMの決定 (カリキュラム適用)
        # ==========================================
        if self.test_mode:
            bpms = torch.full((num_reset,), self.test_bpm, device=self.device)
        else:
            # 現在のレベルに応じた上限インデックスを取得
            max_idx = self.max_bpm_idx_per_level[self.curriculum_level]
            # 0 〜 max_idx の範囲でランダム選択
            idxs = torch.randint(0, max_idx + 1, (num_reset,), device=self.device)
            bpms = self.bpm_options[idxs]

        self.current_bpms[env_ids] = bpms

        # ==========================================
        # 2. グリッド計算
        # ==========================================
        # 1拍(Beat)の時間[s] = 60 / BPM
        # 16分音符(Grid)の時間[s] = 60 / BPM / 4
        # Gridあたりのステップ数
        steps_per_16th = (15.0 / bpms / self.dt).long() # 60/4 = 15
        
        # スパイク行列の初期化
        spikes = torch.zeros((num_reset, self.max_steps), device=self.device)

        # ==========================================
        # 3. 4小節分のパターン生成
        # ==========================================
        # 4小節ループ
        for bar_idx in range(4):
            # この小節の開始ステップ位置
            # 1小節 = 16グリッド
            bar_start_steps = bar_idx * 16 * steps_per_16th # [num_reset]
            
            # --- 1小節目は常に休符 (Count-in) ---
            if bar_idx == 0:
                # ★重要: 最初の1小節は「指揮者の合図（Count-in）」として強制的に休符にする
                # これにより、ロボットは初期位置(Down)から振りかぶる時間を確保できる
                selected_patterns = ["rest"] * num_reset

            # パターンの選択
            # テストモード (2小節目以降)
            if self.test_mode:
                selected_patterns = [self.test_pattern] * num_reset
            # 学習モード (2小節目以降はランダム)
            # ランダムに選択 (Categorical分布)
            # pattern_indices: [num_reset]
            # --- 学習モード ---
            else:
                # ★ここが修正ポイント: レベルに応じて確率分布を変える
                if self.curriculum_level == 0:
                    # Lv0: Singleのみ (Double確率=0.0)
                    # [single_4, single_8, double, rest]
                    probs = torch.tensor([0.4, 0.5, 0.0, 0.1], device=self.device)
                elif self.curriculum_level == 1:
                    # Lv1: Double解禁
                    probs = torch.tensor([0.2, 0.4, 0.3, 0.1], device=self.device)
                else:
                    # Lv2: 全開
                    probs = torch.tensor([0.2, 0.3, 0.4, 0.1], device=self.device)
                
                p_indices = torch.multinomial(probs, num_reset, replacement=True)
                selected_patterns = [self.pattern_keys[i] for i in p_indices.tolist()]

            # 環境ごとにスパイクを配置
            unique_patterns = set(selected_patterns)
            
            for pat_name in unique_patterns:
                # このパターンを選択した環境のマスク
                # (文字列比較なのでリスト内包表記でマスク作成)
                env_mask_list = [p == pat_name for p in selected_patterns]
                env_mask = torch.tensor(env_mask_list, device=self.device, dtype=torch.bool)
                
                if not env_mask.any():
                    continue
                
                # パターンの打撃タイミング(0~15)を取得
                offsets = self.rudiments.get(pat_name, [])
                if not offsets:
                    continue
                
                # 対象環境のインデックス
                target_local_indices = torch.where(env_mask)[0]
                
                # タイミング計算 [Env, Offsets]
                # base: [Env, 1], offset_steps: [Env, 1]
                base = bar_start_steps[target_local_indices].unsqueeze(1)
                step_unit = steps_per_16th[target_local_indices].unsqueeze(1)
                
                # off_tensor: [1, n_hits]
                off_tensor = torch.tensor(offsets, device=self.device).unsqueeze(0)
                
                # hit_times: [Env, n_hits]
                hit_times = base + off_tensor * step_unit
                
                # 範囲外チェックと書き込み
                # max_stepsを超えないものだけ書き込む
                valid_hits = hit_times < self.max_steps
                
                # gather/scatter用にフラット化して書き込み
                # ※ここでは単純に代入ループで処理（Pythonループは回数少なければ許容）
                # Advanced: scatterを使うと更に速いが、可読性重視でループ
                for i in range(len(target_local_indices)):
                    local_env_idx = target_local_indices[i]
                    valid_times = hit_times[i][valid_hits[i]]
                    spikes[local_env_idx, valid_times] = 1.0

        # ==========================================
        # 4. 畳み込みによる波形生成
        # ==========================================
        spikes_reshaped = spikes.unsqueeze(1) # [Batch, 1, Length]
        traj = F.conv1d(spikes_reshaped, self.kernel, padding=self.kernel_padding)
        traj = traj.squeeze(1)
        
        # サイズ調整
        if traj.shape[1] > self.max_steps:
            traj = traj[:, :self.max_steps]
        elif traj.shape[1] < self.max_steps:
            traj = F.pad(traj, (0, self.max_steps - traj.shape[1]))

        self.target_trajectories[env_ids] = traj

    def get_lookahead(self, current_time_step_indices, horizon_steps):
        """GPU並列で先読みデータを取得"""
        offsets = torch.arange(horizon_steps, device=self.device)
        indices = current_time_step_indices.unsqueeze(1) + offsets.unsqueeze(0)
        
        # インデックス制限
        safe_indices = indices.clamp(max=self.max_steps - 1)
        
        # データを取得
        vals = torch.gather(self.target_trajectories, 1, safe_indices)
        
        # エピソード外(max_steps超え)の部分は0にするマスク
        valid_mask = (indices < self.max_steps).float()
        
        return vals * valid_mask

    def get_current_target(self, current_time_step_indices):
        """現在のターゲット値を取得"""
        indices = current_time_step_indices.unsqueeze(1).clamp(max=self.max_steps-1)
        return torch.gather(self.target_trajectories, 1, indices).squeeze(1)