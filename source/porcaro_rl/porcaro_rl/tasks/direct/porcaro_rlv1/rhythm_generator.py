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
       (SimpleRhythmGeneratorの機能を完全に包含)
    """
    def __init__(self, num_envs, device, dt, max_episode_length, 
                 bpm_range=(60, 160), target_force=20.0):
        
        self.num_envs = num_envs
        self.device = device
        self.dt = dt
        self.max_steps = max_episode_length
        self.target_peak_force = target_force
        
        # --- BPM設定 ---
        self.bpm_min = bpm_range[0]
        self.bpm_max = bpm_range[1]
        
        # --- 内部状態 ---
        self.current_bpms = torch.zeros(num_envs, device=device, dtype=torch.float32)
        self.target_trajectories = torch.zeros(
            (num_envs, max_episode_length), device=device, dtype=torch.float32
        )
        
        # --- テスト/検証モード設定 ---
        self.test_mode = False
        self.test_bpm = 120.0
        self.test_pattern = "single"

        # --- ルーディメンツ定義 (16分音符グリッド: 0~15) ---
        # 1小節(4拍) = 16個の16分音符スロット
        self.rudiments = {
            # 表打ち (4分音符)
            "single_4":  [0, 4, 8, 12],
            # 8ビート (8分音符)
            "single_8":  [0, 2, 4, 6, 8, 10, 12, 14],
            # ダブルストローク (RRLL...)
            "double":    [0, 1, 4, 5, 8, 9, 12, 13],
            # パラディドル (RLRR LRLL)
            "paradiddle":[0, 2, 4, 5, 8, 10, 12, 13],
            # シンコペーション (裏拍)
            "upbeat":    [2, 6, 10, 14],
            # 3-3-2 (Clave)
            "clave":     [0, 3, 6, 8, 10, 12],
            # 全休符
            "rest":      []
        }
        
        # 学習時のパターン出現確率重み
        self.pattern_keys = list(self.rudiments.keys())
        # [single_4, single_8, double, paradiddle, upbeat, clave, rest]
        self.pattern_probs = torch.tensor([0.2, 0.3, 0.2, 0.1, 0.1, 0.05, 0.05], device=device)

        # --- 波形生成用カーネル (ガウス関数) ---
        # width_sec=0.05 (約50ms)
        width_sec = 0.05
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
             # 簡易的なマッピング
             if "double" in pattern: self.test_pattern = "double"
             elif "para" in pattern: self.test_pattern = "paradiddle"
             else: self.test_pattern = "single_4"

    def reset(self, env_ids):
        """
        指定された環境IDの譜面を再生成する。
        4小節構成でパターンを生成。
        """
        num_reset = len(env_ids)
        if num_reset == 0:
            return

        # ==========================================
        # 1. BPMの決定
        # ==========================================
        if self.test_mode:
            bpms = torch.full((num_reset,), self.test_bpm, device=self.device)
        else:
            # 60 ~ 160 の範囲でランダム
            rand_factor = torch.rand(num_reset, device=self.device)
            bpms = self.bpm_min + rand_factor * (self.bpm_max - self.bpm_min)
            # 整数に近いBPMにする（任意）
            bpms = torch.round(bpms / 5.0) * 5.0

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

            # パターンの選択
            if self.test_mode:
                # テストモードは全環境同じパターン
                selected_patterns = [self.test_pattern] * num_reset
            else:
                # ランダムに選択 (Categorical分布)
                # pattern_indices: [num_reset]
                p_indices = torch.multinomial(
                    self.pattern_probs, num_reset, replacement=True
                )
                selected_patterns = [self.pattern_keys[i] for i in p_indices.tolist()]

            # 環境ごとにスパイクを配置
            # NOTE: ここはパターン種類数(最大7種類)のループになるため高速
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