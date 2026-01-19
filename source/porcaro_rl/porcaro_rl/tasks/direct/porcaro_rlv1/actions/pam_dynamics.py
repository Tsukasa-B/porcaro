from __future__ import annotations
import torch
import torch.nn as nn
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..cfg.actuator_cfg import PamDelayModelCfg, PamHysteresisModelCfg, ActuatorNetModelCfg

class PamDelayModel(nn.Module):
    """
    空気圧の伝送遅れ(可変)と応答遅れ(可変)を模擬するモデル
    """
    def __init__(self, cfg: PamDelayModelCfg, dt: float, device: str):
        super().__init__()
        self.cfg = cfg
        self.dt = dt
        self.device = device
        
        # --- LUTの登録 (Tau & Deadtime) ---
        # Tau (時定数)
        self.register_buffer('tau_p_axis', torch.tensor(
            self.cfg.tau_pressure_axis if self.cfg.tau_pressure_axis else [0.0, 1.0], 
            dtype=torch.float32, device=self.device
        ))
        self.register_buffer('tau_vals', torch.tensor(
            self.cfg.tau_values if self.cfg.tau_values else [self.cfg.time_constant]*2, 
            dtype=torch.float32, device=self.device
        ))

        # Deadtime (むだ時間)
        self.register_buffer('dead_p_axis', torch.tensor(
            self.cfg.deadtime_pressure_axis if self.cfg.deadtime_pressure_axis else [0.0, 1.0], 
            dtype=torch.float32, device=self.device
        ))
        self.register_buffer('dead_vals', torch.tensor(
            self.cfg.deadtime_values if self.cfg.deadtime_values else [self.cfg.delay_time]*2, 
            dtype=torch.float32, device=self.device
        ))

        # --- リングバッファ設定 ---
        # 最大遅延時間からバッファサイズを決定
        max_delay = self.cfg.max_delay_time
        self.buffer_len = math.ceil(max_delay / self.dt) + 2  # 余裕を持たせる
        
        self.pressure_buffer = None   # Shape: [num_envs, num_channels, buffer_len]
        self.write_ptr = 0            # 書き込み位置 (int)
        self.current_pressure = None  # フィルタ出力

    def _init_buffers(self, num_envs: int, num_channels: int):
        self.pressure_buffer = torch.zeros((num_envs, num_channels, self.buffer_len), device=self.device)
        self.current_pressure = torch.zeros((num_envs, num_channels), device=self.device)
        self.write_ptr = 0

    def reset_idx(self, env_ids: torch.Tensor):
        if self.pressure_buffer is not None:
            self.pressure_buffer[env_ids] = 0.0
            self.current_pressure[env_ids] = 0.0

    def _interp_lut(self, x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
        """汎用LUT線形補間"""
        x_clamped = x.clamp(min=xp[0], max=xp[-1])
        idx = torch.bucketize(x_clamped, xp).clamp(1, len(xp)-1)
        x0, x1 = xp[idx-1], xp[idx]
        f0, f1 = fp[idx-1], fp[idx]
        t = (x_clamped - x0) / (x1 - x0 + 1e-12)
        return f0 + t * (f1 - f0)

    def forward(self, target_pressure: torch.Tensor) -> torch.Tensor:
        # 初回初期化
        if self.pressure_buffer is None:
            self._init_buffers(target_pressure.shape[0], target_pressure.shape[1])
            
        # 1. 現在の指令圧に基づいて「遅延時間 L」と「時定数 Tau」を決定
        #    (遅延や時定数は入力圧の大きさで変わる非線形性を持つため)
        L = self._interp_lut(target_pressure, self.dead_p_axis, self.dead_vals)
        tau = self._interp_lut(target_pressure, self.tau_p_axis, self.tau_vals)
        
        # 2. リングバッファへの書き込み
        self.pressure_buffer[:, :, self.write_ptr] = target_pressure
        
        # 3. 可変遅延読み出し (Fractional Delay)
        #    遅延ステップ数 D = L / dt
        D = (L / self.dt).clamp(min=0.0, max=self.buffer_len - 2.0)
        
        #    読み出し位置 (float) = 現在位置 - 遅延ステップ
        #    バッファが循環するため modulo 計算が必要
        read_idx_float = (self.write_ptr - D) % self.buffer_len
        
        #    線形補間読み出し
        idx0 = torch.floor(read_idx_float).long()
        idx1 = (idx0 + 1) % self.buffer_len
        alpha_idx = read_idx_float - idx0
        
        #    バッチ一括取得のために gather を使う手もあるが、インデックスアクセスで十分
        #    [num_envs, num_channels]
        val0 = self.pressure_buffer[:, :, idx0].clone() # 形状注意: gatherが必要な場合は工夫要
        #    上記インデックスアクセスは全バッチ共通インデックスなら動くが、
        #    遅延 D がバッチ(圧力)ごとに異なるため、advanced indexingが必要
        
        #    --- Advanced Indexing for Batch-Varying Delay ---
        #    idx0, idx1 は [num_envs, num_channels] の形状を持つ
        #    pressure_buffer は [num_envs, num_channels, buffer_len]
        
        #    次元を合わせるために gather を使用
        val0 = self.pressure_buffer.gather(2, idx0.unsqueeze(2)).squeeze(2)
        val1 = self.pressure_buffer.gather(2, idx1.unsqueeze(2)).squeeze(2)
        
        delayed_input = (1.0 - alpha_idx) * val0 + alpha_idx * val1
        
        # 4. 1次遅れフィルタの更新 (可変Tauを使用)
        #    alpha = dt / (tau + dt)
        alpha_filter = self.dt / (tau + self.dt)
        self.current_pressure = (1.0 - alpha_filter) * self.current_pressure + alpha_filter * delayed_input
        
        # 5. ポインタ更新
        self.write_ptr = (self.write_ptr + 1) % self.buffer_len
        
        return self.current_pressure

class PamHysteresisModel(nn.Module):
    """
    Additive Hysteresis Model (Play Operator)
    圧力に対する「摩擦（一定の圧力幅）」を再現するモデル
    """
    def __init__(self, cfg: PamHysteresisModelCfg, device: str):
        super().__init__()
        self.cfg = cfg
        self.device = device
        
        # 状態変数: 前回の出力値 (y_{t-1})
        self.last_output = None

    def reset_idx(self, env_ids: torch.Tensor):
        """
        [追加] 指定された環境の状態をリセットする
        """
        if self.last_output is not None:
            self.last_output[env_ids] = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Play Operator Logic:
        y(t) = min( x(t) + r, max( x(t) - r, y(t-1) ) )
        """
        # 初回初期化: 入力値に合わせて初期化
        if self.last_output is None or self.last_output.shape != x.shape:
            self.last_output = x.clone()

        # ヒステリシス幅 (全幅) から 半幅 (r) を計算
        r = self.cfg.hysteresis_width / 2.0
        
        # Play Operator 計算
        lower_bound = x - r
        upper_bound = x + r
        
        # 前回の値を上下限でクリップする
        output = torch.max(lower_bound, torch.min(upper_bound, self.last_output))
        
        # 状態更新
        self.last_output = output.clone()
        
        return output


class ActuatorNetModel(nn.Module):
    """
    ActuatorNet (MLPによる学習モデル)
    """
    def __init__(self, cfg: ActuatorNetModelCfg, device: str):
        super().__init__()
        self.cfg = cfg
        self.device = device
        
        layers = []
        in_dim = self.cfg.input_dim
        for hidden in self.cfg.hidden_units:
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.ReLU())
            in_dim = hidden
        layers.append(nn.Linear(in_dim, self.cfg.output_dim))
        
        self.net = nn.Sequential(*layers).to(self.device)
        
        # 学習済みモデルのロード
        if self.cfg.model_path:
            import os
            if os.path.exists(self.cfg.model_path):
                state_dict = torch.load(self.cfg.model_path, map_location=self.device)
                self.net.load_state_dict(state_dict)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)