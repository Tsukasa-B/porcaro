from __future__ import annotations
import torch
import torch.nn as nn
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..cfg.actuator_cfg import PamDelayModelCfg, PamHysteresisModelCfg, ActuatorNetModelCfg

class PamDelayModel(nn.Module):
    """
    空気圧の伝送遅れと応答遅れを模擬するモデル
    """
    def __init__(self, cfg: PamDelayModelCfg, dt: float, device: str):
        super().__init__()
        self.cfg = cfg
        self.dt = dt
        self.device = device
        
        # 遅延バッファのサイズ計算
        self.buffer_size = max(1, int(self.cfg.delay_time / self.dt))
        # 1次遅れの係数 (Low Pass Filter)
        self.alpha = self.dt / (self.cfg.time_constant + self.dt)
        
        # 状態変数 [num_envs, num_channels, buffer_size]
        self.pressure_buffer = None
        self.current_pressure = None

    def _init_buffers(self, num_envs: int, num_channels: int):
        self.pressure_buffer = torch.zeros((num_envs, num_channels, self.buffer_size), device=self.device)
        self.current_pressure = torch.zeros((num_envs, num_channels), device=self.device)

    def reset_idx(self, env_ids: torch.Tensor):
        """
        [追加] 指定された環境のバッファをリセットする
        """
        if self.pressure_buffer is not None:
            self.pressure_buffer[env_ids] = 0.0
            self.current_pressure[env_ids] = 0.0

    def forward(self, target_pressure: torch.Tensor) -> torch.Tensor:
        # 初回実行時にバッファ確保
        if self.pressure_buffer is None:
            self._init_buffers(target_pressure.shape[0], target_pressure.shape[1])
            
        # 遅延バッファの更新 (シフトして末尾に新しい値を入れる)
        self.pressure_buffer = torch.roll(self.pressure_buffer, shifts=-1, dims=2)
        self.pressure_buffer[..., -1] = target_pressure
        
        # むだ時間経過後の入力
        delayed_input = self.pressure_buffer[..., 0]
        
        # 1次遅れフィルタの適用
        self.current_pressure = (1.0 - self.alpha) * self.current_pressure + self.alpha * delayed_input
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