# source/porcaro_rl/porcaro_rl/tasks/direct/porcaro_rlv1/actions/pam_dynamics.py
from __future__ import annotations
import torch
import torch.nn as nn
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..cfg.actuator_cfg import PamDelayModelCfg, PamHysteresisModelCfg, ActuatorNetModelCfg

class PamDelayModel(nn.Module):
    """
    空気圧の伝送遅れと応答遅れを模擬するモデル (マルチチャンネル対応版)
    """
    def __init__(self, cfg: PamDelayModelCfg, dt: float, device: str):
        super().__init__()
        self.cfg = cfg
        self.dt = dt
        self.device = device
        
        # バッファサイズ = delay_time / dt
        self.buffer_size = max(1, int(self.cfg.delay_time / self.dt))
        self.alpha = self.dt / (self.cfg.time_constant + self.dt)
        
        # 状態変数 [num_envs, num_channels, buffer_size]
        self.pressure_buffer = None
        self.current_pressure = None

    def _init_buffers(self, num_envs: int, num_channels: int):
        """バッファを初期化する"""
        self.pressure_buffer = torch.zeros((num_envs, num_channels, self.buffer_size), device=self.device)
        self.current_pressure = torch.zeros((num_envs, num_channels), device=self.device)

    def reset_idx(self, env_ids: torch.Tensor):
        """指定された環境のみ状態をリセット"""
        if self.pressure_buffer is not None:
            self.pressure_buffer[env_ids] = 0.0
            self.current_pressure[env_ids] = 0.0

    def forward(self, pressure_cmd: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pressure_cmd: 指令圧力 [num_envs, num_channels] (例: [N, 3])
        """
        num_envs, num_channels = pressure_cmd.shape
        
        # 初回またはサイズ変更時に初期化
        if (self.pressure_buffer is None or 
            self.pressure_buffer.shape[0] != num_envs or 
            self.pressure_buffer.shape[1] != num_channels):
            self._init_buffers(num_envs, num_channels)

        # 1. むだ時間（FIFOバッファ）
        # バッファ形状: [N, C, T] -> 最古のデータ: [:, :, 0]
        delayed_cmd = self.pressure_buffer[:, :, 0].clone()
        
        # シフト (dim=2方向にずらす)
        self.pressure_buffer = torch.roll(self.pressure_buffer, shifts=-1, dims=2)
        
        # 最新データを末尾に格納
        self.pressure_buffer[:, :, -1] = pressure_cmd

        # 2. 一次遅れフィルタ (Low Pass Filter)
        self.current_pressure = (1.0 - self.alpha) * self.current_pressure + self.alpha * delayed_cmd
        
        return self.current_pressure


class PamHysteresisModel(nn.Module):
    """
    [Modified] Additive Hysteresis Model (Play Operator)
    圧力に対する「摩擦（一定の圧力幅）」を再現するモデル。
    """
    def __init__(self, cfg: PamHysteresisModelCfg, device: str):
        super().__init__()
        self.cfg = cfg
        self.device = device
        # 状態変数: 「摩擦引きずり後」の圧力
        self.last_output = None

    def reset(self, num_envs: int, num_channels: int):
        self.last_output = torch.zeros((num_envs, num_channels), device=self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Play Operator:
        y(t) = max( x(t) - r, min( x(t) + r, y(t-1) ) )
        Input x: 空気圧 (Delay後のP_air)
        Output y: 有効圧力 (P_effective)
        """
        if self.last_output is None or self.last_output.shape != x.shape:
            self.last_output = x.clone()

        # 摩擦の半幅 (r)
        r = self.cfg.hysteresis_width / 2.0
        
        # 摩擦帯域の計算
        upper = x + r
        lower = x - r
        
        # 前回の値を、現在の入力の±rの範囲内に強制的に収める（引きずる）
        output = torch.min(upper, torch.max(lower, self.last_output))
        
        self.last_output = output.clone()
        return output


class ActuatorNetModel(nn.Module):
    """ActuatorNet (変更なしだが念のため記載)"""
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
        
        if self.cfg.model_path:
            try:
                state_dict = torch.load(self.cfg.model_path, map_location=self.device)
                self.net.load_state_dict(state_dict)
                print(f"[ActuatorNet] Loaded weights from {self.cfg.model_path}")
            except Exception as e:
                print(f"[Warning] Failed to load ActuatorNet weights: {e}")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)