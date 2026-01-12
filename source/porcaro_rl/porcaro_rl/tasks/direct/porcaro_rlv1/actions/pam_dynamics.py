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
    PAMのヒステリシス特性モデル (マルチチャンネル対応版)
    """
    def __init__(self, cfg: PamHysteresisModelCfg, device: str):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.last_input = None
        self.last_output = None

    def _init_state(self, num_envs: int, num_channels: int):
        self.last_input = torch.zeros((num_envs, num_channels), device=self.device)
        self.last_output = torch.zeros((num_envs, num_channels), device=self.device)

    def reset_idx(self, env_ids: torch.Tensor):
        if self.last_input is not None:
            self.last_input[env_ids] = 0.0
            self.last_output[env_ids] = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num_envs, num_channels = x.shape
        
        if (self.last_input is None or 
            self.last_input.shape[0] != num_envs or
            self.last_input.shape[1] != num_channels):
            self._init_state(num_envs, num_channels)
            
        delta = x - self.last_input
        
        # ヒステリシス係数の適用 (増圧時: 1.0, 減圧時: 1.0 - width)
        hysteresis_factor = torch.where(
            delta > 0, 
            torch.tensor(1.0, device=self.device), 
            torch.tensor(1.0 - self.cfg.hysteresis_width, device=self.device)
        )
        
        output = x * hysteresis_factor
        self.last_input = x.clone()
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