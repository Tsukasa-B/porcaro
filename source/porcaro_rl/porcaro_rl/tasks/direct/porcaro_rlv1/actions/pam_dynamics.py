# source/porcaro_rl/porcaro_rl/tasks/direct/porcaro_rlv1/actions/pam_dynamics.py

from __future__ import annotations
import torch
import torch.nn as nn
from typing import TYPE_CHECKING
from .pneumatic import (
    interp2d_bilinear, 
    get_2d_tables, 
    FractionalDelay, 
    first_order_lag
)

if TYPE_CHECKING:
    # 変更: PamHysteresisModelCfg を削除
    from ..cfg.actuator_cfg import PamDelayModelCfg, ActuatorNetModelCfg

class PamDelayModel(nn.Module):
    """
    空気圧の伝送遅れ(可変)と応答遅れ(可変)を模擬するモデル
    pneumatic.py のロジックをラップして nn.Module 化したもの。
    """
    def __init__(self, cfg: PamDelayModelCfg, dt: float, device: str):
        super().__init__()
        self.cfg = cfg
        self.dt = dt
        self.device = device
        
        # --- モード判定 ---
        tau_vals = self.cfg.tau_values if self.cfg.tau_values else [self.cfg.time_constant]
        dead_vals = self.cfg.deadtime_values if self.cfg.deadtime_values else [self.cfg.delay_time]
        self.is_scalar_mode = (len(tau_vals) <= 1 and len(dead_vals) <= 1)
        
        if self.is_scalar_mode:
            self.fixed_tau = float(tau_vals[0])
            self.fixed_dead = float(dead_vals[0])
        else:
            # Shared Data from pneumatic module
            self._tau_2d, self._dead_2d, self._p_axis_2d = get_2d_tables(self.device)
            # 永続化バッファとして登録（保存・ロード対応）
            self.register_buffer('tau_table', self._tau_2d)
            self.register_buffer('dead_table', self._dead_2d)
            self.register_buffer('p_axis', self._p_axis_2d)

        # --- 共通モジュールの利用 ---
        self.delay_proc = FractionalDelay(self.dt, L_max=self.cfg.max_delay_time)
        
        # 状態変数
        self.current_pressure = None

    def reset_idx(self, env_ids: torch.Tensor):
        if self.current_pressure is not None:
            self.current_pressure[env_ids] = 0.0
        # pneumatic.FractionalDelay のリセット機能を呼ぶ
        self.delay_proc.reset_idx(env_ids)

    def forward(self, target_pressure: torch.Tensor) -> torch.Tensor:
        # 初回初期化 (バッファ未確保時)
        if self.delay_proc.buf is None:
             self.delay_proc.reset(target_pressure.shape, self.device)
        if self.current_pressure is None:
             self.current_pressure = torch.zeros_like(target_pressure)

        # 1. パラメータ(L, Tau) の決定
        if self.is_scalar_mode:
            L = torch.full_like(target_pressure, self.fixed_dead)
            tau = torch.full_like(target_pressure, self.fixed_tau)
        else:
            # Current pressure state for lookup
            P_curr = self.current_pressure.detach() 
            tau = interp2d_bilinear(self.p_axis, self.p_axis, self.tau_table, 
                                  x_query=target_pressure, y_query=P_curr)
            L = interp2d_bilinear(self.p_axis, self.p_axis, self.dead_table, 
                                  x_query=target_pressure, y_query=P_curr)

        # 2. 共通ロジックによる遅延とフィルタ
        P_delayed = self.delay_proc.step(target_pressure, L)
        self.current_pressure = first_order_lag(P_delayed, self.current_pressure, tau, self.dt)
        
        return self.current_pressure

# --------------------------------------------------------
# [削除] PamHysteresisModel クラスをここに定義していましたが削除しました
# --------------------------------------------------------

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
        
        if self.cfg.model_path:
            import os
            if os.path.exists(self.cfg.model_path):
                state_dict = torch.load(self.cfg.model_path, map_location=self.device)
                self.net.load_state_dict(state_dict)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)