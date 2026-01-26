# source/porcaro_rl/porcaro_rl/tasks/direct/porcaro_rlv1/actions/pam_dynamics.py
from __future__ import annotations
import torch
import torch.nn as nn
from typing import TYPE_CHECKING

# pneumatic から必要な関数をインポート
from .pneumatic import (
    interp2d_bilinear, 
    get_2d_tables, 
    FractionalDelay, 
    first_order_lag,
    tau_L_from_pressure # <--- 追加
)

if TYPE_CHECKING:
    from ..cfg.actuator_cfg import PamDelayModelCfg, ActuatorNetModelCfg

class PamDelayModel(nn.Module):
    def __init__(self, cfg: PamDelayModelCfg, dt: float, device: str):
        super().__init__()
        self.cfg = cfg
        self.dt = dt
        self.device = device
        
        # --- Model A 判定 ---
        # fixed_time_constant が設定されている場合は Model A とみなす
        self.is_model_a = getattr(cfg, "fixed_time_constant", None) is not None

        if self.is_model_a:
            # [Model A] 固定時定数モード
            self.fixed_tau = cfg.fixed_time_constant
            # 1Dテーブルは関数(tau_L_from_pressure)内で持つためバッファは不要だが
            # 分岐ロジックとの互換性のためダミー登録
            self.register_buffer('p_axis', torch.tensor([])) 
        else:
            # [Model B] 既存ロジック
            tau_vals = self.cfg.tau_values if self.cfg.tau_values else [self.cfg.time_constant]
            dead_vals = self.cfg.deadtime_values if self.cfg.deadtime_values else [self.cfg.delay_time]
            self.is_scalar_mode = (len(tau_vals) <= 1 and len(dead_vals) <= 1)
            
            if self.is_scalar_mode:
                self.fixed_tau = float(tau_vals[0])
                self.fixed_dead = float(dead_vals[0])
            else:
                self._tau_2d, self._dead_2d, self._p_axis_2d = get_2d_tables(self.device)
                self.register_buffer('tau_table', self._tau_2d)
                self.register_buffer('dead_table', self._dead_2d)
                self.register_buffer('p_axis', self._p_axis_2d)

        self.delay_proc = FractionalDelay(self.dt, L_max=self.cfg.max_delay_time)
        self.current_pressure = None

    def reset_idx(self, env_ids: torch.Tensor):
        if self.current_pressure is not None:
            self.current_pressure[env_ids] = 0.0
        self.delay_proc.reset_idx(env_ids)

    def forward(self, target_pressure: torch.Tensor) -> torch.Tensor:
        if self.delay_proc.buf is None:
             self.delay_proc.reset(target_pressure.shape, self.device)
        if self.current_pressure is None:
             self.current_pressure = torch.zeros_like(target_pressure)

        # 1. パラメータ決定ロジックの分岐
        if self.is_model_a:
            # [Model A]
            # Time Constant: 固定値
            tau = torch.full_like(target_pressure, self.fixed_tau)
            
            # Dead Time: 1D Lookup Table (L only)
            # pneumatic.tau_L_from_pressure を利用して L だけ取得
            # (tau_L_from_pressure は (tau, L) を返すが、Model AではLだけ使う)
            _, L = tau_L_from_pressure(target_pressure)
            
        elif self.is_scalar_mode:
            L = torch.full_like(target_pressure, self.fixed_dead)
            tau = torch.full_like(target_pressure, self.fixed_tau)
        else:
            # [Model B] 2D Lookup
            P_curr = self.current_pressure.detach() 
            tau = interp2d_bilinear(self.p_axis, self.p_axis, self.tau_table, 
                                  x_query=target_pressure, y_query=P_curr)
            L = interp2d_bilinear(self.p_axis, self.p_axis, self.dead_table, 
                                  x_query=target_pressure, y_query=P_curr)

        # 2. 遅延とフィルタ (共通)
        P_delayed = self.delay_proc.step(target_pressure, L)
        self.current_pressure = first_order_lag(P_delayed, self.current_pressure, tau, self.dt)
        
        return self.current_pressure

# ActuatorNetModel は変更なし
class ActuatorNetModel(nn.Module):
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