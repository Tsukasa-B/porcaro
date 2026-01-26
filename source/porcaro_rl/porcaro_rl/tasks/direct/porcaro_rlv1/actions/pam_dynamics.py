# source/porcaro_rl/porcaro_rl/tasks/direct/porcaro_rlv1/actions/pam_dynamics.py
from __future__ import annotations
import torch
import torch.nn as nn
from typing import TYPE_CHECKING
from .pneumatic import (
    interp2d_bilinear, 
    get_2d_tables, 
    FractionalDelay, 
    first_order_lag,
    get_dynamics_params_model_a, # <--- [New] 追加
    P_TAB, L_TAB # <--- [New] 追加 (Model A default)
)

if TYPE_CHECKING:
    from ..cfg.actuator_cfg import PamDelayModelCfg, PamHysteresisModelCfg, ActuatorNetModelCfg

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
        
        # --- [Modified] モード判定ロジックの刷新 ---
        self.model_type = getattr(cfg, "model_type", "B") # デフォルトB

        # --- Model B (2D High-Fidelity) 用テーブル準備 ---
        if self.model_type == "B":
            # 既存のロジック + 2Dテーブルロード
            self._tau_2d, self._dead_2d, self._p_axis_2d = get_2d_tables(self.device)
            self.register_buffer('tau_table', self._tau_2d)
            self.register_buffer('dead_table', self._dead_2d)
            self.register_buffer('p_axis', self._p_axis_2d)
            
        # --- Model A (Baseline) 用テーブル準備 ---
        elif self.model_type == "A":
            # Configに定義があればそれを使う、なければpneumatic.pyのデフォルト(P_TAB, L_TAB)
            p_axis = torch.tensor(cfg.deadtime_pressure_axis, device=device) if cfg.deadtime_pressure_axis else P_TAB.to(device)
            l_vals = torch.tensor(cfg.deadtime_values, device=device) if cfg.deadtime_values else L_TAB.to(device)
            
            self.register_buffer('_p_axis_1d', p_axis)
            self.register_buffer('_l_values_1d', l_vals)
            self.tau_const = getattr(cfg, "tau_const", 0.15) # デフォルト0.15

        # --- 共通モジュールの利用 ---
        # FractionalDelayは状態を持つがnn.Moduleではないため、ここでインスタンス化
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
        # --- [Modified] Model A / B 分岐 ---
        if self.model_type == "A":
            # Model A: Constant Tau + 1D Deadtime
            tau, L = get_dynamics_params_model_a(
                target_pressure, self.tau_const, self._p_axis_1d, self._l_values_1d
            )
            
        elif self.model_type == "B":
            # Model B: 2D Maps (Pressure-Dependent)
            # Current pressure state for lookup
            P_curr = self.current_pressure.detach() 
            tau = interp2d_bilinear(self.p_axis, self.p_axis, self.tau_table, 
                                  x_query=target_pressure, y_query=P_curr)
            L = interp2d_bilinear(self.p_axis, self.p_axis, self.dead_table, 
                                  x_query=target_pressure, y_query=P_curr)
        else:
            # Fallback (Legacy)
            tau = torch.full_like(target_pressure, 0.15)
            L = torch.full_like(target_pressure, 0.04)

        # 2. 共通ロジックによる遅延とフィルタ
        P_delayed = self.delay_proc.step(target_pressure, L)
        self.current_pressure = first_order_lag(P_delayed, self.current_pressure, tau, self.dt)
        
        return self.current_pressure

# (PamHysteresisModel, ActuatorNetModel は変更なし)
class PamHysteresisModel(nn.Module):
    # ... (変更なし) ...
    def __init__(self, cfg: PamHysteresisModelCfg, device: str):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.last_output = None
    def reset_idx(self, env_ids: torch.Tensor):
        if self.last_output is not None:
            self.last_output[env_ids] = 0.0
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.last_output is None or self.last_output.shape != x.shape:
            self.last_output = x.clone()
        r = self.cfg.hysteresis_width / 2.0
        lower_bound = x - r
        upper_bound = x + r
        output = torch.max(lower_bound, torch.min(upper_bound, self.last_output))
        self.last_output = output.clone()
        return output

class ActuatorNetModel(nn.Module):
    # ... (変更なし) ...
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