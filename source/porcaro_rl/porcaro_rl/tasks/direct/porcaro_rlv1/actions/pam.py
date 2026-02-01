# source/porcaro_rl/porcaro_rl/tasks/direct/porcaro_rlv1/actions/pam.py
from __future__ import annotations
import math
import torch
import torch.nn as nn
import csv
from typing import Sequence

# 循環参照回避のため、pneumaticのみインポート
from .pneumatic import(
    tau_L_from_pressure,
    first_order_lag,
    FractionalDelay,
    interp1d_clamp_torch,
    interp2d_bilinear,
    get_2d_tables,
)

# === Model A 用の計算ロジック ===

def calculate_absolute_contraction(theta_deg, theta_t_deg, r, L0, clamp: bool = True):
    """
    Model A用: 符号を無視した絶対的な幾何学的収縮率を計算
    epsilon = | (r/L0) * (theta - theta_t) |
    """
    theta_rad = torch.deg2rad(theta_deg)
    theta_t_rad = math.radians(theta_t_deg)
    
    # 符号(sign)は考慮せず、変位の絶対値を取る
    delta_theta = torch.abs(theta_rad - theta_t_rad)
    
    # 幾何学的収縮率
    epsilon = (r * delta_theta) / L0
    
    if clamp:
        return torch.clamp(epsilon, min=0.0)
    return epsilon

def apply_model_a_force(force_map: PamForceMap, h0_map: H0Map, pressure: torch.Tensor, epsilon_geo: torch.Tensor) -> torch.Tensor:
    """
    Model A用: Soft Engagementなしの力生成
    - h0判定によるハードカットオフ (縮みすぎたら即座に0)
    - Slack Offset計算なし (epsilon_geoを直接使用)
    """
    # 1. 平衡収縮率 h0 の取得
    h0 = h0_map(pressure)
    
    # 2. 力の計算 (Map直引き)
    force = force_map(pressure, epsilon_geo)
    
    # 3. ハードカットオフ (Soft Engagementなし)
    mask = (epsilon_geo <= h0).float()
    
    return force * mask

# === 以下、既存クラス定義 ===

class PamForceMap(nn.Module):
    def __init__(self, P_axis, h_axis, F_table):
        super().__init__()
        self.register_buffer('P', torch.as_tensor(P_axis, dtype=torch.float32))
        self.register_buffer('h', torch.as_tensor(h_axis, dtype=torch.float32))
        self.register_buffer('F', torch.as_tensor(F_table, dtype=torch.float32))

    @staticmethod
    def from_csv(path: str):
        with open(path, newline="") as f:
            rows = list(csv.reader(f))
        h_axis = [float(x) for x in rows[0][1:] if x != ""]
        P_axis, F = [], []
        for r in rows[1:]:
            if not r or r[0] == "": continue
            P_axis.append(float(r[0]))
            F.append([float(x) for x in r[1:1+len(h_axis)]])
        if max(h_axis) > 1.0:
            h_axis = [x / 100.0 for x in h_axis]
        return PamForceMap(P_axis, h_axis, F)

    def forward(self, P_in: torch.Tensor, h_in: torch.Tensor) -> torch.Tensor:
        P_in = P_in.squeeze(-1)
        h_in = h_in.squeeze(-1)
        P_axis = self.P; h_axis = self.h; Ftab = self.F
        P_in = torch.clamp(P_in, P_axis[0], P_axis[-1])
        h_in = torch.clamp(h_in, h_axis[0], h_axis[-1])
        return interp2d_bilinear(P_axis, h_axis, Ftab.T, P_in, h_in)

class H0Map(nn.Module):
    def __init__(self, P_axis, h0_axis):
        super().__init__()
        self.register_buffer('P', torch.as_tensor(P_axis, dtype=torch.float32))
        self.register_buffer('h0', torch.as_tensor(h0_axis, dtype=torch.float32))

    @staticmethod
    def from_csv(path: str):
        P_axis, h0_axis = [], []
        with open(path, newline="") as f:
            rows = list(csv.reader(f))
        for r in rows[1:]:
            if len(r) < 2: continue
            try:
                P_axis.append(float(r[0]))
                h0_axis.append(float(r[1]))
            except Exception: pass
        if max(h0_axis) > 1.0: h0_axis = [x / 100.0 for x in h0_axis]
        return H0Map(P_axis, h0_axis)

    def forward(self, P_in: torch.Tensor) -> torch.Tensor:
        P_in = P_in.squeeze(-1)
        return interp1d_clamp_torch(self.P, self.h0, P_in)

@torch.no_grad()
def contraction_ratio_from_angle(theta: torch.Tensor, theta_t_deg: float, r: float, L: float, sign: float = 1.0) -> torch.Tensor:
    theta_t = math.radians(theta_t_deg)
    return (r / L) * sign * (theta - theta_t)

def calculate_effective_contraction(theta_deg, theta_t_deg, r, L0, 
                                    slack_offset, pressure: torch.Tensor = 0.0, 
                                    shrink_gain: float = 0.0, clamp: bool = False,
                                    sign: float = 1.0):
    theta_rad = torch.deg2rad(theta_deg)
    theta_t_rad = math.radians(theta_t_deg)
    delta_geo = r * sign * (theta_rad - theta_t_rad)
    L_geo = L0 - delta_geo
    dynamic_offset = slack_offset - (shrink_gain * pressure)
    L_eff = L_geo - dynamic_offset
    epsilon = (L0 - L_eff) / L0
    if clamp:
        return torch.clamp(epsilon, min=0.0)
    return epsilon

def apply_soft_engagement(force: torch.Tensor, epsilon: torch.Tensor, smoothness: float = 100.0) -> torch.Tensor:
    slack_amount = torch.clamp(epsilon, min=0.0)
    x = slack_amount * smoothness
    mask_val = torch.clamp(1.0 - x, min=0.0)
    mask_val = mask_val * mask_val
    return force * mask_val

@torch.no_grad()
def Fpam_quasi_static(P: torch.Tensor, h: torch.Tensor, N: float = 1200.0, Pmax: float = 0.6) -> torch.Tensor:
    h0 = 0.25 * (P / Pmax).clamp(0,1)
    eff = torch.clamp(h0 - h, min=0.0)
    return N * P * eff

class PAMChannel:
    def __init__(self, dt_ctrl: float, tau: float = 0.09, dead_time: float = 0.03, Pmax: float = 0.6,
                 tau_lut: tuple[list[float], list[float]] | None = None,
                 use_table_i: bool = True,
                 use_2d_dynamics: bool = False):
        
        self.dt = dt_ctrl
        self.dead_time = float(dead_time)
        self.tau = float(tau) # 修正: デフォルト値を 1.09 -> 0.09 へ
        self.Pmax = float(Pmax)
        self.delay = FractionalDelay(dt_ctrl, L_max=0.20)
        
        self.P_state = None
        # Latch用
        self.p_start_latch = None 
        self.prev_target = None
        
        self.use_2d_dynamics = use_2d_dynamics
        self.use_table_i = use_table_i
        self._tau_2d, self._dead_2d, self._p_axis_2d = None, None, None
        self._L_prev = None
        self.last_tau = None 
        self._tau_x, self._tau_y = None, None
        
        if tau_lut is not None:
            x, y = tau_lut
            self._tau_x = torch.tensor(x, dtype=torch.float32)
            self._tau_y = torch.tensor(y, dtype=torch.float32)

    def reset(self, n_envs: int, device: str | torch.device):
        dev = torch.device(device) if not isinstance(device, torch.device) else device
        z = torch.zeros(n_envs, device=dev, dtype=torch.float32)
        self.P_state = z.clone()
        
        self.p_start_latch = z.clone()
        self.prev_target = z.clone()
        
        self.delay.reset(z.shape, dev)
        self.last_tau = torch.full_like(z, float(self.tau))
        self._L_prev  = None
        if self.use_2d_dynamics:
            self._tau_2d, self._dead_2d, self._p_axis_2d = get_2d_tables(dev)
        if self._tau_x is not None:
            self._tau_x = self._tau_x.to(dev)
            self._tau_y = self._tau_y.to(dev)

    @torch.no_grad()
    def reset_idx(self, env_ids: torch.Tensor | Sequence[int]):
        if self.P_state is not None:
            self.P_state[env_ids] = 0.0
        
        if self.p_start_latch is not None:
            self.p_start_latch[env_ids] = 0.0
        if self.prev_target is not None:
            self.prev_target[env_ids] = 0.0
            
        if self.last_tau is not None:
            self.last_tau[env_ids] = float(self.tau)
        if self._L_prev is not None:
            self._L_prev[env_ids] = 0.0
        self.delay.reset_idx(env_ids)

    @torch.no_grad()
    def step(self, P_cmd: torch.Tensor) -> torch.Tensor:
        P_cmd = torch.clamp(P_cmd, 0.0, self.Pmax)
        
        # === Change Detection & Latch Logic ===
        if self.P_state is None:
             self.P_state = torch.zeros_like(P_cmd)
             self.p_start_latch = torch.zeros_like(P_cmd)
             self.prev_target = torch.zeros_like(P_cmd)

        change_mask = torch.abs(P_cmd - self.prev_target) > 1e-4
        if change_mask.any():
            self.p_start_latch[change_mask] = self.P_state[change_mask].detach()
            self.prev_target[change_mask] = P_cmd[change_mask]
        # ======================================

        if self.use_2d_dynamics and self._tau_2d is not None:
            # Model B: 2D Dynamics
            tau_now = interp2d_bilinear(self._p_axis_2d, self._p_axis_2d, self._tau_2d, 
                                      x_query=P_cmd, y_query=self.p_start_latch)
            L_cmd   = interp2d_bilinear(self._p_axis_2d, self._p_axis_2d, self._dead_2d, 
                                      x_query=P_cmd, y_query=self.p_start_latch)
            
        elif self.use_table_i:
            # ★修正箇所: Model A/Default Logic
            # use_table_i が True なら、むだ時間(L)は常に1Dテーブル(tau_L_from_pressure)から取得する。
            # これにより、ConfigでLUTが指定されている場合でも、むだ時間の計算はテーブルに従う。
            
            # 1. むだ時間の決定 (Table I Logic)
            _, L_table_val = tau_L_from_pressure(P_cmd)
            L_cmd = L_table_val

            # 2. 時定数の決定
            if self._tau_x is None:
                # Model A (Ideal): 固定時定数
                tau_now = torch.full_like(P_cmd, self.tau)
            else:
                # Custom LUTが提供されている場合 (Configで use_pressure_dependent_tau=True の場合)
                tau_now = interp1d_clamp_torch(self._tau_x, self._tau_y, P_cmd)
        else:
            # 完全固定モデル
            tau_now = torch.full_like(P_cmd, self.tau)
            L_cmd   = torch.full_like(P_cmd, self.dead_time)

        P_delayed = self.delay.step(P_cmd, L_cmd)
        
        if self.P_state is None or self.P_state.shape != P_cmd.shape:
             self.P_state = P_delayed.clone()

        self.last_tau = torch.clamp(tau_now, min=1e-6)
        self.P_state = first_order_lag(P_delayed, self.P_state, self.last_tau, self.dt)
        return self.P_state