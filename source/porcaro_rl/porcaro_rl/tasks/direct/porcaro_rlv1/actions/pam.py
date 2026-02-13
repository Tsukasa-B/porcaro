# source/porcaro_rl/porcaro_rl/tasks/direct/porcaro_rlv1/actions/pam.py
from __future__ import annotations
import math
import torch
import torch.nn as nn
import csv
from typing import Sequence

# 循環参照回避のため
from .pneumatic import(
    tau_L_from_pressure,
    first_order_lag,
    FractionalDelay,
    interp1d_clamp_torch,
    interp2d_bilinear,
    get_2d_tables,
    upsample_2d_bicubic,
)

# === [修正] Model A 用計算ロジック (Rad版) ===

def calculate_absolute_contraction(theta_rad, theta_t_rad, r, L0, clamp: bool = True):
    """
    Model A用: 符号を無視した絶対的な幾何学的収縮率を計算 (入力: Radian Tensor)
    """
    # ★修正: 内部での deg2rad を削除。入力は既に Tensor(rad) である前提。
    delta_theta = torch.abs(theta_rad - theta_t_rad)
    
    epsilon = (r * delta_theta) / L0
    
    if clamp:
        return torch.clamp(epsilon, min=0.0)
    return epsilon

def apply_model_a_force(force_map: PamForceMap, h0_map: H0Map, pressure: torch.Tensor, epsilon_geo: torch.Tensor) -> torch.Tensor:
    # 変更なし（Tensor演算のみなのでOK）
    h0 = h0_map(pressure)
    force = force_map(pressure, epsilon_geo)
    mask = (epsilon_geo <= h0).float()
    return force * mask

# === 既存クラス定義 (PamForceMap, H0Map) は変更なし ===
# ... (PamForceMap, H0Map のコードはそのまま) ...

class PamForceMap(nn.Module):
    def __init__(self, P_axis, h_axis, F_table, upsample: bool = True):
        super().__init__()
        if upsample:
            # 初期化時のCPU計算は許容
            P_fine, h_fine, F_fine = upsample_2d_bicubic(
                P_axis, h_axis, F_table, 
                num_x=128, num_y=128, device='cpu'
            )
            self.register_buffer('P', P_fine)
            self.register_buffer('h', h_fine)
            self.register_buffer('F', F_fine)
        else:
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
        h_axis = [x / 100.0 for x in h_axis]
        return PamForceMap(P_axis, h_axis, F, upsample=True)

    def forward(self, P_in: torch.Tensor, h_in: torch.Tensor) -> torch.Tensor:
        P_in = P_in.squeeze(-1)
        h_in = h_in.squeeze(-1)
        # Tensor演算のみなのでOK
        return interp2d_bilinear(self.P, self.h, self.F.T, P_in, h_in)

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
        h0_axis = [x / 100.0 for x in h0_axis]
        return H0Map(P_axis, h0_axis)

    def forward(self, P_in: torch.Tensor) -> torch.Tensor:
        P_in = P_in.squeeze(-1)
        return interp1d_clamp_torch(self.P, self.h0, P_in)


# === [修正] 有効収縮率計算 (Rad版) ===

@torch.no_grad()
def calculate_effective_contraction(theta_rad, theta_t_rad, r, L0, 
                                    slack_offset, pressure: torch.Tensor = 0.0, 
                                    shrink_gain: float = 0.0, clamp: bool = False,
                                    sign: float = 1.0):
    """
    Sim-to-Real用: 有効収縮率の計算 (入力: Radian Tensor)
    """
    # ★修正: 内部での変換(deg2rad/radians)を削除。全てTensor(rad)として扱う。
    
    # 幾何学的な縮み量
    delta_geo = r * sign * (theta_rad - theta_t_rad)
    L_geo = L0 - delta_geo
    
    # 圧力による「たるみ取り」
    slack_removal = shrink_gain * pressure
    
    # Slack Offset の適用
    s_off = torch.as_tensor(slack_offset, device=pressure.device, dtype=pressure.dtype)
    
    dynamic_offset = torch.where(
        s_off > 0.0,
        torch.clamp(s_off - slack_removal, min=0.0),
        s_off
    )
    
    L_eff = L_geo - dynamic_offset
    epsilon = (L0 - L_eff) / L0
    
    if clamp:
        return torch.clamp(epsilon, min=0.0)
    return epsilon

# === apply_soft_engagement, calculate_simple_latched_friction, PAMChannel は変更なし ===
# ... (これらの関数・クラスはTensor演算のみで構成されているため、そのままでOK) ...

def apply_soft_engagement(force: torch.Tensor, epsilon: torch.Tensor, h0: torch.Tensor, transition_width: float = 0.01) -> torch.Tensor:
    tautness = h0 - epsilon
    mask_val = torch.clamp(tautness / transition_width, min=0.0, max=1.0)
    return force * mask_val

def calculate_simple_latched_friction(h_dot, p_dot, pressure, prev_direction, viscosity, hys_coef_p, hys_const, p_dot_scale=1.0, tanh_width=0.1, latch_epsilon=0.25, contract_gain=1.0, extend_gain=1.0):
    f_viscous = -1.0 * viscosity * h_dot
    friction_magnitude = hys_const + hys_coef_p * torch.abs(pressure)
    v_virtual = torch.clamp(p_dot * p_dot_scale, min=-1.0, max=1.0) 
    v_effective = v_virtual
    current_dir = torch.tanh(v_effective / tanh_width)
    mask = (torch.abs(v_effective) > latch_epsilon).float()
    final_direction = current_dir * mask + prev_direction * (1.0 - mask)
    blend_pos = 0.5 * (1.0 + final_direction)
    blend_neg = 0.5 * (1.0 - final_direction)
    asym_scale = (contract_gain * blend_pos) + (extend_gain * blend_neg)
    f_hysteresis = -1.0 * (friction_magnitude * asym_scale) * final_direction
    return f_viscous + f_hysteresis, final_direction

class PAMChannel:
    # ... (既存コードそのまま) ...
    def __init__(self, dt_ctrl: float, tau: float = 0.09, dead_time: float = 0.03, Pmax: float = 0.6,
                 tau_lut: tuple[list[float], list[float]] | None = None,
                 use_table_i: bool = True,
                 use_2d_dynamics: bool = False,
                 latch_threshold: float = 0.02,
                 tau_scale_range: tuple[float, float] = (1.0, 1.0)):
        
        self.dt = dt_ctrl
        self.dead_time = float(dead_time)
        self.tau = float(tau)
        self.Pmax = float(Pmax)
        self.tau_scale_range = tau_scale_range
        self.delay = FractionalDelay(dt_ctrl, L_max=0.20)
        self.P_state = None
        self.use_2d_dynamics = True
        self.use_table_i = use_table_i
        self._tau_2d = None
        self._dead_2d = None
        self._p_axis_2d = None
        self.last_tau = None 
        self.current_tau_scale = None
        self.P_cmd_prev = None
        self.P_start_latch = None
        self.last_valid_direction = None 
        self.deadband = 1.0e-4
        self._tau_x, self._tau_y = None, None
        if tau_lut is not None:
            x, y = tau_lut
            self._tau_x = torch.tensor(x, dtype=torch.float32)
            self._tau_y = torch.tensor(y, dtype=torch.float32)

    def reset(self, n_envs: int, device: str | torch.device):
        dev = torch.device(device) if not isinstance(device, torch.device) else device
        z = torch.zeros(n_envs, device=dev, dtype=torch.float32)
        self.P_state = z.clone()
        self.delay.reset(z.shape, dev)
        self.last_tau = torch.full_like(z, float(self.tau))
        self.P_cmd_prev = z.clone()
        self.P_start_latch = z.clone()
        self.last_valid_direction = torch.zeros(n_envs, dtype=torch.long, device=dev)
        mid_val = sum(self.tau_scale_range) / 2.0
        self.current_tau_scale = torch.full((n_envs,), mid_val, device=dev, dtype=torch.float32)
        if self.use_2d_dynamics:
            self._tau_2d, self._dead_2d, self._p_axis_2d = get_2d_tables(dev)
        if self._tau_x is not None:
            self._tau_x = self._tau_x.to(dev)
            self._tau_y = self._tau_y.to(dev)

    @torch.no_grad()
    def reset_idx(self, env_ids: torch.Tensor | Sequence[int]):
        if self.P_state is not None:
            self.P_state[env_ids] = 0.0
        if self.last_tau is not None:
            self.last_tau[env_ids] = float(self.tau)
        if self.P_cmd_prev is not None:
            self.P_cmd_prev[env_ids] = 0.0
        if self.P_start_latch is not None:
            self.P_start_latch[env_ids] = 0.0
        if self.last_valid_direction is not None:
            self.last_valid_direction[env_ids] = 0
        if self.current_tau_scale is not None:
            low, high = self.tau_scale_range
            num_resets = len(env_ids)
            rand_scales = torch.rand(num_resets, device=self.current_tau_scale.device) * (high - low) + low
            self.current_tau_scale[env_ids] = rand_scales
        self.delay.reset_idx(env_ids)

    @torch.no_grad()
    def step(self, P_cmd: torch.Tensor) -> torch.Tensor:
        P_cmd = torch.clamp(P_cmd, 0.0, self.Pmax)
        if self.P_state is None or self.P_state.shape != P_cmd.shape:
            self.P_state = torch.zeros_like(P_cmd)
            self.P_cmd_prev = torch.zeros_like(P_cmd)
            self.P_start_latch = torch.zeros_like(P_cmd)
            self.last_valid_direction = torch.zeros_like(P_cmd, dtype=torch.long)
        
        diff = P_cmd - self.P_cmd_prev
        curr_direction = torch.zeros_like(diff, dtype=torch.long)
        curr_direction[diff > self.deadband] = 1
        curr_direction[diff < -self.deadband] = -1
        is_moving = (curr_direction != 0)
        direction_changed = (curr_direction != self.last_valid_direction)
        update_mask = is_moving & direction_changed
        if update_mask.any():
            self.P_start_latch[update_mask] = self.P_state[update_mask].detach()
            self.last_valid_direction[update_mask] = curr_direction[update_mask]
        self.P_cmd_prev = P_cmd.clone()

        if self.use_2d_dynamics and self._tau_2d is not None:
            tau_base = interp2d_bilinear(self._p_axis_2d, self._p_axis_2d, self._tau_2d, 
                                        x_query=P_cmd, y_query=self.P_start_latch)
            L_cmd    = interp2d_bilinear(self._p_axis_2d, self._p_axis_2d, self._dead_2d, 
                                        x_query=P_cmd, y_query=self.P_start_latch)
        elif self.use_table_i:
            if self._tau_x is None:
                tau_base, L_cmd = tau_L_from_pressure(P_cmd)
            else:
                tau_base = interp1d_clamp_torch(self._tau_x, self._tau_y, P_cmd)
                L_cmd = torch.full_like(P_cmd, self.dead_time)
        else:
            tau_base = torch.full_like(P_cmd, self.tau)
            L_cmd    = torch.full_like(P_cmd, self.dead_time)

        scale = self.current_tau_scale
        if P_cmd.ndim > 1:
            scale = scale.view(-1, 1)
        tau_final = tau_base * scale
        tau_final = torch.clamp(tau_final, min=1e-4)

        P_delayed = self.delay.step(P_cmd, L_cmd)
        self.last_tau = tau_final
        self.P_state = first_order_lag(P_delayed, self.P_state, self.last_tau, self.dt)
        return self.P_state