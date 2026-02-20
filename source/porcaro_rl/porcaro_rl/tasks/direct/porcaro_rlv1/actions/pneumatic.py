# source/porcaro_rl/porcaro_rl/tasks/direct/porcaro_rlv1/actions/pneumatic.py
from __future__ import annotations
import math
import torch
from typing import Sequence
import numpy as np  # 追加
from scipy.interpolate import PchipInterpolator, RectBivariateSpline

# =========================================================
#  Physical Parameters (Single Source of Truth)
# =========================================================

# Table I (Legacy 1D)
P_TAB   = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=torch.float32)
TAU_TAB = torch.tensor([0.043, 0.045, 0.060, 0.066, 0.094, 0.131], dtype=torch.float32)
L_TAB   = torch.tensor([0.038, 0.035, 0.032, 0.030, 0.023, 0.023], dtype=torch.float32)

# High-Fidelity 2D Tables
TAU_TABLE_2D_DATA = [
    [0.0010, 0.0950, 0.0850, 0.0800, 0.0700, 0.0950, 0.1000],
    [0.0850, 0.0010, 0.0750, 0.0950, 0.0900, 0.0800, 0.1050],
    [0.0800, 0.0550, 0.0010, 0.0500, 0.1050, 0.0800, 0.1000],
    [0.0550, 0.0950, 0.0550, 0.0010, 0.0350, 0.0750, 0.0700],
    [0.0600, 0.0850, 0.1150, 0.0950, 0.0010, 0.1550, 0.1400],
    [0.0600, 0.0950, 0.0550, 0.1000, 0.1150, 0.0010, 0.1750],
    [0.0600, 0.0850, 0.0700, 0.0850, 0.0900, 0.1600, 0.0010],
]

DEAD_TABLE_2D_DATA = [
    [0.0010, 0.0050, 0.0100, 0.0150, 0.0600, 0.0550, 0.0550],
    [0.0100, 0.0010, 0.0450, 0.0100, 0.0300, 0.0650, 0.0500],
    [0.0150, 0.0450, 0.0010, 0.0750, 0.0100, 0.0750, 0.0600],
    [0.0550, 0.0800, 0.0850, 0.0010, 0.1050, 0.0850, 0.0650],
    [0.0500, 0.0650, 0.0200, 0.0400, 0.0010, 0.0050, 0.0250],
    [0.0450, 0.0700, 0.0700, 0.0300, 0.0300, 0.0010, 0.0350],
    [0.0700, 0.0750, 0.0750, 0.0350, 0.0300, 0.0050, 0.0010],
]

def get_2d_tables(device: torch.device | str):
    dev = torch.device(device) if isinstance(device, str) else device
    tau_t = torch.tensor(TAU_TABLE_2D_DATA, dtype=torch.float32, device=dev)
    dead_t = torch.tensor(DEAD_TABLE_2D_DATA, dtype=torch.float32, device=dev)
    p_axis = torch.linspace(0.0, 0.6, 7, dtype=torch.float32, device=dev)
    return tau_t, dead_t, p_axis

# =========================================================
#  Math & Physics Functions
# =========================================================

def _tabs_on(P: torch.Tensor):
    return (
        P_TAB.to(device=P.device, dtype=P.dtype),
        TAU_TAB.to(device=P.device, dtype=P.dtype),
        L_TAB.to(device=P.device, dtype=P.dtype),
    )

@torch.no_grad()
def interp1d_clamp_torch(xp: torch.Tensor, fp: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    x = x.to(xp.dtype).to(xp.device)
    if torch.isnan(x).any():
        x = torch.nan_to_num(x, nan=float(xp[0]))
    x = torch.clamp(x, min=float(xp[0]), max=float(xp[-1]))

    idx = torch.bucketize(x, xp).clamp(1, xp.numel()-1)
    x0, x1 = xp[idx-1], xp[idx]
    f0, f1 = fp[idx-1], fp[idx]
    t = (x - x0) / (x1 - x0 + 1e-12)
    return f0 + t * (f1 - f0)

@torch.no_grad()
def interp2d_bilinear(
    x_grid: torch.Tensor, y_grid: torch.Tensor, z_data: torch.Tensor, 
    x_query: torch.Tensor, y_query: torch.Tensor
) -> torch.Tensor:
    dev = x_query.device
    dtype = x_query.dtype
    
    if x_grid.device != dev: x_grid = x_grid.to(dev, dtype=dtype)
    if y_grid.device != dev: y_grid = y_grid.to(dev, dtype=dtype)
    if z_data.device != dev: z_data = z_data.to(dev, dtype=dtype)
    
    x_q = torch.nan_to_num(x_query, nan=float(x_grid[0])).clamp(min=float(x_grid[0]), max=float(x_grid[-1]))
    y_q = torch.nan_to_num(y_query, nan=float(y_grid[0])).clamp(min=float(y_grid[0]), max=float(y_grid[-1]))

    dx = x_grid[1] - x_grid[0]
    x_idx_f = (x_q - x_grid[0]) / dx
    x0 = x_idx_f.floor().long().clamp(0, len(x_grid)-2)
    x1 = x0 + 1
    wx = x_idx_f - x0

    dy = y_grid[1] - y_grid[0]
    y_idx_f = (y_q - y_grid[0]) / dy
    y0 = y_idx_f.floor().long().clamp(0, len(y_grid)-2)
    y1 = y0 + 1
    wy = y_idx_f - y0

    v00 = z_data[y0, x0]
    v01 = z_data[y0, x1]
    v10 = z_data[y1, x0]
    v11 = z_data[y1, x1]

    r0 = v00 * (1 - wx) + v01 * wx
    r1 = v10 * (1 - wx) + v11 * wx
    
    val = r0 * (1 - wy) + r1 * wy
    return val

@torch.no_grad()
def upsample_1d_pchip(x_coarse: Sequence[float], y_coarse: Sequence[float], num_points: int = 100, device='cpu'):
    """
    1次元データをPCHIP補間（単調性維持）で高解像度化し、Torch Tensorとして返す。
    """
    x_np = np.array(x_coarse)
    y_np = np.array(y_coarse)
    
    # 念のためソート
    idx = np.argsort(x_np)
    x_np = x_np[idx]
    y_np = y_np[idx]

    interpolator = PchipInterpolator(x_np, y_np)
    
    x_fine_np = np.linspace(x_np[0], x_np[-1], num_points)
    y_fine_np = interpolator(x_fine_np)
    
    return (
        torch.as_tensor(x_fine_np, dtype=torch.float32, device=device),
        torch.as_tensor(y_fine_np, dtype=torch.float32, device=device)
    )

@torch.no_grad()
def upsample_2d_bicubic(x_axis: Sequence[float], y_axis: Sequence[float], z_data: list[list[float]], 
                        num_x: int = 100, num_y: int = 100, device='cpu'):
    """
    2次元データを双3次スプライン(Bicubic)で高解像度化し、Torch Tensorとして返す。
    """
    x_np = np.array(x_axis)
    y_np = np.array(y_axis)
    z_np = np.array(z_data) 

    # 配列形状の整合性チェックと転置
    if z_np.shape != (len(x_np), len(y_np)):
        z_np = z_np.T

    # 双3次スプライン補間 (kx=3, ky=3)
    interpolator = RectBivariateSpline(x_np, y_np, z_np, kx=3, ky=3)
    
    x_fine_np = np.linspace(x_np[0], x_np[-1], num_x)
    y_fine_np = np.linspace(y_np[0], y_np[-1], num_y)
    
    # 新しいグリッド上の値を計算
    z_fine_np = interpolator(x_fine_np, y_fine_np)
    
    return (
        torch.as_tensor(x_fine_np, dtype=torch.float32, device=device),
        torch.as_tensor(y_fine_np, dtype=torch.float32, device=device),
        torch.as_tensor(z_fine_np, dtype=torch.float32, device=device)
    )

@torch.no_grad()
def tau_L_from_pressure(P, zero_mode="clamp"):
    P = P if isinstance(P, torch.Tensor) else torch.as_tensor(P, dtype=torch.float32)
    XP, TAU, L = _tabs_on(P)
    P = torch.nan_to_num(P, nan=0.0)
    Pq  = torch.clamp(P, XP[0], XP[-1])
    tau = interp1d_clamp_torch(XP, TAU, Pq)
    L   = interp1d_clamp_torch(XP, L,   Pq)
    return tau, L

@torch.no_grad()
def first_order_lag(u_now: torch.Tensor, x_prev: torch.Tensor, tau, dt: float) -> torch.Tensor:
    if isinstance(tau, torch.Tensor):
        tau_t = torch.clamp(tau.to(u_now.dtype).to(u_now.device), min=1e-9)
        alpha = dt / (tau_t + dt)
        x_next = x_prev + alpha * (u_now - x_prev)
        return torch.where(tau <= 1e-6, u_now, x_next)
    else:
        tau_f = float(tau)
        if tau_f <= 1e-6:
            return u_now
        alpha = dt / (tau_f + dt)
        return x_prev + alpha * (u_now - x_prev)

class FractionalDelay:
    def __init__(self, dt: float, L_max: float = 0.20): 
        self.dt = float(dt)
        self.K  = int(math.ceil(L_max / self.dt) + 5)
        self.buf = None
        self.wp  = 0

    def reset(self, shape: torch.Size, device):
        self.buf = torch.zeros(self.K, *shape, dtype=torch.float32, device=device)
        self.wp  = 0

    @torch.no_grad()
    def reset_idx(self, env_ids: torch.Tensor | Sequence[int]):
        if self.buf is None: return
        if self.buf.ndim >= 2:
            self.buf[:, env_ids] = 0.0
        else:
            if isinstance(env_ids, (torch.Tensor, list, tuple)) and len(env_ids) > 0:
                self.buf.fill_(0.0)

    @torch.no_grad()
    def step(self, x: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        if self.buf is None:
            self.reset(x.shape, x.device)

        L = torch.nan_to_num(L, nan=0.0)
        x = torch.nan_to_num(x, nan=0.0)

        D  = torch.clamp(L / self.dt, min=0.0, max=self.K - 2.0)
        M  = torch.floor(D).to(torch.int64)
        mu = (D - M).to(torch.float32)

        self.buf[self.wp] = x

        if x.ndim == 0:
            idx0 = (self.wp - int(M)) % self.K
            idx1 = (self.wp - int(M) - 1) % self.K
            out = (1.0 - float(mu)) * self.buf[idx0] + float(mu) * self.buf[idx1]
        else:
            if x.ndim == 2:
                b_idx = torch.arange(x.shape[0], device=x.device).unsqueeze(1).expand_as(x)
                c_idx = torch.arange(x.shape[1], device=x.device).unsqueeze(0).expand_as(x)
                idx0 = (self.wp - M) % self.K
                idx1 = (self.wp - M - 1) % self.K
                out0 = self.buf[idx0, b_idx, c_idx]
                out1 = self.buf[idx1, b_idx, c_idx]
            else:
                b_idx = torch.arange(x.shape[0], device=x.device)
                idx0 = (self.wp - M) % self.K
                idx1 = (self.wp - M - 1) % self.K
                out0 = self.buf[idx0, b_idx]
                out1 = self.buf[idx1, b_idx]
                
            out  = (1.0 - mu) * out0 + mu * out1

        self.wp = (self.wp + 1) % self.K
        return out

class PAMChannel:
    def __init__(self, dt_ctrl: float, tau: float = 0.09, dead_time: float = 0.03, Pmax: float = 0.6,
                 tau_lut: tuple[list[float], list[float]] | None = None,
                 use_table_i: bool = True,
                 latch_threshold: float = 0.02): 
        
        self.dt = dt_ctrl
        self.dead_time = float(dead_time)
        self.tau = float(tau)
        self.Pmax = float(Pmax)
        self.delay = FractionalDelay(dt_ctrl, L_max=0.20)
        self.P_state = None
        
        self.use_2d_dynamics = True
        self.use_table_i = use_table_i
        
        self._tau_2d = None
        self._dead_2d = None
        self._p_axis_2d = None

        self.last_tau = None 
        
        self.P_cmd_prev = None
        self.P_start_latch = None
        # 修正: "直前の有効な方向" を保持する変数
        self.last_valid_direction = None # 1:Inf, -1:Def, 0:Init
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
            
        self.delay.reset_idx(env_ids)

    @torch.no_grad()
    def step(self, P_cmd: torch.Tensor) -> torch.Tensor:
        P_cmd = torch.clamp(P_cmd, 0.0, self.Pmax)
        
        if self.P_state is None or self.P_state.shape != P_cmd.shape:
            # 初回初期化
            self.P_state = torch.zeros_like(P_cmd)
            self.P_cmd_prev = torch.zeros_like(P_cmd)
            self.P_start_latch = torch.zeros_like(P_cmd)
            self.last_valid_direction = torch.zeros_like(P_cmd, dtype=torch.long)

        # === 修正箇所: Hold状態を挟んでも文脈を維持するロジック ===
        # 1. 変化量の計算
        diff = P_cmd - self.P_cmd_prev
        
        # 2. 現在の方向判定 (1: Inflation, -1: Deflation, 0: Hold)
        curr_direction = torch.zeros_like(diff, dtype=torch.long)
        curr_direction[diff > self.deadband] = 1
        curr_direction[diff < -self.deadband] = -1
        
        # 3. 状態変化の検出とラッチ更新
        #    更新条件:
        #    A. 現在動いている (curr != 0)
        #    B. かつ、方向が「直前の有効方向」と異なる (curr != last_valid)
        #       ※ これにより、1 -> 0 -> 1 の場合、last_validは1のままなので更新されない。
        #       ※ 1 -> 0 -> -1 の場合、last_valid(1) != curr(-1) なので更新される。
        
        is_moving = (curr_direction != 0)
        direction_changed = (curr_direction != self.last_valid_direction)
        
        # Start Pressure 更新が必要なインデックス
        update_mask = is_moving & direction_changed
        
        if update_mask.any():
            self.P_start_latch[update_mask] = self.P_state[update_mask].detach()
            # last_valid_direction を更新 (動いている方向のみ)
            self.last_valid_direction[update_mask] = curr_direction[update_mask]
            
        # 次回のために指令値を保存
        self.P_cmd_prev = P_cmd.clone()
        # ===============================================

        # 1. むだ時間 L と 時定数 tau の決定
        if self.use_2d_dynamics and self._tau_2d is not None:
            # P_start_latch (変化開始時の圧力) と P_cmd (目標圧力) でテーブルを引く
            tau_now = interp2d_bilinear(self._p_axis_2d, self._p_axis_2d, self._tau_2d, 
                                        x_query=P_cmd, y_query=self.P_start_latch)
            L_cmd   = interp2d_bilinear(self._p_axis_2d, self._p_axis_2d, self._dead_2d, 
                                        x_query=P_cmd, y_query=self.P_start_latch)
            
        elif self.use_table_i:
            if self._tau_x is None:
                tau_now, L_cmd = tau_L_from_pressure(P_cmd)
            else:
                tau_now = interp1d_clamp_torch(self._tau_x, self._tau_y, P_cmd)
                L_cmd = torch.full_like(P_cmd, self.dead_time)
        else:
            tau_now = torch.full_like(P_cmd, self.tau)
            L_cmd   = torch.full_like(P_cmd, self.dead_time)

        # 2. 遅れ実行
        P_delayed = self.delay.step(P_cmd, L_cmd)

        # 3. 一次遅れ
        self.last_tau = torch.clamp(tau_now, min=1e-6)
        self.P_state = first_order_lag(P_delayed, self.P_state, self.last_tau, self.dt)
        
        return self.P_state