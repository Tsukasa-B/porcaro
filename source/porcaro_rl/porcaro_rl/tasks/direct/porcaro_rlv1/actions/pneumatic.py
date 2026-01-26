# source/porcaro_rl/porcaro_rl/tasks/direct/porcaro_rlv1/actions/pneumatic.py
from __future__ import annotations
import math
import torch
from typing import Sequence

# =========================================================
#  Physical Parameters (Single Source of Truth)
# =========================================================

# Table I (Legacy 1D)
P_TAB   = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=torch.float32)
TAU_TAB = torch.tensor([0.043, 0.045, 0.060, 0.066, 0.094, 0.131], dtype=torch.float32)
L_TAB   = torch.tensor([0.038, 0.035, 0.032, 0.030, 0.023, 0.023], dtype=torch.float32)

# High-Fidelity 2D Tables (moved from pam.py)
# ... (既存の TAU_TABLE_2D_DATA, DEAD_TABLE_2D_DATA はそのまま維持) ...
TAU_TABLE_2D_DATA = [
    [0.0010, 0.0200, 0.0404, 0.0404, 0.0808, 0.0909, 0.1313],
    [0.0908, 0.0010, 0.0404, 0.0405, 0.0505, 0.0908, 0.0808],
    [0.0404, 0.1313, 0.0010, 0.0913, 0.0404, 0.0811, 0.0808],
    [0.0404, 0.0808, 0.0910, 0.0010, 0.0809, 0.0403, 0.0808],
    [0.0909, 0.0909, 0.0404, 0.1717, 0.0010, 0.0914, 0.0808],
    [0.0404, 0.0909, 0.0406, 0.0404, 0.0404, 0.0010, 0.0404],
    [0.0404, 0.0909, 0.0505, 0.0404, 0.0505, 0.0200, 0.0010],
]

DEAD_TABLE_2D_DATA = [
    [0.0010, 0.0809, 0.0908, 0.0404, 0.0404, 0.0807, 0.0404],
    [0.0010, 0.0010, 0.0505, 0.0505, 0.0808, 0.0404, 0.0908],
    [0.0809, 0.0808, 0.0010, 0.0010, 0.0911, 0.0505, 0.0505],
    [0.0404, 0.0910, 0.1313, 0.0010, 0.0010, 0.0807, 0.0404],
    [0.0404, 0.0808, 0.0808, 0.0010, 0.0010, 0.0405, 0.0909],
    [0.0808, 0.0808, 0.0913, 0.0808, 0.0909, 0.0010, 0.0810],
    [0.0809, 0.0808, 0.0813, 0.0908, 0.0807, 0.1313, 0.0010],
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

# ... (_tabs_on, interp1d_clamp_torch, interp2d_bilinear はそのまま維持) ...
def _tabs_on(P: torch.Tensor):
    """P の device / dtype に合わせたテーブルを返す (Legacy support)"""
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
def interp2d_bilinear(x_grid, y_grid, z_data, x_query, y_query):
    # (既存の実装そのまま)
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

    v00 = z_data[y0, x0]; v01 = z_data[y0, x1]
    v10 = z_data[y1, x0]; v11 = z_data[y1, x1]

    r0 = v00 * (1 - wx) + v01 * wx
    r1 = v10 * (1 - wx) + v11 * wx
    val = r0 * (1 - wy) + r1 * wy
    return val

# --- [New] Model A 用のパラメータ計算関数を追加 ---
def get_dynamics_params_model_a(
    P_cmd: torch.Tensor, 
    tau_const: float, 
    L_axis: torch.Tensor, 
    L_values: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Model A (Baseline) 用のパラメータ計算
    - Tau: 定数 (tau_const)
    - L (Deadtime): P_cmd に基づく 1D テーブル参照
    """
    # Tauはバッチサイズに合わせて定数を拡張
    tau = torch.full_like(P_cmd, tau_const)
    
    # Lは1D補間
    L = interp1d_clamp_torch(L_axis.to(P_cmd.device), L_values.to(P_cmd.device), P_cmd)
    
    return tau, L

@torch.no_grad()
def tau_L_from_pressure(P, zero_mode="clamp"):
    # (既存の実装そのまま)
    P = P if isinstance(P, torch.Tensor) else torch.as_tensor(P, dtype=torch.float32)
    XP, TAU, L = _tabs_on(P)
    P = torch.nan_to_num(P, nan=0.0)
    Pq  = torch.clamp(P, XP[0], XP[-1])
    tau = interp1d_clamp_torch(XP, TAU, Pq)
    L   = interp1d_clamp_torch(XP, L,   Pq)
    return tau, L

# ... (first_order_lag, FractionalDelay はそのまま維持) ...
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
    # (既存の実装そのまま)
    def __init__(self, dt: float, L_max: float = 0.20): 
        self.dt = float(dt)
        self.K  = int(math.ceil(L_max / self.dt) + 5)
        self.buf = None
        self.wp  = 0
    # ... (rest of methods step, reset, reset_idx are same as your provided code) ...
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
        if self.buf is None: self.reset(x.shape, x.device)
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
            B = x.shape[0]
            # Advanced indexing (copied from your provided code)
            idx0 = (self.wp - M) % self.K
            idx1 = (self.wp - M - 1) % self.K
            if x.ndim == 2: # [Batch, Channel]
                b_idx = torch.arange(x.shape[0], device=x.device).unsqueeze(1).expand_as(x)
                c_idx = torch.arange(x.shape[1], device=x.device).unsqueeze(0).expand_as(x)
                out0 = self.buf[idx0, b_idx, c_idx]
                out1 = self.buf[idx1, b_idx, c_idx]
            else: # [Batch]
                b_idx = torch.arange(x.shape[0], device=x.device)
                out0 = self.buf[idx0, b_idx]
                out1 = self.buf[idx1, b_idx]
            out  = (1.0 - mu) * out0 + mu * out1
        self.wp = (self.wp + 1) % self.K
        return out