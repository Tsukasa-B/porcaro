# porcaro_rl/utils/pneumatic.py
from __future__ import annotations
import math
import torch
from typing import Sequence

# ---- Table I（旧来の1次元データ）：互換性のため残存 ----
P_TAB   = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=torch.float32)
TAU_TAB = torch.tensor([0.043, 0.045, 0.060, 0.066, 0.094, 0.131], dtype=torch.float32)
L_TAB   = torch.tensor([0.038, 0.035, 0.032, 0.030, 0.023, 0.023], dtype=torch.float32)

def _tabs_on(P: torch.Tensor):
    """P の device / dtype に合わせたテーブルを返す"""
    return (
        P_TAB.to(device=P.device, dtype=P.dtype),
        TAU_TAB.to(device=P.device, dtype=P.dtype),
        L_TAB.to(device=P.device, dtype=P.dtype),
    )

@torch.no_grad()
def interp1d_clamp_torch(xp: torch.Tensor, fp: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """1次補間（端はクランプ）。xp昇順・1D"""
    x = x.to(xp.dtype).to(xp.device)
    
    # NaN対策と範囲クランプ
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
    """
    双線形補間 (Bilinear Interpolation)
    x_grid: 列方向 (Target Pressure) の軸
    y_grid: 行方向 (Current Pressure) の軸
    z_data: [H, W] のデータ (H=len(y_grid), W=len(x_grid))
    x_query: Target Pressure (P_cmd)
    y_query: Current Pressure (P_curr)
    """
    # テンソルのデバイス/型合わせ
    dev = x_query.device
    dtype = x_query.dtype
    
    # データが別デバイスにある場合の転送
    if x_grid.device != dev: x_grid = x_grid.to(dev, dtype=dtype)
    if y_grid.device != dev: y_grid = y_grid.to(dev, dtype=dtype)
    if z_data.device != dev: z_data = z_data.to(dev, dtype=dtype)
    
    # NaN対策
    x_q = torch.nan_to_num(x_query, nan=float(x_grid[0])).clamp(min=float(x_grid[0]), max=float(x_grid[-1]))
    y_q = torch.nan_to_num(y_query, nan=float(y_grid[0])).clamp(min=float(y_grid[0]), max=float(y_grid[-1]))

    # --- x軸 (Target) のインデックス特定 ---
    # 等間隔グリッドを仮定して高速化 (0.0~0.6, step=0.1)
    # 汎用的にやるなら bucketize ですが、ここでは計算コスト重視で直接計算
    dx = x_grid[1] - x_grid[0]
    x_idx_f = (x_q - x_grid[0]) / dx
    x0 = x_idx_f.floor().long().clamp(0, len(x_grid)-2)
    x1 = x0 + 1
    wx = x_idx_f - x0 # 重み

    # --- y軸 (Current) のインデックス特定 ---
    dy = y_grid[1] - y_grid[0]
    y_idx_f = (y_q - y_grid[0]) / dy
    y0 = y_idx_f.floor().long().clamp(0, len(y_grid)-2)
    y1 = y0 + 1
    wy = y_idx_f - y0 # 重み

    # 4近傍の値を取得 z_data[row, col] -> z_data[y, x]
    v00 = z_data[y0, x0] # (y0, x0)
    v01 = z_data[y0, x1] # (y0, x1)
    v10 = z_data[y1, x0] # (y1, x0)
    v11 = z_data[y1, x1] # (y1, x1)

    # 補間計算
    # 1. x方向
    r0 = v00 * (1 - wx) + v01 * wx # 上段
    r1 = v10 * (1 - wx) + v11 * wx # 下段
    
    # 2. y方向
    val = r0 * (1 - wy) + r1 * wy
    
    return val

@torch.no_grad()
def tau_L_from_pressure(P, zero_mode="clamp"):
    # 旧関数（互換性維持）
    P = P if isinstance(P, torch.Tensor) else torch.as_tensor(P, dtype=torch.float32)
    XP, TAU, L = _tabs_on(P)
    
    P = torch.nan_to_num(P, nan=0.0)

    if zero_mode == "extrapolate":
        # (簡略化のため中略、元のロジック通り)
        Pq  = torch.clamp(P, XP[0], XP[-1])
        tau = interp1d_clamp_torch(XP, TAU, Pq)
        L   = interp1d_clamp_torch(XP, L,   Pq)
    else:
        Pq  = torch.clamp(P, XP[0], XP[-1])
        tau = interp1d_clamp_torch(XP, TAU, Pq)
        L   = interp1d_clamp_torch(XP, L,   Pq)
    return tau, L

@torch.no_grad()
def first_order_lag(u_now: torch.Tensor, x_prev: torch.Tensor, tau, dt: float) -> torch.Tensor:
    """一次遅れ x'=(u-x)/τ"""
    if isinstance(tau, torch.Tensor):
        tau_t = torch.clamp(tau.to(u_now.dtype).to(u_now.device), min=1e-9)
        alpha = dt / tau_t
        x_next = x_prev + alpha * (u_now - x_prev)
        return torch.where(tau <= 1e-6, u_now, x_next)
    else:
        tau_f = float(tau)
        if tau_f <= 1e-6:
            return u_now
        return x_prev + (dt / tau_f) * (u_now - x_prev)

class FractionalDelay:
    """分数サンプル遅延"""
    def __init__(self, dt: float, L_max: float = 0.20): # 少しバッファ余裕を持たせる
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
            B = x.shape[0]
            cols = torch.arange(B, device=x.device)
            idx0 = (self.wp - M) % self.K
            idx1 = (self.wp - M - 1) % self.K
            out0 = self.buf[idx0, cols]
            out1 = self.buf[idx1, cols]
            out  = (1.0 - mu) * out0 + mu * out1

        self.wp = (self.wp + 1) % self.K
        return out