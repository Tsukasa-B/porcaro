# porcaro_rl/utils/neumatic.py
from __future__ import annotations
import math
import torch
from typing import Sequence

# ---- Table I（MPa→秒）：必要に応じて書き換えOK ----
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
    idx = torch.bucketize(x, xp).clamp(1, xp.numel()-1)
    x0, x1 = xp[idx-1], xp[idx]
    f0, f1 = fp[idx-1], fp[idx]
    t = (x - x0) / (x1 - x0 + 1e-12)
    return f0 + t * (f1 - f0)

@torch.no_grad()
def tau_L_from_pressure(P, zero_mode="clamp"):
    P = P if isinstance(P, torch.Tensor) else torch.as_tensor(P, dtype=torch.float32)
    XP, TAU, L = _tabs_on(P)  # ← ここで device/dtype を揃える
    if zero_mode == "extrapolate":
        slope_tau = (TAU[1]-TAU[0])/(XP[1]-XP[0])
        slope_L   = (L[1]  -L[0])  /(XP[1]-XP[0])
        zero = XP.new_tensor(0.0)
        tau0 = TAU[0] - slope_tau*(XP[0]-zero)
        L0   = L[0]   - slope_L  *(XP[0]-zero)
        P_ext   = torch.cat([zero[None], XP])
        TAU_ext = torch.cat([tau0[None], TAU])
        L_ext   = torch.cat([L0[None],   L])
        tau = interp1d_clamp_torch(P_ext, TAU_ext, P)
        L   = interp1d_clamp_torch(P_ext, L_ext,   P)
    else:
        Pq  = torch.clamp(P, XP[0], XP[-1])   # ← CPU 定数を使わない
        tau = interp1d_clamp_torch(XP, TAU, Pq)
        L   = interp1d_clamp_torch(XP, L,   Pq)
    return tau, L

@torch.no_grad()
def first_order_lag(u_now: torch.Tensor, x_prev: torch.Tensor, tau, dt: float) -> torch.Tensor:
    """一次遅れ x'=(u-x)/τ（前進オイラー）。tauはfloat/TensorどちらでもOK"""
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
    """分数サンプル遅延（1次線形補間）。x: [B] or [] を想定"""
    def __init__(self, dt: float, L_max: float = 0.05):
        self.dt = float(dt)
        self.K  = int(math.ceil(L_max / self.dt) + 3)  # バッファ長
        self.buf = None
        self.wp  = 0

    def reset(self, shape: torch.Size, device):
        self.buf = torch.zeros(self.K, *shape, dtype=torch.float32, device=device)
        self.wp  = 0

    # pneumatic.py の FractionalDelay クラスの *中* に以下を追加

    @torch.no_grad()
    def reset_idx(self, env_ids: torch.Tensor | Sequence[int]):
        """指定した env_ids の遅延バッファ（過去履歴）をゼロクリア"""
        if self.buf is None:
            return  # まだ初期化されていない

        # self.buf の形状は [K, B, ...] (K=バッファ長, B=環境数)
        # B の次元 (dim=1) に対してインデックスを指定する
        if self.buf.ndim >= 2:
            self.buf[:, env_ids] = 0.0
        else:
            # バッチ次元(B)がない場合 (ndim=1, [K])
            # このケースは並列環境では通常発生しないが、念のため
            if isinstance(env_ids, (torch.Tensor, list, tuple)) and len(env_ids) > 0:
                 # 全クリアで代用（部分リセットができないため）
                self.buf.fill_(0.0)

    @torch.no_grad()
    def step(self, x: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        if self.buf is None:
            self.reset(x.shape, x.device)

        D  = torch.clamp(L / self.dt, min=0.0, max=self.K - 2.0)
        M  = torch.floor(D).to(torch.int64)
        mu = (D - M).to(torch.float32)

        self.buf[self.wp] = x  # 書き込み

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