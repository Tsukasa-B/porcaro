# source/porcaro_rl/porcaro_rl/tasks/direct/porcaro_rlv1/actions/pam.py
from __future__ import annotations
import math
import torch
import torch.nn as nn
import csv
from .pneumatic import(
    tau_L_from_pressure,
    first_order_lag,
    FractionalDelay,
    interp1d_clamp_torch,
    interp2d_bilinear,
    get_2d_tables,
)

# =========================================================
#  Contraction & Force Logic (ここに集約)
# =========================================================

@torch.no_grad()
def calculate_geometric_contraction(theta: torch.Tensor, theta_t_deg: float, r: float, L: float, sign: float = 1.0) -> torch.Tensor:
    """
    幾何学的な収縮率（符号付き）を計算する基本関数
    epsilon = (r / L) * sign * (theta - theta_t)
    
    Args:
        sign: +1.0 (屈曲/F,G) or -1.0 (背屈/DF)
    """
    theta_rad = torch.deg2rad(theta) if theta.numel() > 0 else theta
    theta_t_rad = math.radians(theta_t_deg)
    
    # 符号付き変位 (正なら収縮方向、負なら伸展/たるみ方向)
    delta_L = r * sign * (theta_rad - theta_t_rad)
    epsilon = delta_L / L
    return epsilon


@torch.no_grad()
def calculate_effective_contraction(theta_deg: torch.Tensor, theta_t_deg: float, r: float, L0: float, 
                                    slack_offset: float, pressure: torch.Tensor = 0.0, 
                                    shrink_gain: float = 0.0, sign: float = 1.0) -> torch.Tensor:
    """
    [Model B用] 有効収縮率の計算
    - オフセット（たるみ）補正を含む
    - 圧力による収縮ゲイン補正を含む
    - 値はクランプせず、負の値（たるみ）をそのまま返す
    """
    # 1. 幾何学的収縮率 (符号付き)
    eps_geo = calculate_geometric_contraction(theta_deg, theta_t_deg, r, L0, sign)
    
    # 2. オフセット補正
    # slack_offset(正) = 初期のたるみ量。これが引かれることで、より縮まないと力が生まれない(0を超えない)ようになる。
    # pressure * shrink_gain = 加圧による収縮アシスト（オプション）
    dynamic_offset = slack_offset - (shrink_gain * pressure)
    
    eps_eff = eps_geo - dynamic_offset
    return eps_eff


@torch.jit.script
def apply_soft_engagement(force: torch.Tensor, epsilon: torch.Tensor, smoothness: float = 100.0) -> torch.Tensor:
    """
    [修正版v3] 正しいSoft Engagement
    - epsilon <= 0 (伸び/張っている): Mask = 1.0 (そのまま)
    - epsilon > 0  (縮み/たるみ): epsilonが増えるほど Mask -> 0.0 になる
    """
    # 1. 伸びている時(負)は常に1.0
    # 2. たるんでいる時(正)は、0付近なら通し、離れると0にする
    
    # たるみ量 (正の値)
    slack_amount = torch.clamp(epsilon, min=0.0)
    
    # たるみが大きいほど x は大きくなる
    x = slack_amount * smoothness
    
    # x=0(張り始め) -> mask=1.0
    # x=1(たるみ大) -> mask=0.0
    # ReLU(1-x) を使って線形減衰、あるいはSigmoid等
    mask_val = torch.clamp(1.0 - x, min=0.0)
    
    # 2乗でスムーズに立ち上がり (0付近の連続性確保)
    mask_val = mask_val * mask_val
    
    return force * mask_val


# =========================================================
#  Map Classes (変更なし)
# =========================================================

class PamForceMap(nn.Module):
    """F(P,h) を表の双一次補間で返す。P[MPa], h[無次元]"""
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
        # マップ参照時は範囲外をクランプする（補間のため）
        P_in = torch.clamp(P_in, self.P[0], self.P[-1])
        h_in = torch.clamp(h_in, self.h[0], self.h[-1])
        return interp2d_bilinear(self.P, self.h, self.F.T, P_in, h_in)


class H0Map(nn.Module):
    """h0(P): 力が0のときの収縮率を返す"""
    def __init__(self, P_axis, h0_axis):
        super().__init__()
        self.register_buffer('P', torch.as_tensor(P_axis, dtype=torch.float32))
        self.register_buffer('h0', torch.as_tensor(h0_axis, dtype=torch.float32))

    @staticmethod
    def from_csv(path: str):
        import csv
        P_axis, h0_axis = [], []
        with open(path, newline="") as f:
            rdr = csv.reader(f)
            rows = list(rdr)
        start = 1 if rows and not rows[0][0].replace('.','',1).isdigit() else 0
        for r in rows[start:]:
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
def Fpam_quasi_static(P: torch.Tensor, h: torch.Tensor, N: float = 1200.0, Pmax: float = 0.6) -> torch.Tensor:
    # 簡易モデル計算用
    h0 = 0.25 * (P / Pmax).clamp(0,1)
    eff = torch.clamp(h0 - h, min=0.0)
    return N * P * eff


class PAMChannel:
    """空気圧ダイナミクスを管理するクラス"""
    def __init__(self, dt_ctrl: float, tau: float = 0.09, dead_time: float = 0.03, Pmax: float = 0.6,
                 tau_lut: tuple[list[float], list[float]] | None = None,
                 use_table_i: bool = True):
    
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

        self._L_prev = None
        self.last_tau = None 
        
        # 1D LUT (互換用)
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
        self._L_prev  = None
        
        if self.use_2d_dynamics:
            self._tau_2d, self._dead_2d, self._p_axis_2d = get_2d_tables(dev)
        if self._tau_x is not None:
            self._tau_x = self._tau_x.to(dev)
            self._tau_y = self._tau_y.to(dev)

    @torch.no_grad()
    def reset_idx(self, env_ids: torch.Tensor):
        if self.P_state is not None:
            self.P_state[env_ids] = 0.0
        if self.last_tau is not None:
            self.last_tau[env_ids] = float(self.tau)
        self.delay.reset_idx(env_ids)

    @torch.no_grad()
    def step(self, P_cmd: torch.Tensor) -> torch.Tensor:
        P_cmd = torch.clamp(P_cmd, 0.0, self.Pmax)
        
        # 1. パラメータ決定
        if self.use_2d_dynamics and self._tau_2d is not None:
            P_curr = self.P_state if self.P_state is not None else torch.zeros_like(P_cmd)
            tau_now = interp2d_bilinear(self._p_axis_2d, self._p_axis_2d, self._tau_2d, x_query=P_cmd, y_query=P_curr)
            L_cmd   = interp2d_bilinear(self._p_axis_2d, self._p_axis_2d, self._dead_2d, x_query=P_cmd, y_query=P_curr)
        elif self.use_table_i and self._tau_x is not None:
            tau_now = interp1d_clamp_torch(self._tau_x, self._tau_y, P_cmd)
            L_cmd = torch.full_like(P_cmd, self.dead_time)
        else:
            tau_now = torch.full_like(P_cmd, self.tau)
            L_cmd   = torch.full_like(P_cmd, self.dead_time)

        # 2. 遅れ計算
        P_delayed = self.delay.step(P_cmd, L_cmd)
        if self.P_state is None: self.P_state = P_delayed.clone()

        self.last_tau = torch.clamp(tau_now, min=1e-6)
        self.P_state = first_order_lag(P_delayed, self.P_state, self.last_tau, self.dt)
        return self.P_state