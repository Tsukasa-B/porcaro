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
    interp2d_bilinear, # <--- 追加
)

# --- ユーザー提供の高精度データ (Raw Lists) ---
# Rows: Start Pressure (0.0 to 0.6)
# Cols: Target Pressure (0.0 to 0.6)
_TAU_TABLE_DATA = [
    [0.0010, 0.0200, 0.0404, 0.0404, 0.0808, 0.0909, 0.1313], # Start=0.0
    [0.0908, 0.0010, 0.0404, 0.0405, 0.0505, 0.0908, 0.0808], # Start=0.1
    [0.0404, 0.1313, 0.0010, 0.0913, 0.0404, 0.0811, 0.0808], # Start=0.2
    [0.0404, 0.0808, 0.0910, 0.0010, 0.0809, 0.0403, 0.0808], # Start=0.3
    [0.0909, 0.0909, 0.0404, 0.1717, 0.0010, 0.0914, 0.0808], # Start=0.4
    [0.0404, 0.0909, 0.0406, 0.0404, 0.0404, 0.0010, 0.0404], # Start=0.5
    [0.0404, 0.0909, 0.0505, 0.0404, 0.0505, 0.0200, 0.0010], # Start=0.6
]

_DEAD_TABLE_DATA = [
    [0.0010, 0.0809, 0.0908, 0.0404, 0.0404, 0.0807, 0.0404], # Start=0.0
    [0.0010, 0.0010, 0.0505, 0.0505, 0.0808, 0.0404, 0.0908], # Start=0.1
    [0.0809, 0.0808, 0.0010, 0.0010, 0.0911, 0.0505, 0.0505], # Start=0.2
    [0.0404, 0.0910, 0.1313, 0.0010, 0.0010, 0.0807, 0.0404], # Start=0.3
    [0.0404, 0.0808, 0.0808, 0.0010, 0.0010, 0.0405, 0.0909], # Start=0.4
    [0.0808, 0.0808, 0.0913, 0.0808, 0.0909, 0.0010, 0.0810], # Start=0.5
    [0.0809, 0.0808, 0.0813, 0.0908, 0.0807, 0.1313, 0.0010], # Start=0.6
]
# ---------------------------------------------

class PamForceMap(nn.Module):
    """F(P,h) を表の双一次補間で返す。P[MPa], h[無次元]"""
    def __init__(self, P_axis, h_axis, F_table):
        super().__init__()
        self.register_buffer('P', torch.as_tensor(P_axis, dtype=torch.float32))
        self.register_buffer('h', torch.as_tensor(h_axis, dtype=torch.float32))
        self.register_buffer('F', torch.as_tensor(F_table, dtype=torch.float32))
        assert self.F.shape == (self.P.numel(), self.h.numel())

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
def contraction_ratio_from_angle(theta: torch.Tensor, theta_t_deg: float, r: float, L: float) -> torch.Tensor:
    theta_t = math.radians(theta_t_deg)
    return (r / L) * torch.abs(theta - theta_t)

def calculate_effective_contraction(theta_deg, theta_t_deg, r, L0, slack_offset):
    theta_rad = torch.deg2rad(theta_deg)
    theta_t_rad = math.radians(theta_t_deg)
    delta_geo = r * torch.abs(theta_rad - theta_t_rad)
    L_geo = L0 - delta_geo
    L_eff = L_geo + slack_offset
    epsilon = (L0 - L_eff) / L0
    return torch.clamp(epsilon, min=0.0)

@torch.no_grad()
def Fpam_quasi_static(P: torch.Tensor, h: torch.Tensor, N: float = 1200.0, Pmax: float = 0.6) -> torch.Tensor:
    h0 = 0.25 * (P / Pmax).clamp(0,1)
    eff = torch.clamp(h0 - h, min=0.0)
    return N * P * eff

class PAMChannel:
    def __init__(self, dt_ctrl: float, tau: float = 0.09, dead_time: float = 0.03, Pmax: float = 0.6,
                 tau_lut: tuple[list[float], list[float]] | None = None,
                 use_table_i: bool = True):
    
        self.dt = dt_ctrl
        self.dead_time = float(dead_time)
        self.tau = float(tau)
        self.Pmax = float(Pmax)
        self.delay = FractionalDelay(dt_ctrl, L_max=0.20) # 2Dテーブルの最大遅延に対応できるようバッファ拡張
        self.P_state = None
        
        # --- モード選択 ---
        self.use_2d_dynamics = True # デフォルトで高精度2Dモードを使用
        self.use_table_i = use_table_i # 1Dテーブル (後方互換用)
        
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
        
        # --- 2Dテーブルの初期化 (Device転送) ---
        if self.use_2d_dynamics:
            self._tau_2d = torch.tensor(_TAU_TABLE_DATA, dtype=torch.float32, device=dev)
            self._dead_2d = torch.tensor(_DEAD_TABLE_DATA, dtype=torch.float32, device=dev)
            self._p_axis_2d = torch.linspace(0.0, 0.6, 7, dtype=torch.float32, device=dev)
            
        if self._tau_x is not None:
            self._tau_x = self._tau_x.to(dev)
            self._tau_y = self._tau_y.to(dev)

    @torch.no_grad()
    def reset_idx(self, env_ids: torch.Tensor | Sequence[int]):
        if self.P_state is not None:
            self.P_state[env_ids] = 0.0
        if self.last_tau is not None:
            self.last_tau[env_ids] = float(self.tau)
        if self._L_prev is not None:
            self._L_prev[env_ids] = 0.0
        self.delay.reset_idx(env_ids)

    @torch.no_grad()
    def step(self, P_cmd: torch.Tensor) -> torch.Tensor:
        P_cmd = torch.clamp(P_cmd, 0.0, self.Pmax)
        
        # 1. むだ時間 L と 時定数 tau の決定
        if self.use_2d_dynamics and self._tau_2d is not None:
            # === 高精度 2D モード ===
            # 現在の圧力(Start) と 指令圧力(Target) から補間
            P_curr = self.P_state if self.P_state is not None else torch.zeros_like(P_cmd)
            
            # Start(Row)=P_curr, Target(Col)=P_cmd
            # interp2d_bilinear(x_grid, y_grid, z_data, x_query, y_query)
            tau_now = interp2d_bilinear(self._p_axis_2d, self._p_axis_2d, self._tau_2d, x_query=P_cmd, y_query=P_curr)
            L_cmd   = interp2d_bilinear(self._p_axis_2d, self._p_axis_2d, self._dead_2d, x_query=P_cmd, y_query=P_curr)
            
        elif self.use_table_i:
            # === 旧来の 1D モード ===
            if self._tau_x is None:
                tau_now, L_cmd = tau_L_from_pressure(P_cmd)
            else:
                tau_now = interp1d_clamp_torch(self._tau_x, self._tau_y, P_cmd)
                L_cmd = torch.full_like(P_cmd, self.dead_time)
        else:
            # === 固定値モード ===
            tau_now = torch.full_like(P_cmd, self.tau)
            L_cmd   = torch.full_like(P_cmd, self.dead_time)

        # 2. むだ時間の平滑化 (オプション)
        # 急激なLの変化はバッファ読み出し位置を飛ばすので少し緩和してもよいが、今回は直値
        L_now = L_cmd 
        self._L_prev = L_now

        # 3. 遅れ実行 (Fractional Delay)
        P_delayed = self.delay.step(P_cmd, L_now)

        # 4. 一次遅れ (Low Pass Filter)
        if self.P_state is None or self.P_state.shape != P_cmd.shape:
            self.P_state = P_delayed.clone()

        self.last_tau = torch.clamp(tau_now, min=1e-6)
        self.P_state = first_order_lag(P_delayed, self.P_state, self.last_tau, self.dt)
        
        return self.P_state