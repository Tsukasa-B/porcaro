# porcaro_rl/utils/pam.py
from __future__ import annotations
import math
import torch
import csv
from .pneumatic import(
    tau_L_from_pressure,    # 圧力->(τ，L)
    first_order_lag,        # 一次遅れ（統一）
    FractionalDelay,        # 可変・分数サンプル遅延
    interp1d_clamp_torch,   # 1D補間(必要なら)
)

class PamForceMap:
    """F(P,h) を表の双一次補間で返す。P[MPa], h[無次元]（hが%なら自動で/100）"""
    def __init__(self, P_axis, h_axis, F_table):
        self.P = torch.as_tensor(P_axis, dtype=torch.float32)   # (M,)
        self.h = torch.as_tensor(h_axis, dtype=torch.float32)   # (N,)
        self.F = torch.as_tensor(F_table, dtype=torch.float32)  # (M,N)
        assert self.F.shape == (self.P.numel(), self.h.numel())

    @staticmethod
    def from_csv(path: str):
        with open(path, newline="") as f:
            rows = list(csv.reader(f))
        # 1行目: ["", h1, h2, ...] / 2行目以降: [P, F...]
        h_axis = [float(x) for x in rows[0][1:] if x != ""]
        P_axis, F = [], []
        for r in rows[1:]:
            if not r or r[0] == "":
                continue
            P_axis.append(float(r[0]))
            F.append([float(x) for x in r[1:1+len(h_axis)]])
        # h が %（最大が1超）なら 0..1 へ自動変換
        if max(h_axis) > 1.0:
            h_axis = [x / 100.0 for x in h_axis]
        return PamForceMap(P_axis, h_axis, F)

    @torch.no_grad()
    def __call__(self, P: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        P = P.squeeze(-1); h = h.squeeze(-1)
        P_axis = self.P.to(P.device); h_axis = self.h.to(P.device); Ftab = self.F.to(P.device)
        # 軸内クランプ
        P = torch.clamp(P, P_axis[0], P_axis[-1])
        h = torch.clamp(h, h_axis[0], h_axis[-1])
        # 区間インデックス
        i1 = torch.searchsorted(P_axis, P) - 1
        i1 = torch.clamp(i1, 0, P_axis.numel()-2); i2 = i1 + 1
        j1 = torch.searchsorted(h_axis, h) - 1
        j1 = torch.clamp(j1, 0, h_axis.numel()-2); j2 = j1 + 1
        P1, P2 = P_axis[i1], P_axis[i2]
        h1, h2 = h_axis[j1], h_axis[j2]
        t = (P - P1) / torch.clamp(P2 - P1, min=1e-12)
        u = (h - h1) / torch.clamp(h2 - h1, min=1e-12)
        F11 = Ftab[i1, j1]; F21 = Ftab[i2, j1]
        F12 = Ftab[i1, j2]; F22 = Ftab[i2, j2]
        return (1-t)*(1-u)*F11 + t*(1-u)*F21 + (1-t)*u*F12 + t*u*F22
    
# --- 追加: 無負荷収縮率 h0(P) の1D補間 ---
class H0Map:
    """h0(P): 力が0のときの収縮率 [%でもOK→自動で/100] を1D線形補間で返す"""
    def __init__(self, P_axis, h0_axis):
        self.P  = torch.as_tensor(P_axis,  dtype=torch.float32)  # (M,)
        self.h0 = torch.as_tensor(h0_axis, dtype=torch.float32)  # (M,)
        assert self.P.numel() == self.h0.numel()

    @staticmethod
    def from_csv(path: str):
        # 想定: 2列CSV（ヘッダ有無どちらも可）: P, h0
        import csv
        P_axis, h0_axis = [], []
        with open(path, newline="") as f:
            rdr = csv.reader(f)
            rows = list(rdr)
        # 先頭行がヘッダっぽければ飛ばす
        start = 1 if rows and not rows[0][0].replace('.','',1).isdigit() else 0
        for r in rows[start:]:
            if len(r) < 2: 
                continue
            try:
                P_axis.append(float(r[0]))
                h0_axis.append(float(r[1]))
            except Exception:
                pass
        # h0 が % 表記なら 0..1 へ
        if max(h0_axis) > 1.0:
            h0_axis = [x / 100.0 for x in h0_axis]
        return H0Map(P_axis, h0_axis)

    @torch.no_grad()
    def __call__(self, P: torch.Tensor) -> torch.Tensor:
        P = P.squeeze(-1)
        P_axis = self.P.to(P.device); h0_axis = self.h0.to(P.device)
        # 端でクランプ
        P = torch.clamp(P, P_axis[0], P_axis[-1])
        # 区間探索
        i1 = torch.searchsorted(P_axis, P) - 1
        i1 = torch.clamp(i1, 0, P_axis.numel()-2); i2 = i1 + 1
        P1, P2 = P_axis[i1], P_axis[i2]
        w = (P - P1) / torch.clamp(P2 - P1, min=1e-12)
        return (1 - w) * h0_axis[i1] + w * h0_axis[i2]

@torch.no_grad()
def contraction_ratio_from_angle(theta: torch.Tensor, theta_t_deg: float, r: float, L: float) -> torch.Tensor:
    # 角度はラジアン入力なので、円弧長 = r * Δθ を L で正規化
    theta_t = math.radians(theta_t_deg)
    return (r / L) * torch.abs(theta - theta_t)

def calculate_effective_contraction(
    theta_deg: torch.Tensor,
    theta_t_deg: float,
    r: float,
    L0: float,
    slack_offset: float | torch.Tensor
) -> torch.Tensor:
    """有効収縮率 epsilon_eff を計算する (Slack補正あり)"""
    
    # 1. 幾何学的長さの計算 (L_geo)
    theta_rad = torch.deg2rad(theta_deg)
    theta_t_rad = math.radians(theta_t_deg)
    
    # 修正: 絶対値 (abs) をとる
    # これにより、アンカー(theta_t)から「どちらに回転しても」ワイヤーが引き出される挙動を再現
    delta_geo = r * torch.abs(theta_rad - theta_t_rad)
    
    # 現在の幾何学的長さ (アンカーから離れるほど、ワイヤー有効長は短くなる = 収縮扱い)
    # ※元の contraction_ratio_from_angle と同じ幾何特性
    L_geo = L0 - delta_geo
    
    # 2. 有効長の計算 (L_eff)
    # Slackがある場合、実際の「張り」は弱くなる -> 長さが増える
    L_eff = L_geo + slack_offset
    
    # 3. 収縮率の計算
    epsilon = (L0 - L_eff) / L0
    
    return torch.clamp(epsilon, min=0.0)


@torch.no_grad()
def h0_of_P_linear(P: torch.Tensor, Pmax: float = 0.6, h0_max: float = 0.25) -> torch.Tensor:
    """
    無負荷収縮率 h0(P) の簡易近似（線形）。McKibbenで 0.2〜0.3 程度が典型。
    後でFESTOの静特性から置き換える前提の暫定。
    """
    P = torch.clamp(P, 0.0, Pmax)
    return h0_max * (P / Pmax)

@torch.no_grad()
def Fpam_quasi_static(P: torch.Tensor, h: torch.Tensor,
                      N: float = 1200.0, Pmax: float = 0.6, h0_max: float = 0.25) -> torch.Tensor:
    """
    簡易静特性: F = N * P * max(h0(P) - h, 0)
    ・Pは[MPa]想定、Nはスケール係数（後で実データにフィット）
    ・ワイヤ弛緩( h > h0 )なら 0 にクリップ
    """
    h0 = h0_of_P_linear(P, Pmax=Pmax, h0_max=h0_max)
    eff = torch.clamp(h0 - h, min=0.0)
    return N * P * eff  # [N]相当のスケール

class PAMChannel:
    """
    1つのPAMチャネル（圧力コマンド -> デッドタイム -> 一次遅れ -> 実圧力）
    """
    def __init__(self, dt_ctrl: float, tau: float = 0.09, dead_time: float = 0.03, Pmax: float = 0.6,
                 tau_lut: tuple[list[float], list[float]] | None = None,
                 dead_lut: tuple[list[float], list[float]] | None = None,
                 use_table_i: bool = True,
                 L_smooth_T: float = 0.0):
    
        self.dt = dt_ctrl
        self.dead_time = float(dead_time)
        self.tau = float(tau)     # フォールバック用（定数）
        self.Pmax = float(Pmax)
        # 可変デッドタイム用（Table I の最大 0.038s に余裕を足す）
        self.delay = FractionalDelay(dt_ctrl, L_max=0.05)
        self.P_state = None
        self.use_table_i = use_table_i
        self.L_smooth_T = float(L_smooth_T)
        self._L_prev = None
        # 追加：LUT（存在すれば使用）
        self._tau_x, self._tau_y = None, None
        if tau_lut is not None:
            x, y = tau_lut
            self._tau_x = torch.tensor([x[0]] + x, dtype=torch.float32) if x[0] > 0.0 else torch.tensor(x, dtype=torch.float32)
            self._tau_y = torch.tensor([y[0]] + y, dtype=torch.float32) if x[0] > 0.0 else torch.tensor(y, dtype=torch.float32)
        self._dead_x, self._dead_y = None, None   # 拡張用（今回は未使用）
        self.last_tau = None  # Telemetry

    def reset(self, n_envs: int, device: str | torch.device):
        dev = torch.device(device) if not isinstance(device, torch.device) else device
        z = torch.zeros(n_envs, device=dev, dtype=torch.float32)
        # 出力状態と遅延線をゼロでプリチャージ
        self.P_state = z.clone()
        self.delay.reset(z.shape, dev)
        # Telemetry用（任意）
        self.last_tau = torch.full_like(z, float(self.tau))
        self._L_prev  = None


    @torch.no_grad()
    def reset_idx(self, env_ids: torch.Tensor | Sequence[int]):
        """指定した env_ids の内部状態（圧力・遅延バッファ）をリセット"""
        if self.P_state is not None:
            self.P_state[env_ids] = 0.0
        
        if self.last_tau is not None:
            # 圧力ゼロ相当の時定数（＝固定値 or LUTの最小値）に戻す
            fallback_tau = float(self.tau)
            if self._tau_y is not None:
                fallback_tau = self.last_tau.new_tensor(self._tau_y[0])
            self.last_tau[env_ids] = fallback_tau

        if self._L_prev is not None:
            self._L_prev[env_ids] = 0.0 # L(P=0) は 0 と仮定

        # 遅延バッファもリセット
        self.delay.reset_idx(env_ids)

    @torch.no_grad()
    def step(self, P_cmd: torch.Tensor) -> torch.Tensor:
        P_cmd = torch.clamp(P_cmd, 0.0, self.Pmax)
        # --- 無駄時間 L は「指令圧」から決定（Table I or LUT） ---
        if self.use_table_i and self._dead_x is None:
            _, L_cmd = tau_L_from_pressure(P_cmd)    # 指令圧に応じたL
        else:
            # 既存 dead_lut を使う構成があればこちらに分岐
            if self._dead_x is not None:
                L_cmd = interp1_linear(P_cmd, self._dead_x.to(P_cmd.device), self._dead_y.to(P_cmd.device))
            else:
                L_cmd = P_cmd.new_full(P_cmd.shape, float(self.dead_time))  # フォールバック
        # オプション：Lのスムージングでガタつきを抑制
        if self.L_smooth_T > 1e-9:
            beta = self.dt / self.L_smooth_T
            self._L_prev = L_cmd if self._L_prev is None else (self._L_prev + beta * (L_cmd - self._L_prev))
            L_now = self._L_prev
        else:
            L_now = L_cmd
        # 分数サンプル遅延で純遅延を生成
        P_delayed = self.delay.step(P_cmd, L_now)

        if self.P_state is None or self.P_state.shape != P_cmd.shape or self.P_state.device != P_cmd.device:
            self.P_state = P_delayed.clone()

        # --- 可変時定数 τ(P)（遅延後のレベルで決定） ---
        if self.use_table_i and self._tau_x is None:
            tau_now, _ = tau_L_from_pressure(P_delayed)
        else:
            tau_now = (interp1_linear(P_delayed, self._tau_x.to(P_delayed.device), self._tau_y.to(P_delayed.device))
                       if self._tau_x is not None else torch.full_like(P_delayed, float(self.tau)))
        self.last_tau = torch.clamp(tau_now, min=1e-6)

        # --- 一次遅れ（pam_core に統一） ---
        self.P_state = first_order_lag(P_delayed, self.P_state, self.last_tau, self.dt)
        return self.P_state

    
# --- 追加：1次元線形補間（torch版） ---
@torch.no_grad()
def interp1_linear(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """xp: 昇順 (M,), fp: (M,)。x と同device/shapeで線形補間"""
    x  = x.squeeze(-1)
    xp = xp.to(x.device); fp = fp.to(x.device)
    x  = torch.clamp(x, xp[0], xp[-1])
    i1 = torch.searchsorted(xp, x) - 1
    i1 = torch.clamp(i1, 0, xp.numel()-2)
    i2 = i1 + 1
    x1, x2 = xp[i1], xp[i2]
    y1, y2 = fp[i1], fp[i2]
    w = (x - x1) / torch.clamp(x2 - x1, min=1e-12)
    return (1 - w) * y1 + w * y2