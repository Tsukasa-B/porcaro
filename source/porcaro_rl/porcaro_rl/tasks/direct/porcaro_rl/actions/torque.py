# porcaro_rl/actions/torque.py
from __future__ import annotations
import torch
from .base import ActionController, RobotLike     # IFは既存を流用:contentReference[oaicite:3]{index=3}
from .pam import (
    PAMChannel, contraction_ratio_from_angle, Fpam_quasi_static, PamForceMap, H0Map
)

class TorqueActionController(ActionController):
    def __init__(self,
                 dt_ctrl: float,
                 r: float = 0.014, L: float = 0.150,
                 theta_t_DF_deg: float = 0.0,
                 theta_t_F_deg:  float = -90.0,
                 theta_t_G_deg:  float = -45.0,
                 Pmax: float = 0.8,
                 tau: float = 0.09, dead_time: float = 0.03,
                 N: float = 630.0,
                 force_map_csv: str | None = None,
                 h0_map_csv: str | None = None,
                 use_pressure_dependent_tau: bool = True):

        # === ここで apply が使う全メンバを定義 ===
        self.dt_ctrl = float(dt_ctrl)
        self.r, self.L = float(r), float(L)
        self.theta_t = {"DF": float(theta_t_DF_deg),
                        "F":  float(theta_t_F_deg),
                        "G":  float(theta_t_G_deg)}
        self.Pmax = float(Pmax)          # ← Pmax はここで必ず持つ
        self.N = float(N)

        # 力マップ / h0マップ（CSVがあれば使う）
        self.force_map = PamForceMap.from_csv(force_map_csv) if force_map_csv else None
        self.h0_map    = H0Map.from_csv(h0_map_csv) if h0_map_csv else None

        # 可変時定数 τ(P) のLUT（Table I; 0.1..0.6 MPa）
        tau_lut = None
        if use_pressure_dependent_tau:
            tau_P_axis = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
            tau_vals   = [0.043,0.045,0.060,0.066,0.094,0.131]  # [s]
            tau_lut = (tau_P_axis, tau_vals)

        # 圧力一次遅れチャネル（DF/F/G）
        self.ch_DF = PAMChannel(dt_ctrl, tau=tau, dead_time=dead_time, Pmax=self.Pmax, tau_lut=tau_lut)
        self.ch_F  = PAMChannel(dt_ctrl, tau=tau, dead_time=dead_time, Pmax=self.Pmax, tau_lut=tau_lut)
        self.ch_G  = PAMChannel(dt_ctrl, tau=tau, dead_time=dead_time, Pmax=self.Pmax, tau_lut=tau_lut)

        self._dbg_n = 0  # デバッグ出力回数カウンタ

    # --- デバッグ出力（最初の数回だけ） ---
    def _dbg(self, name, *xs):
        if self._dbg_n >= 10:   # うるさければ 0 に
            return
        parts = []
        for x in xs:
            if torch.is_tensor(x):
                parts.append(f"{x.detach().float().mean().item():.3f}")
            else:
                parts.append(str(x))
        print(f"[DBG] {name}: " + ", ".join(parts))
        self._dbg_n += 1

    def reset(self, n_envs: int, device: str | torch.device):
        self.ch_DF.reset(n_envs, device)
        self.ch_F.reset(n_envs, device)
        self.ch_G.reset(n_envs, device)

    # torque.py の TorqueActionController クラスの *中* に以下を追加
    # (reset メソッドの下あたりに追加すると分かりやすいです)

    def reset_idx(self, env_ids: torch.Tensor | Sequence[int]):
        """指定した env_ids のコントローラ状態をリセット"""
        self.ch_DF.reset_idx(env_ids)
        self.ch_F.reset_idx(env_ids)
        self.ch_G.reset_idx(env_ids)

    @torch.no_grad()
    def apply(self, *, actions: torch.Tensor, q: torch.Tensor,
              joint_ids: tuple[int, int], robot: RobotLike) -> None:
        wid, gid = joint_ids
        assert actions.shape[-1] == 3, "TorqueActionController expects 3-dim action (DF,F,G)."
        n_envs = int(q.shape[0])

        # 余分な軸が混入しても [n_envs] に落とすユーティリティ
        def _env1(x: torch.Tensor) -> torch.Tensor:
            # 例： [n_envs] / [n_envs,1] / [n_envs,K] / [n_envs*K] などを許容
            x = x.reshape(n_envs, -1)
            return x[:, 0].contiguous()  # 先頭列を採用（平均にしたいなら .mean(dim=1) に変更）

        # 1) 行動 → 圧力コマンド [0, Pmax]
        # a = torch.tanh(actions)                            # (-1..1)
        a = actions
        P_cmd = (a + 1.0) * 0.5 * self.Pmax               # (0..Pmax)
        # P_cmd = torch.maximum(P_cmd, torch.full_like(P_cmd, 0.02))
        self._dbg("P_cmd(mean DF,F,G)", P_cmd.mean(dim=0))

        # 2) 一次遅れ＋デッドタイム（各チャネル）
        P_DF, P_F, P_G = self.ch_DF.step(P_cmd[:, 0]), self.ch_F.step(P_cmd[:, 1]), self.ch_G.step(P_cmd[:, 2])
        P_DF, P_F, P_G = _env1(P_DF), _env1(P_F), _env1(P_G)  # ← 正規化
        self._dbg("P_delayed(mean DF,F,G)", P_DF, P_F, P_G)

        # 3) 収縮率 h(θ) は env ベクトルに統一
        theta_w = q[:, wid]; theta_g = q[:, gid]          # [num_envs]
        h_DF = _env1(contraction_ratio_from_angle(theta_w, self.theta_t["DF"], self.r, self.L))
        h_F  = _env1(contraction_ratio_from_angle(theta_w, self.theta_t["F"],  self.r, self.L))
        h_G  = _env1(contraction_ratio_from_angle(theta_g, self.theta_t["G"],  self.r, self.L))
        self._dbg("h(mean DF,F,G)", h_DF, h_F, h_G)

        # 4) F(P,h)（CSVマップが無ければ簡易式で代用）
        if self.force_map is not None:
            F_DF = _env1(self.force_map(P_DF, h_DF))
            F_F  = _env1(self.force_map(P_F,  h_F))
            F_G  = _env1(self.force_map(P_G,  h_G))
        else:
            F_DF = _env1(Fpam_quasi_static(P_DF, h_DF, N=self.N, Pmax=self.Pmax))
            F_F  = _env1(Fpam_quasi_static(P_F,  h_F,  N=self.N, Pmax=self.Pmax))
            F_G  = _env1(Fpam_quasi_static(P_G,  h_G,  N=self.N, Pmax=self.Pmax))
        self._dbg("F(mean DF,F,G)", F_DF, F_F, F_G)

        # 4.5) 弛み判定（h > h0(P) なら 0）
        if self.h0_map is not None:
            h0_DF = _env1(self.h0_map(P_DF)); h0_F = _env1(self.h0_map(P_F)); h0_G = _env1(self.h0_map(P_G))
            self._dbg("h0(mean DF,F,G)", h0_DF, h0_F, h0_G)
            F_DF = torch.where(h_DF <= h0_DF, F_DF, torch.zeros_like(F_DF))
            F_F  = torch.where(h_F  <= h0_F,  F_F,  torch.zeros_like(F_F))
            F_G  = torch.where(h_G  <= h0_G,  F_G,  torch.zeros_like(F_G))

        # 5) 関節トルク印加（ここから置き換え）
        def _env_vec(x):
            x = x.reshape(-1)
            # 期待：numel == n_envs。違う場合は (n_envs, k) と見なして env 次元に畳み込む
            if x.numel() == n_envs:
                return x
            elif x.numel() % n_envs == 0:
                return x.view(n_envs, -1).mean(dim=1)
            else:
                # 最低限クラッシュ回避（先頭 n_envs を使用）
                return x[:n_envs]

        tau_w = _env1(self.r * (F_F - F_DF))
        tau_g = _env1(self.r * F_G)

        # フルサイズのトルク行列を作って該当列に代入
        n_envs = q.shape[0]
        n_dof  = q.shape[1] if q.ndim > 1 else 1
        tau_full = torch.zeros(n_envs, n_dof, device=q.device, dtype=q.dtype)
        tau_full[:, wid] = tau_w
        tau_full[:, gid] = tau_g
        robot.set_joint_effort_target(tau_full)

        # 6) taut は必ず [num_envs] に（余計な軸があれば env 次元で集約）
        def _env_bool(x):
            x = x.reshape(x.shape[0], -1)   # [N, M]
            return x.all(dim=1).float()     # 全要素が張っていれば1, どこか弛めば0
        if self.h0_map is not None:
            taut_DF = _env_bool(h_DF <= h0_DF)
            taut_F  = _env_bool(h_F  <= h0_F)
            taut_G  = _env_bool(h_G  <= h0_G)
        else:
            taut_DF = torch.zeros_like(h_DF)
            taut_F  = torch.zeros_like(h_F)
            taut_G  = torch.zeros_like(h_G)

        self._last_telemetry = {
            "P_cmd": P_cmd.detach(),  # [n_envs,3]
            "P_out": torch.stack([P_DF, P_F, P_G], -1),  # [n_envs,3]
            "F":     torch.stack([F_DF, F_F, F_G], -1),  # [n_envs,3]
            "h":     torch.stack([h_DF, h_F, h_G], -1),  # [n_envs,3]
            "tau_w": tau_w.detach(),                     # [n_envs]
            "tau_g": tau_g.detach(),                     # [n_envs]
            "tau_ch": torch.stack([
                _env1(self.ch_DF.last_tau),
                _env1(self.ch_F.last_tau),
                _env1(self.ch_G.last_tau),
            ], -1),                                      # [n_envs,3]
            "taut":  torch.stack([
                (h_DF <= h0_DF).float() if self.h0_map is not None else torch.zeros_like(h_DF),
                (h_F  <= h0_F ).float() if self.h0_map is not None else torch.zeros_like(h_F),
                (h_G  <= h0_G ).float() if self.h0_map is not None else torch.zeros_like(h_G),
            ], -1),                                      # [n_envs,3]
        }

    def get_last_telemetry(self):
        return getattr(self, "_last_telemetry", None)

        