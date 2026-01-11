# porcaro_rl/actions/torque.py
from __future__ import annotations
import torch
from .base import ActionController, RobotLike
from .pam import (
    PAMChannel, contraction_ratio_from_angle, Fpam_quasi_static, PamForceMap, H0Map
)

class TorqueActionController(ActionController):
    def __init__(self,
                 dt_ctrl: float,
                 control_mode: str = "ep", 
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

        self.dt_ctrl = float(dt_ctrl)
        self.control_mode = control_mode
        self.r, self.L = float(r), float(L)
        self.theta_t = {"DF": float(theta_t_DF_deg),
                        "F":  float(theta_t_F_deg),
                        "G":  float(theta_t_G_deg)}
        self.Pmax = float(Pmax)
        self.N = float(N)

        self.force_map = PamForceMap.from_csv(force_map_csv) if force_map_csv else None
        self.h0_map    = H0Map.from_csv(h0_map_csv) if h0_map_csv else None

        tau_lut = None
        if use_pressure_dependent_tau:
            tau_P_axis = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
            tau_vals   = [0.043,0.045,0.060,0.066,0.094,0.131]
            tau_lut = (tau_P_axis, tau_vals)

        self.ch_DF = PAMChannel(dt_ctrl, tau=tau, dead_time=dead_time, Pmax=self.Pmax, tau_lut=tau_lut)
        self.ch_F  = PAMChannel(dt_ctrl, tau=tau, dead_time=dead_time, Pmax=self.Pmax, tau_lut=tau_lut)
        self.ch_G  = PAMChannel(dt_ctrl, tau=tau, dead_time=dead_time, Pmax=self.Pmax, tau_lut=tau_lut)

        # printなどのI/O処理も極力排除
        # print(f"[TorqueActionController] Initialized with Mode: {self.control_mode}")

    def reset(self, n_envs: int, device: str | torch.device):
        self.ch_DF.reset(n_envs, device)
        self.ch_F.reset(n_envs, device)
        self.ch_G.reset(n_envs, device)

    def reset_idx(self, env_ids: torch.Tensor):
        self.ch_DF.reset_idx(env_ids)
        self.ch_F.reset_idx(env_ids)
        self.ch_G.reset_idx(env_ids)

    def _compute_command_pressure(self, actions: torch.Tensor) -> torch.Tensor:
        if self.control_mode == "pressure":
            P_cmd_unscaled = (actions + 1.0) * 0.5
            P_cmd_stack = P_cmd_unscaled * self.Pmax
        elif self.control_mode == "ep":
            MAX_P_BASE = self.Pmax * 0.5
            MAX_P_DIFF = self.Pmax * 0.5
            a = actions
            P_base = MAX_P_BASE * (a[:, 1] + 1.0) * 0.5
            P_diff = MAX_P_DIFF * a[:, 0] 
            P_cmd_DF = torch.clamp(P_base + P_diff, 0.0, self.Pmax)
            P_cmd_F  = torch.clamp(P_base - P_diff, 0.0, self.Pmax)
            P_cmd_G = self.Pmax * (a[:, 2] + 1.0) * 0.5
            P_cmd_stack = torch.stack([P_cmd_DF, P_cmd_F, P_cmd_G], dim=-1)
        else:
            # エラーチェックすらコストになるので、開発が終わっていればassertのみにするか削除
            P_cmd_stack = torch.zeros_like(actions) 
        
        return torch.clamp(P_cmd_stack, 0.0, self.Pmax)

    @torch.no_grad()
    def apply(self, *, actions: torch.Tensor, q: torch.Tensor,
              joint_ids: tuple[int, int], robot: RobotLike) -> None:
        wid, gid = joint_ids
        n_envs = int(q.shape[0])

        # 1) 指令値計算
        P_cmd_stack = self._compute_command_pressure(actions)

        # 2) 遅れ計算
        P_DF = self.ch_DF.step(P_cmd_stack[:, 0])
        P_F  = self.ch_F.step(P_cmd_stack[:, 1])
        P_G  = self.ch_G.step(P_cmd_stack[:, 2])

        # 3) 収縮率 h
        theta_w = q[:, wid]
        theta_g = q[:, gid]
        h_DF = contraction_ratio_from_angle(theta_w, self.theta_t["DF"], self.r, self.L)
        h_F  = contraction_ratio_from_angle(theta_w, self.theta_t["F"],  self.r, self.L)
        h_G  = contraction_ratio_from_angle(theta_g, self.theta_t["G"],  self.r, self.L)

        # 4) 力 F
        if self.force_map is not None:
            F_DF = self.force_map(P_DF, h_DF)
            F_F  = self.force_map(P_F,  h_F)
            F_G  = self.force_map(P_G,  h_G)
        else:
            F_DF = Fpam_quasi_static(P_DF, h_DF, N=self.N, Pmax=self.Pmax)
            F_F  = Fpam_quasi_static(P_F,  h_F,  N=self.N, Pmax=self.Pmax)
            F_G  = Fpam_quasi_static(P_G,  h_G,  N=self.N, Pmax=self.Pmax)

        # 4.5) 弛み判定 (h0_mapがある場合のみ計算)
        if self.h0_map is not None:
            h0_DF = self.h0_map(P_DF)
            h0_F  = self.h0_map(P_F)
            h0_G  = self.h0_map(P_G)
            
            F_DF = torch.where(h_DF <= h0_DF, F_DF, torch.zeros_like(F_DF))
            F_F  = torch.where(h_F  <= h0_F,  F_F,  torch.zeros_like(F_F))
            F_G  = torch.where(h_G  <= h0_G,  F_G,  torch.zeros_like(F_G))

        # 5) トルク計算
        tau_w = self.r * (F_F - F_DF)
        tau_g = self.r * F_G

        tau_full = torch.zeros(n_envs, robot.num_joints, device=q.device, dtype=q.dtype)
        tau_full[:, wid] = tau_w
        tau_full[:, gid] = tau_g
        robot.set_joint_effort_target(tau_full)

        # テレメトリ保存 (必要な場合のみ)
        # self._last_telemetry = ... (学習速度優先ならコメントアウト推奨)
    
    # ログ用メソッドが必要なら残すが、学習ループで呼ばないこと
    def get_last_telemetry(self):
        return None