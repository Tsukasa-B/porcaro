# source/porcaro_rl/porcaro_rl/tasks/direct/porcaro_rlv1/actions/torque.py
from __future__ import annotations
import torch
from .base import ActionController, RobotLike
from .pam import (
    PAMChannel, 
    calculate_geometric_contraction, 
    calculate_effective_contraction,
    apply_soft_engagement,
    Fpam_quasi_static, PamForceMap, H0Map
)
from ..cfg.actuator_cfg import PamGeometricCfg

class TorqueActionController(ActionController):
    def __init__(self,
                 dt_ctrl: float,
                 control_mode: str = "ep", 
                 r: float = 0.014, L: float = 0.150,
                 theta_t_DF_deg: float = 0.0,
                 theta_t_F_deg:  float = -60.0,
                 theta_t_G_deg:  float = -45.0,
                 Pmax: float = 0.6,
                 tau: float = 0.09, dead_time: float = 0.03,
                 N: float = 630.0,
                 pam_viscosity: float = 500.0,
                 force_map_csv: str | None = None,
                 force_scale: float = 1.0,
                 h0_map_csv: str | None = None,
                 use_pressure_dependent_tau: bool = True,
                 geometric_cfg: PamGeometricCfg | None = None,
                 pressure_shrink_gain: float = 0.02, 
                 engagement_smoothness: float = 100.0):

        self.dt_ctrl = float(dt_ctrl)
        self.control_mode = control_mode
        self.r, self.L = float(r), float(L)
        self.theta_t = {"DF": float(theta_t_DF_deg), "F": float(theta_t_F_deg), "G": float(theta_t_G_deg)}
        self.Pmax = float(Pmax)
        self.N = float(N)
        
        # --- Config Check ---
        if geometric_cfg is not None:
            self.model_type = geometric_cfg.model_type
            self.use_effective_contraction = geometric_cfg.enable_slack_compensation
            self.slack_offsets = torch.tensor(geometric_cfg.wire_slack_offsets)
            self.L0_sim = geometric_cfg.natural_length
            
            # Model Specifics
            self.use_abs_epsilon = geometric_cfg.use_abs_epsilon
            self.pam_viscosity = geometric_cfg.viscosity
            self.enable_soft_engagement = geometric_cfg.enable_soft_engagement
            self.engagement_smoothness = geometric_cfg.engagement_smoothness
            self.force_scale = geometric_cfg.force_scale
        else:
            self.model_type = "B"
            self.use_effective_contraction = False
            self.slack_offsets = torch.zeros(3)
            self.L0_sim = self.L
            self.use_abs_epsilon = False
            self.pam_viscosity = float(pam_viscosity)
            self.enable_soft_engagement = True
            self.engagement_smoothness = float(engagement_smoothness)
            self.force_scale = float(force_scale)

        self.pressure_shrink_gain = float(pressure_shrink_gain)

        if force_map_csv:
            self.force_map = PamForceMap.from_csv(force_map_csv)
        else:
            self.force_map = None

        self.h0_map = H0Map.from_csv(h0_map_csv) if h0_map_csv else None

        # Note: Dynamics are mainly handled in env via PamDelayModel.
        # This channel is kept for structure or legacy compatibility if needed.
        self.ch_DF = PAMChannel(dt_ctrl, tau=tau, dead_time=dead_time, Pmax=self.Pmax)
        self.ch_F  = PAMChannel(dt_ctrl, tau=tau, dead_time=dead_time, Pmax=self.Pmax)
        self.ch_G  = PAMChannel(dt_ctrl, tau=tau, dead_time=dead_time, Pmax=self.Pmax)
        
        self._last_telemetry: dict | None = None

    def reset(self, n_envs: int, device: str | torch.device):
        self.ch_DF.reset(n_envs, device)
        self.ch_F.reset(n_envs, device)
        self.ch_G.reset(n_envs, device)
        if self.force_map: self.force_map.to(device)
        if self.h0_map: self.h0_map.to(device)
        self._last_telemetry = None

    def reset_idx(self, env_ids: torch.Tensor):
        self.ch_DF.reset_idx(env_ids)
        self.ch_F.reset_idx(env_ids)
        self.ch_G.reset_idx(env_ids)

    def compute_pressure(self, actions: torch.Tensor) -> torch.Tensor:
        actions = torch.clamp(torch.nan_to_num(actions, 0.0), -1.0, 1.0)
        
        if self.control_mode == "pressure":
            P_stack = (actions + 1.0) * 0.5 * self.Pmax
        elif self.control_mode == "ep":
            MAX_P = self.Pmax * 0.5
            a = actions
            P_base = MAX_P * (a[:, 1] + 1.0) * 0.5
            P_diff = MAX_P * a[:, 0]
            P_DF = torch.clamp(P_base + P_diff, 0.0, self.Pmax)
            P_F  = torch.clamp(P_base - P_diff, 0.0, self.Pmax)
            P_G  = self.Pmax * (a[:, 2] + 1.0) * 0.5
            P_stack = torch.stack([P_DF, P_F, P_G], dim=-1)
        else:
            P_stack = torch.zeros_like(actions)
        
        # Quantization
        return torch.clamp(torch.round(P_stack / 0.05) * 0.05, 0.0, self.Pmax)

    @torch.no_grad()
    def apply(self, *, actions: torch.Tensor, q: torch.Tensor,
              joint_ids: tuple[int, int], robot: RobotLike) -> None:
        wid, gid = joint_ids
        n_envs = int(q.shape[0])
        if self.slack_offsets.device != q.device: self.slack_offsets = self.slack_offsets.to(q.device)

        # 1. State
        dq = robot.data.joint_vel
        dq_wrist = torch.rad2deg(dq[:, wid])
        dq_grip  = torch.rad2deg(dq[:, gid])
        q_wrist = torch.rad2deg(q[:, wid])
        q_grip  = torch.rad2deg(q[:, gid])

        # 2. Pressure & Dynamics
        # P_cmd_stack には env側ですでにDynamicsがかかっている値が入っている想定
        P_cmd_stack = self.compute_pressure(actions)
        
        # コントローラ内部のチャネルも一応更新しておく (描画用など)
        # ※ 実質的な遅れ処理は env.pam_delay で行われているので、ここはパススルー設定(tau=0)になっているはず
        P_DF = self.ch_DF.step(P_cmd_stack[:, 0])
        P_F  = self.ch_F.step(P_cmd_stack[:, 1])
        P_G  = self.ch_G.step(P_cmd_stack[:, 2])

        # 3. 収縮率計算 (epsilon)
        if self.use_effective_contraction:
            # Model B: 有効収縮率 (Slack考慮)
            eps_DF = calculate_effective_contraction(q_wrist, self.theta_t["DF"], self.r, self.L0_sim, self.slack_offsets[0], P_DF, self.pressure_shrink_gain, sign=-1.0)
            eps_F  = calculate_effective_contraction(q_wrist, self.theta_t["F"],  self.r, self.L0_sim, self.slack_offsets[1], P_F,  self.pressure_shrink_gain, sign=1.0)
            eps_G  = calculate_effective_contraction(q_grip,  self.theta_t["G"],  self.r, self.L0_sim, self.slack_offsets[2], P_G,  self.pressure_shrink_gain, sign=1.0)
        else:
            # Model A: 幾何学収縮率
            eps_DF = calculate_geometric_contraction(q_wrist, self.theta_t["DF"], self.r, self.L, sign=-1.0)
            eps_F  = calculate_geometric_contraction(q_wrist, self.theta_t["F"],  self.r, self.L, sign=1.0)
            eps_G  = calculate_geometric_contraction(q_grip,  self.theta_t["G"],  self.r, self.L, sign=1.0)

        # [Model A] 絶対値化 (たるみでも力発生)
        if self.use_abs_epsilon:
            eps_DF = torch.abs(eps_DF)
            eps_F  = torch.abs(eps_F)
            eps_G  = torch.abs(eps_G)

        # 4. 静的力計算 (Force Map)
        if self.force_map is not None:
            # マップ内で epsilon < 0 (伸展) は epsilon=0 (最大力) として扱われる
            F_DF_st = self.force_map(P_DF, eps_DF) * self.force_scale
            F_F_st  = self.force_map(P_F,  eps_F)  * self.force_scale
            F_G_st  = self.force_map(P_G,  eps_G)  * self.force_scale
        else:
            # 簡易モデル
            F_DF_st = Fpam_quasi_static(P_DF, eps_DF.clamp(min=0), N=self.N, Pmax=self.Pmax) * self.force_scale
            F_F_st  = Fpam_quasi_static(P_F,  eps_F.clamp(min=0),  N=self.N, Pmax=self.Pmax) * self.force_scale
            F_G_st  = Fpam_quasi_static(P_G,  eps_G.clamp(min=0),  N=self.N, Pmax=self.Pmax) * self.force_scale

        # 5. 粘性力
        def calc_visc(dq_deg, sign, coeff):
            if coeff <= 1e-6: return 0.0
            # v_muscle > 0 : 収縮方向速度
            v_muscle = sign * self.r * torch.deg2rad(dq_deg)
            # 抵抗力 = -c * v
            return -coeff * v_muscle

        # Model Aでは visc=0 なので効かない
        visc_DF = calc_visc(dq_wrist, -1.0, self.pam_viscosity)
        visc_F  = calc_visc(dq_wrist,  1.0, self.pam_viscosity)
        visc_G  = calc_visc(dq_grip,   1.0, self.pam_viscosity)

        # 6. 合力 & Soft Engagement
        F_DF_raw = F_DF_st + visc_DF
        F_F_raw  = F_F_st  + visc_F
        F_G_raw  = F_G_st  + visc_G

        if self.enable_soft_engagement:
            # [Model B] Soft Engagement (負論理: epsilon増でマスク0)
            F_DF = apply_soft_engagement(F_DF_raw, eps_DF, self.engagement_smoothness)
            F_F  = apply_soft_engagement(F_F_raw,  eps_F,  self.engagement_smoothness)
            F_G  = apply_soft_engagement(F_G_raw,  eps_G,  self.engagement_smoothness)
        else:
            # [Model A] 単純クランプ
            F_DF = torch.clamp(F_DF_raw, min=0.0)
            F_F  = torch.clamp(F_F_raw,  min=0.0)
            F_G  = torch.clamp(F_G_raw,  min=0.0)

        # 7. トルク適用
        tau_w = self.r * (F_F - F_DF)
        tau_g = self.r * F_G

        torques = torch.zeros_like(q)
        torques[:, wid] = tau_w
        torques[:, gid] = tau_g
        robot.set_joint_effort_target(torques)

        self._last_telemetry = {
            "P_out": P_cmd_stack, "tau_w": tau_w, "tau_g": tau_g
        }
    
    def get_last_telemetry(self):
        return self._last_telemetry