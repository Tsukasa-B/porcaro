# source/porcaro_rl/porcaro_rl/tasks/direct/porcaro_rlv1/actions/torque.py
from __future__ import annotations
import torch
from .base import ActionController, RobotLike
from .pam import (
    PAMChannel, contraction_ratio_from_angle, calculate_effective_contraction,
    Fpam_quasi_static, PamForceMap, H0Map,
    apply_soft_engagement,
    calculate_absolute_contraction,
    apply_model_a_force
)
from ..cfg.actuator_cfg import PamGeometricCfg

class TorqueActionController(ActionController):
    def __init__(self,
                 dt_ctrl: float,
                 control_mode: str = "pressure", 
                 r: float = 0.014, L: float = 0.150,
                 theta_t_DF_deg: float = 0.0,
                 theta_t_F_deg:  float = 70.0,
                 theta_t_G_deg:  float = 45.0,
                 Pmax: float = 0.6,
                 tau: float = 0.09, dead_time: float = 0.03,
                 N: float = 630.0,
                 pam_viscosity: float = 0.0,
                 force_map_csv: str | None = None,
                 force_scale: float = 1.0,
                 h0_map_csv: str | None = None,
                 use_pressure_dependent_tau: bool = True,
                 geometric_cfg: PamGeometricCfg | None = None,
                 pressure_shrink_gain: float = 0.02, 
                 engagement_smoothness: float = 150.0):

        self.dt_ctrl = float(dt_ctrl)
        self.control_mode = control_mode
        self.r, self.L = float(r), float(L)
        self.theta_t = {"DF": float(theta_t_DF_deg),
                        "F":  float(theta_t_F_deg),
                        "G":  float(theta_t_G_deg)}
        self.Pmax = float(Pmax)
        self.N = float(N)
        self.pam_viscosity = float(pam_viscosity)
        self.force_scale = float(force_scale)
        self.engagement_smoothness = float(engagement_smoothness)

        # Force Mapの読み込み
        print("-" * 60)
        print(f"[TorqueActionController] Initializing PAM Force Model...")
        if force_map_csv:
            try:
                self.force_map = PamForceMap.from_csv(force_map_csv)
                print(f"  >>> SUCCESS: Loaded Real Force Map")
            except Exception as e:
                print(f"  >>> ERROR: Failed to load Force Map: {e}")
                raise e
        else:
            self.force_map = None
            print(f"  >>> WARNING: No CSV provided. Using Ideal Quasi-static Model")
        print("-" * 60)
        
        self.h0_map = H0Map.from_csv(h0_map_csv) if h0_map_csv else None

        # --- Model A/B 切り替え設定 ---
        if geometric_cfg is not None:
            self.use_effective_contraction = geometric_cfg.enable_slack_compensation
            self.slack_offsets = torch.tensor(geometric_cfg.wire_slack_offsets)
            self.L0_sim = geometric_cfg.natural_length
            self.use_absolute_geometry = getattr(geometric_cfg, "use_absolute_geometry", False)
        else:
            self.use_effective_contraction = False
            self.slack_offsets = torch.zeros(3)
            self.L0_sim = self.L
            self.use_absolute_geometry = False
        
        # Model Aの場合は2Dダイナミクスを使わない、Model Bなら使う
        use_2d = not self.use_absolute_geometry
        
        tau_lut = None
        if use_pressure_dependent_tau:
            tau_P_axis = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
            tau_vals   = [0.043,0.045,0.060,0.066,0.094,0.131]
            tau_lut = (tau_P_axis, tau_vals)

        # PAMChannel 初期化
        # Model A (use_absolute=True) -> use_2d_dynamics=False -> use_table_i=True (tau固定/L可変)
        # Model B (use_absolute=False) -> use_2d_dynamics=True
        self.ch_DF = PAMChannel(dt_ctrl, tau=tau, dead_time=dead_time, Pmax=self.Pmax, tau_lut=tau_lut, use_2d_dynamics=use_2d)
        self.ch_F  = PAMChannel(dt_ctrl, tau=tau, dead_time=dead_time, Pmax=self.Pmax, tau_lut=tau_lut, use_2d_dynamics=use_2d)
        self.ch_G  = PAMChannel(dt_ctrl, tau=tau, dead_time=dead_time, Pmax=self.Pmax, tau_lut=tau_lut, use_2d_dynamics=use_2d)
        
        self._last_telemetry: dict | None = None
        self.pressure_shrink_gain = float(pressure_shrink_gain)

    def reset(self, n_envs: int, device: str | torch.device):
        self.ch_DF.reset(n_envs, device)
        self.ch_F.reset(n_envs, device)
        self.ch_G.reset(n_envs, device)
        if self.force_map is not None: self.force_map.to(device)
        if self.h0_map is not None: self.h0_map.to(device)
        self.slack_offsets = self.slack_offsets.to(device)
        self._last_telemetry = None

    def reset_idx(self, env_ids: torch.Tensor):
        self.ch_DF.reset_idx(env_ids)
        self.ch_F.reset_idx(env_ids)
        self.ch_G.reset_idx(env_ids)

    def compute_pressure(self, actions: torch.Tensor) -> torch.Tensor:
        actions = torch.nan_to_num(actions, nan=0.0)
        actions = torch.clamp(actions, min=-1.0, max=1.0)
        return self._compute_command_pressure(actions)

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
            P_cmd_stack = torch.zeros_like(actions) 
        
        STEP_SIZE = 0.05
        P_cmd_quantized = torch.round(P_cmd_stack / STEP_SIZE) * STEP_SIZE
        return torch.clamp(P_cmd_quantized, 0.0, self.Pmax)

    @torch.no_grad()
    def apply(self, *, actions: torch.Tensor, q: torch.Tensor,
              joint_ids: tuple[int, int], robot: RobotLike) -> None:
        wid, gid = joint_ids
        n_envs = int(q.shape[0])

        # 速度取得 (Sim生:下=正 -> Code:上=正 へ変換)
        dq_sim = robot.data.joint_vel  
        dq_wrist = -torch.rad2deg(dq_sim[:, wid]) 
        dq_grip  = -torch.rad2deg(dq_sim[:, gid])

        if self.slack_offsets.device != q.device:
            self.slack_offsets = self.slack_offsets.to(q.device)

        if torch.isnan(actions).any():
            actions = torch.nan_to_num(actions, nan=0.0)
        actions = torch.clamp(actions, min=-1.0, max=1.0)

        # 1) 指令値計算
        P_cmd_stack = self._compute_command_pressure(actions)

        # 2) 遅れ計算 (Dynamics) - ここが計算の本体
        P_DF = self.ch_DF.step(P_cmd_stack[:, 0])
        P_F  = self.ch_F.step(P_cmd_stack[:, 1])
        P_G  = self.ch_G.step(P_cmd_stack[:, 2])

        q_wrist = torch.rad2deg(q[:, joint_ids[0]])
        q_grip  = torch.rad2deg(q[:, joint_ids[1]])

        # 動作方向 (Code座標系: 上=正)
        SIGN_DF =  1.0      # 上がると縮む
        SIGN_F  = -1.0      # 上がると伸びる
        SIGN_G  = -1.0      # 上がると伸びる

        if self.use_absolute_geometry:
            # === [Model A] ===
            eps_DF = calculate_absolute_contraction(q_wrist, self.theta_t["DF"], self.r, self.L0_sim)
            eps_F  = calculate_absolute_contraction(q_wrist, self.theta_t["F"],  self.r, self.L0_sim)
            eps_G  = calculate_absolute_contraction(q_grip,  self.theta_t["G"],  self.r, self.L0_sim)
            
            if self.force_map is not None and self.h0_map is not None:
                F_DF = apply_model_a_force(self.force_map, self.h0_map, P_DF, eps_DF) * self.force_scale
                F_F  = apply_model_a_force(self.force_map, self.h0_map, P_F,  eps_F)  * self.force_scale
                F_G  = apply_model_a_force(self.force_map, self.h0_map, P_G,  eps_G)  * self.force_scale
            else:
                F_DF = Fpam_quasi_static(P_DF, eps_DF, N=self.N, Pmax=self.Pmax) * self.force_scale
                F_F  = Fpam_quasi_static(P_F,  eps_F,  N=self.N, Pmax=self.Pmax) * self.force_scale
                F_G  = Fpam_quasi_static(P_G,  eps_G,  N=self.N, Pmax=self.Pmax) * self.force_scale

        else:
            # === [Model B] ===
            if self.use_effective_contraction:
                h_DF = calculate_effective_contraction(
                    q_wrist, self.theta_t["DF"], self.r, self.L0_sim, self.slack_offsets[0], 
                    pressure=P_DF, shrink_gain=self.pressure_shrink_gain, clamp=False, sign=SIGN_DF) 
                h_F = calculate_effective_contraction(
                    q_wrist, self.theta_t["F"], self.r, self.L0_sim, self.slack_offsets[1], 
                    pressure=P_F, shrink_gain=self.pressure_shrink_gain, clamp=False, sign=SIGN_F)
                h_G = calculate_effective_contraction(
                    q_grip, self.theta_t["G"], self.r, self.L0_sim, self.slack_offsets[2], 
                    pressure=P_G, shrink_gain=self.pressure_shrink_gain, clamp=False, sign=SIGN_G)
                raw_eps_DF, raw_eps_F, raw_eps_G = h_DF, h_F, h_G
            else:
                h_DF = contraction_ratio_from_angle(q_wrist, self.theta_t["DF"], self.r, self.L, sign=SIGN_DF)
                h_F  = contraction_ratio_from_angle(q_wrist, self.theta_t["F"],  self.r, self.L, sign=SIGN_F)
                h_G  = contraction_ratio_from_angle(q_grip,  self.theta_t["G"],  self.r, self.L, sign=SIGN_G)
                raw_eps_DF, raw_eps_F, raw_eps_G = h_DF, h_F, h_G

            # 静的力
            if self.force_map is not None:
                F_DF_static = self.force_map(P_DF, h_DF) * self.force_scale
                F_F_static  = self.force_map(P_F,  h_F)  * self.force_scale
                F_G_static  = self.force_map(P_G,  h_G)  * self.force_scale
            else:
                F_DF_static = Fpam_quasi_static(P_DF, torch.clamp(h_DF, min=0), N=self.N, Pmax=self.Pmax) * self.force_scale
                F_F_static  = Fpam_quasi_static(P_F,  torch.clamp(h_F, min=0),  N=self.N, Pmax=self.Pmax) * self.force_scale
                F_G_static  = Fpam_quasi_static(P_G,  torch.clamp(h_G, min=0),  N=self.N, Pmax=self.Pmax) * self.force_scale

            # 粘性力
            def calculate_viscous_force(dq_deg, r, sign, viscosity_coeff):
                dq_rad = torch.deg2rad(dq_deg)
                v_muscle = -1.0 * sign * r * dq_rad
                return viscosity_coeff * v_muscle
            
            visc_DF = calculate_viscous_force(dq_wrist, self.r, SIGN_DF, self.pam_viscosity)
            visc_F  = calculate_viscous_force(dq_wrist, self.r, SIGN_F,  self.pam_viscosity)
            visc_G  = calculate_viscous_force(dq_grip,  self.r, SIGN_G,  self.pam_viscosity)

            F_DF_total_raw = F_DF_static + visc_DF
            F_F_total_raw  = F_F_static  + visc_F
            F_G_total_raw  = F_G_static  + visc_G

            # Soft Engagement
            if self.use_effective_contraction:
                F_DF = apply_soft_engagement(F_DF_total_raw, raw_eps_DF, self.engagement_smoothness)
                F_F  = apply_soft_engagement(F_F_total_raw,  raw_eps_F,  self.engagement_smoothness)
                F_G  = apply_soft_engagement(F_G_total_raw,  raw_eps_G,  self.engagement_smoothness)
            else:
                F_DF = torch.clamp(F_DF_total_raw, min=0.0)
                F_F  = torch.clamp(F_F_total_raw,  min=0.0)
                F_G  = torch.clamp(F_G_total_raw,  min=0.0)

            # H0 Cutoff
            if self.h0_map is not None:
                h0_DF, h0_F, h0_G = self.h0_map(P_DF), self.h0_map(P_F), self.h0_map(P_G)
                F_DF = torch.where(h_DF <= h0_DF, F_DF, torch.zeros_like(F_DF))
                F_F  = torch.where(h_F  <= h0_F,  F_F,  torch.zeros_like(F_F))
                F_G  = torch.where(h_G  <= h0_G,  F_G,  torch.zeros_like(F_G))

        # トルク計算 (F_DF: 上向き力, F_F: 下向き力)
        tau_w = self.r * (F_DF - F_F)
        tau_g = self.r * (- F_G)

        tau_full = torch.zeros(n_envs, robot.num_joints, device=q.device, dtype=q.dtype)
        # Code(上=正) -> Sim(下=正) 変換のため符号反転
        tau_full[:, wid] = -tau_w
        tau_full[:, gid] = -tau_g
        robot.set_joint_effort_target(tau_full)

        P_act_stack = torch.stack([P_DF, P_F, P_G], dim=1)
        self._last_telemetry = {
            "P_cmd": P_cmd_stack.clone(),
            "P_out": P_act_stack.clone(),
            "tau_w": tau_w.clone(),
            "tau_g": tau_g.clone()
        }
    
    def get_last_telemetry(self):
        return self._last_telemetry