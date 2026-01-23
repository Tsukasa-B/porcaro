# source/porcaro_rl/porcaro_rl/tasks/direct/porcaro_rlv1/actions/torque.py
from __future__ import annotations
import torch
from .base import ActionController, RobotLike
from .pam import (
    PAMChannel, contraction_ratio_from_angle, calculate_effective_contraction,
    Fpam_quasi_static, PamForceMap, H0Map,
    apply_soft_engagement  # <--- [追加]
)
# 型ヒント用に Config を import
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
                 # ▼▼▼ [追加] 粘性係数 (N / (m/s)) ▼▼▼
                 # 目安: 10.0 ~ 50.0 くらいから調整
                 pam_viscosity: float = 500.0,
                 force_map_csv: str | None = None,
                 force_scale: float = 1.0,
                 h0_map_csv: str | None = None,
                 use_pressure_dependent_tau: bool = True,
                 geometric_cfg: PamGeometricCfg | None = None,
                 pressure_shrink_gain: float = 0.02, 
                 engagement_smoothness: float = 20.0):

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
        self.engagement_smoothness = float(engagement_smoothness) # <--- [追加] 保存

        # === ForceMap読み込み ===
        print("-" * 60)
        print(f"[TorqueActionController] Initializing PAM Force Model...")
        
        if force_map_csv:
            try:
                self.force_map = PamForceMap.from_csv(force_map_csv)
                print(f"  >>> SUCCESS: Loaded Real Force Map from: {force_map_csv}")
                print(f"      Range: P=[{self.force_map.P[0]:.2f}, {self.force_map.P[-1]:.2f}] MPa, "
                      f"h=[{self.force_map.h[0]:.2f}, {self.force_map.h[-1]:.2f}]")
                print(f"  >>> INFO: Applying Force Scale: {self.force_scale:.2f} (Real/Catalog Ratio)")
            except Exception as e:
                print(f"  >>> ERROR: Failed to load Force Map: {e}")
                raise e # 失敗時は止める
        else:
            self.force_map = None
            print(f"  >>> WARNING: No CSV provided. Using Ideal Quasi-static Model (N={self.N})")
            print("      This may cause Sim-to-Real mismatch!")
        
        print("-" * 60)
        # ========================

        # H0Map (弛み判定用) の読み込み
        self.h0_map = H0Map.from_csv(h0_map_csv) if h0_map_csv else None

        # --- 幾何学補正設定 ---
        if geometric_cfg is not None:
            self.use_effective_contraction = geometric_cfg.enable_slack_compensation
            self.slack_offsets = torch.tensor(geometric_cfg.wire_slack_offsets)
            self.L0_sim = geometric_cfg.natural_length 
        else:
            self.use_effective_contraction = False
            self.slack_offsets = torch.zeros(3)
            self.L0_sim = self.L
        # ---------------------

        tau_lut = None
        if use_pressure_dependent_tau:
            tau_P_axis = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
            tau_vals   = [0.043,0.045,0.060,0.066,0.094,0.131]
            tau_lut = (tau_P_axis, tau_vals)

        self.ch_DF = PAMChannel(dt_ctrl, tau=tau, dead_time=dead_time, Pmax=self.Pmax, tau_lut=tau_lut)
        self.ch_F  = PAMChannel(dt_ctrl, tau=tau, dead_time=dead_time, Pmax=self.Pmax, tau_lut=tau_lut)
        self.ch_G  = PAMChannel(dt_ctrl, tau=tau, dead_time=dead_time, Pmax=self.Pmax, tau_lut=tau_lut)
        
        self._last_telemetry: dict | None = None
        self.pressure_shrink_gain = float(pressure_shrink_gain)

    def reset(self, n_envs: int, device: str | torch.device):
        self.ch_DF.reset(n_envs, device)
        self.ch_F.reset(n_envs, device)
        self.ch_G.reset(n_envs, device)
        
        # --- 修正箇所: nn.Module化したマップをデバイスへ転送 ---
        if self.force_map is not None:
            self.force_map.to(device)
        if self.h0_map is not None:
            self.h0_map.to(device)
        # ----------------------------------------------------

        self._last_telemetry = None

    def reset_idx(self, env_ids: torch.Tensor):
        self.ch_DF.reset_idx(env_ids)
        self.ch_F.reset_idx(env_ids)
        self.ch_G.reset_idx(env_ids)

    def compute_pressure(self, actions: torch.Tensor) -> torch.Tensor:
        # ここでも念のためサニタイズしておく
        actions = torch.nan_to_num(actions, nan=0.0)
        actions = torch.clamp(actions, min=-1.0, max=1.0)
        return self._compute_command_pressure(actions)

    def _compute_command_pressure(self, actions: torch.Tensor) -> torch.Tensor:
        # --- 1. 各モードごとの理想的な圧力計算 (ここは変更なし) ---
        if self.control_mode == "pressure":
            P_cmd_unscaled = (actions + 1.0) * 0.5
            P_cmd_stack = P_cmd_unscaled * self.Pmax
            
        elif self.control_mode == "ep":
            MAX_P_BASE = self.Pmax * 0.5
            MAX_P_DIFF = self.Pmax * 0.5
            a = actions
            
            # P_base (剛性): 0 ~ 0.3 MPa
            P_base = MAX_P_BASE * (a[:, 1] + 1.0) * 0.5
            # P_diff (位置): -0.3 ~ +0.3 MPa
            P_diff = MAX_P_DIFF * a[:, 0] 
            
            P_cmd_DF = torch.clamp(P_base + P_diff, 0.0, self.Pmax)
            P_cmd_F  = torch.clamp(P_base - P_diff, 0.0, self.Pmax)
            # Gripは独立制御
            P_cmd_G = self.Pmax * (a[:, 2] + 1.0) * 0.5
            
            P_cmd_stack = torch.stack([P_cmd_DF, P_cmd_F, P_cmd_G], dim=-1)
            
        else:
            P_cmd_stack = torch.zeros_like(actions) 

        # --- 2. ★追加: 圧力の量子化 (Quantization) ---
        # 0.05 MPa 刻みに丸め込む処理
        STEP_SIZE = 0.05
        P_cmd_quantized = torch.round(P_cmd_stack / STEP_SIZE) * STEP_SIZE

        # --- 3. 範囲制限 (念のため再度クランプ) ---
        return torch.clamp(P_cmd_quantized, 0.0, self.Pmax)

    @torch.no_grad()
    def apply(self, *, actions: torch.Tensor, q: torch.Tensor,
              joint_ids: tuple[int, int], robot: RobotLike) -> None:
        wid, gid = joint_ids
        n_envs = int(q.shape[0])

        # ▼▼▼ [追加] 角速度 (dq) の取得 ▼▼▼
        # robot.data.joint_vel は全関節の速度を含んでいるので、wid, gidで抽出
        # 注意: IsaacLabのバージョンによっては robot.data.joint_vel へのアクセス方法を確認
        dq = robot.data.joint_vel  
        dq_wrist = torch.rad2deg(dq[:, wid]) # [deg/s]
        dq_grip  = torch.rad2deg(dq[:, gid]) # [deg/s]

        if self.slack_offsets.device != q.device:
            self.slack_offsets = self.slack_offsets.to(q.device)

        if torch.isnan(actions).any():
            actions = torch.nan_to_num(actions, nan=0.0)
        actions = torch.clamp(actions, min=-1.0, max=1.0)

        # 1) 指令値計算
        P_cmd_stack = self._compute_command_pressure(actions)

        # 2) 遅れ計算
        P_DF = self.ch_DF.step(P_cmd_stack[:, 0])
        P_F  = self.ch_F.step(P_cmd_stack[:, 1])
        P_G  = self.ch_G.step(P_cmd_stack[:, 2])

        # 3) 収縮率 h (epsilon) の計算
        q_wrist = torch.rad2deg(q[:, joint_ids[0]])
        q_grip  = torch.rad2deg(q[:, joint_ids[1]])

        if self.use_effective_contraction:
            # 変更: signの適用
            # DF (Negative Torque): theta減で縮む -> sign = -1
            eps_DF = calculate_effective_contraction(
                q_wrist, self.theta_t["DF"], self.r, self.L0_sim, self.slack_offsets[0], 
                pressure=P_DF, shrink_gain=self.pressure_shrink_gain, clamp=False,
                sign=-1.0) # <---
            
            # F (Positive Torque): theta増で縮む -> sign = +1
            eps_F = calculate_effective_contraction(
                q_wrist, self.theta_t["F"], self.r, self.L0_sim, self.slack_offsets[1], 
                pressure=P_F, shrink_gain=self.pressure_shrink_gain, clamp=False,
                sign=1.0) # <---
            
            # Grip (Positive Torque): theta増で縮む(閉じる) -> sign = +1
            eps_G = calculate_effective_contraction(
                q_grip, self.theta_t["G"], self.r, self.L0_sim, self.slack_offsets[2], 
                pressure=P_G, shrink_gain=self.pressure_shrink_gain, clamp=False,
                sign=1.0) # <---
            
            raw_eps_DF, raw_eps_F, raw_eps_G = eps_DF, eps_F, eps_G

            # デバッグ: 負の値（伸び）が正しく出ているか確認
            # if n_envs > 0:
            #     print(f"DEBUG Eps: DF={eps_DF[0].item():.3f}, F={eps_F[0].item():.3f}")

            # 変更: 0クランプを削除 (負の値を許容)
            h_DF = eps_DF
            h_F  = eps_F
            h_G  = eps_G
        else:
            # 従来モードも sign 対応
            h_DF = contraction_ratio_from_angle(q_wrist, self.theta_t["DF"], self.r, self.L, sign=-1.0)
            h_F  = contraction_ratio_from_angle(q_wrist, self.theta_t["F"], self.r, self.L, sign=1.0)
            h_G  = contraction_ratio_from_angle(q_grip, self.theta_t["G"], self.r, self.L, sign=1.0)
            raw_eps_DF, raw_eps_F, raw_eps_G = h_DF, h_F, h_G

            # 4) 静的力 F (Static Force)
        if self.force_map is not None:
            F_DF_static = self.force_map(P_DF, h_DF) * self.force_scale
            F_F_static  = self.force_map(P_F,  h_F)  * self.force_scale
            F_G_static  = self.force_map(P_G,  h_G)  * self.force_scale
        else:
            # 理想モデル
            F_DF_static = Fpam_quasi_static(P_DF, torch.clamp(h_DF, min=0), N=self.N, Pmax=self.Pmax) * self.force_scale
            F_F_static  = Fpam_quasi_static(P_F,  torch.clamp(h_F, min=0),  N=self.N, Pmax=self.Pmax) * self.force_scale
            F_G_static  = Fpam_quasi_static(P_G,  torch.clamp(h_G, min=0),  N=self.N, Pmax=self.Pmax) * self.force_scale

        # 5) 粘性力 F_visc (Viscous Force)
        # 係数を大きくする必要があります (例: 1000.0 ~ 5000.0)
        def calculate_viscous_force(dq_deg, r, sign, viscosity_coeff):
            dq_rad = torch.deg2rad(dq_deg)
            # 筋肉の収縮速度 v (縮む方向が負、伸びる方向が正)
            v_muscle = -1.0 * sign * r * dq_rad
            # 粘性抵抗: 速度に対抗する力
            return viscosity_coeff * v_muscle

        visc_DF = calculate_viscous_force(dq_wrist, self.r, -1.0, self.pam_viscosity)
        visc_F  = calculate_viscous_force(dq_wrist, self.r,  1.0, self.pam_viscosity)
        visc_G  = calculate_viscous_force(dq_grip,  self.r,  1.0, self.pam_viscosity)

        # 6) 【重要修正】合算してから Engagement を適用
        # ワイヤーがたるんでいる時(Mask=0)は、静的力も粘性力も伝わらないのが物理的に正しい
        F_DF_total_raw = F_DF_static + visc_DF
        F_F_total_raw  = F_F_static  + visc_F
        F_G_total_raw  = F_G_static  + visc_G

        if self.use_effective_contraction:
            # Soft Engagement (Mask) を「合力」にかける
            F_DF = apply_soft_engagement(F_DF_total_raw, raw_eps_DF, self.engagement_smoothness)
            F_F  = apply_soft_engagement(F_F_total_raw,  raw_eps_F,  self.engagement_smoothness)
            F_G  = apply_soft_engagement(F_G_total_raw,  raw_eps_G,  self.engagement_smoothness)
        else:
            # Engagementを使わない場合はそのまま (負の値は0クリップ推奨)
            F_DF = torch.clamp(F_DF_total_raw, min=0.0)
            F_F  = torch.clamp(F_F_total_raw,  min=0.0)
            F_G  = torch.clamp(F_G_total_raw,  min=0.0)

        # 7) h0マップによる不感帯処理 (これも合力にかける)
        if self.h0_map is not None:
            h0_DF = self.h0_map(P_DF)
            h0_F  = self.h0_map(P_F)
            h0_G  = self.h0_map(P_G)
            
            # h <= h0 (張っている) なら力を通す。そうでなければ0
            F_DF = torch.where(h_DF <= h0_DF, F_DF, torch.zeros_like(F_DF))
            F_F  = torch.where(h_F  <= h0_F,  F_F,  torch.zeros_like(F_F))
            F_G  = torch.where(h_G  <= h0_G,  F_G,  torch.zeros_like(F_G))

        # 8) トルク計算
        tau_w = self.r * (F_F - F_DF)
        tau_g = self.r * F_G

        tau_full = torch.zeros(n_envs, robot.num_joints, device=q.device, dtype=q.dtype)
        tau_full[:, wid] = tau_w
        tau_full[:, gid] = tau_g
        robot.set_joint_effort_target(tau_full)

        self._last_telemetry = {
            "P_out": P_cmd_stack.clone(),
            "tau_w": tau_w.clone(),
            "tau_g": tau_g.clone()
        }
    
    def get_last_telemetry(self):
        return self._last_telemetry