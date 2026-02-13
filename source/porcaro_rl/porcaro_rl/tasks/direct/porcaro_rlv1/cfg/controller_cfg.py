# controller_cfg.py
from __future__ import annotations
from isaaclab.utils import configclass

from .assets import FORCE_MAP_CSV, H0_MAP_CSV

@configclass
class TorqueControllerCfg:
    """TorqueActionController の設定"""
    # 制御モード ("ep" or "pressure")
    control_mode: str = "pressure"
    
    r: float = 0.014
    L: float = 0.150
    theta_t_DF_deg: float = 7.0
    theta_t_F_deg: float = 70.0
    theta_t_G_deg: float = 45.0
    Pmax: float = 0.6 # pneumatic.py (Table I) の最大に合わせる
    tau: float = 0.09
    dead_time: float = 0.03
    N: float = 630.0 # 簡易式 Fpam_quasi_static 用 (CSVがあれば不要)
    pam_viscosity: float = 0.0
    force_map_csv: str = FORCE_MAP_CSV
    force_scale: float = 0.2
    h0_map_csv: str = H0_MAP_CSV
    use_pressure_dependent_tau: bool = True
