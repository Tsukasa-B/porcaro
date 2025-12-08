# controller_cfg.py
from __future__ import annotations
from isaaclab.utils import configclass

from .assets import FORCE_MAP_CSV, H0_MAP_CSV

@configclass
class TorqueControllerCfg:
    """TorqueActionController の設定"""
    r: float = 0.014
    L: float = 0.150
    theta_t_DF_deg: float = 0.0
    theta_t_F_deg: float = -90.0
    theta_t_G_deg: float = -45.0
    Pmax: float = 0.6 # pneumatic.py (Table I) の最大に合わせる
    tau: float = 0.09
    dead_time: float = 0.03
    N: float = 630.0 # 簡易式 Fpam_quasi_static 用 (CSVがあれば不要)
    force_map_csv: str | None = FORCE_MAP_CSV
    h0_map_csv: str | None = H0_MAP_CSV
    use_pressure_dependent_tau: bool = True