# assets.py
from __future__ import annotations
from pathlib import Path
import math
import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sim.schemas import MassPropertiesCfg

# --- 定数 ---
WRIST_J0       = math.radians(0.0)
GRIP_J0        = math.radians(-8.1)

# --- ヘルパー関数 ---
def quat_from_euler_zyx(yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0):
    # 入力は度。Z(ヨー)→Y(ピッチ)→X(ロール)の順で回転をかける想定
    z = math.radians(yaw_deg)
    y = math.radians(pitch_deg)
    x = math.radians(roll_deg)
    cz, sz = math.cos(z/2), math.sin(z/2)
    cy, sy = math.cos(y/2), math.sin(y/2)
    cx, sx = math.cos(x/2), math.sin(x/2)
    # q = qz * qy * qx（右手系）
    w =  cz*cy*cx + sz*sy*sx
    qx = cz*cy*sx - sz*sy*cx
    qy = cz*sy*cx + sz*cy*sx
    qz = sz*cy*cx - cz*sy*sx
    # 正規化（数値誤差対策）
    norm = math.sqrt(w*w + qx*qx + qy*qy + qz*qz)
    return (w/norm, qx/norm, qy/norm, qz/norm)

# --- アセットのパス設定 ---
# このスクリプトが (リポジトリルート)/source/standalone/environments/porcaro/
# に置かれることを想定しています。
try:
    ASSETS_DIR = (Path(__file__).resolve().parents[3] / "assets")
    DATA_DIR = (Path(__file__).resolve().parents[3] / "data")
except (IndexError, NameError):
    # 対話環境などで __file__ が未定義の場合のフォールバック
    print("[Warning] __file__ が未定義です。ASSETS_DIR をカレントディレクトリ基準で仮設定します。")
    ASSETS_DIR = Path.cwd().parents[2] / "assets"
    DATA_DIR = Path.cwd().parents[2] / "data"

ROBOT_USD  = str(ASSETS_DIR / "porcaro.usd")
DRUM_USD   = str(ASSETS_DIR / "sneadrum.usd")
FORCE_MAP_CSV_PATH = str(DATA_DIR / "pam_force_map.csv")
H0_MAP_CSV_PATH    = str(DATA_DIR / "pam_force_0_map.csv")

if not os.path.exists(ROBOT_USD):
    raise FileNotFoundError(f"Robot USD not found: {ROBOT_USD}")
if not os.path.exists(DRUM_USD):
    raise FileNotFoundError(f"Drum USD not found: {DRUM_USD}")

if not os.path.exists(FORCE_MAP_CSV_PATH):
    print(f"[Warning] Force map CSV not found (setting to None): {FORCE_MAP_CSV_PATH}")
    FORCE_MAP_CSV = None
else:
    FORCE_MAP_CSV = FORCE_MAP_CSV_PATH

if not os.path.exists(H0_MAP_CSV_PATH):
    print(f"[Warning] H0 map CSV not found (setting to None): {H0_MAP_CSV_PATH}")
    H0_MAP_CSV = None
else:
    H0_MAP_CSV = H0_MAP_CSV_PATH

# --- アセットCFG定義 ---

ROBOT_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot", # PrimPath は Cfg クラス内で上書き
    spawn=sim_utils.UsdFileCfg(
        usd_path=ROBOT_USD,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.11348, -0.34041, 0.0),
        rot=quat_from_euler_zyx(yaw_deg=30, pitch_deg=0, roll_deg=0),
        joint_pos={
            "Base_link_Wrist_joint": WRIST_J0,
            "Hand_link_Grip_joint":  GRIP_J0,
        },
    ),
    actuators={
        "wrist": ImplicitActuatorCfg(
            joint_names_expr=["Base_link_Wrist_joint"],
            stiffness=0,
            damping=0,
            effort_limit_sim=500.0,
        ),
        "grip": ImplicitActuatorCfg(
            joint_names_expr=["Hand_link_Grip_joint"],
            stiffness=0,
            damping=0,
            effort_limit_sim=500.0,
        ),
    },
)

DRUM_CFG = RigidObjectCfg(
    prim_path="/World/envs/env_.*/Drum", # PrimPath は Cfg クラス内で上書き
    spawn=sim_utils.UsdFileCfg(
        usd_path=DRUM_USD,
        mass_props=MassPropertiesCfg(mass=1.0e3),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=False,
            disable_gravity=False,
            ),
        activate_contact_sensors=True,
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.74573, 0.04029, 0.0),
        rot=quat_from_euler_zyx(yaw_deg=0, pitch_deg=0, roll_deg=0),
    ),
)