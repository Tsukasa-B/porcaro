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

# =========================================================================
# ★ 変更箇所: データパス探索ロジックの強化
# =========================================================================
def find_project_root_and_data(start_path: Path, max_depth: int = 6) -> tuple[Path, Path]:
    """
    カレントまたはファイルの場所から親ディレクトリを遡り、'data' フォルダを探す。
    見つからない場合はエラーとする。
    """
    current = start_path.resolve()
    for _ in range(max_depth):
        # 候補: dataフォルダがあるかチェック
        candidate_data = current / "data"
        if candidate_data.exists() and candidate_data.is_dir():
             # さらにその中にpam_force_map.csvがあるか確認（確実性のため）
             if (candidate_data / "pam_force_map.csv").exists():
                 return current, candidate_data
        
        # 親へ
        if current.parent == current: # ルート到達
            break
        current = current.parent
    
    raise FileNotFoundError(
        f"Could not find 'data' directory containing 'pam_force_map.csv' by traversing up from {start_path}. "
        "Please ensure the 'data' folder exists in the project root."
    )

# パス解決の実行
try:
    # このファイルの場所を基準に探索
    BASE_DIR = Path(__file__).parent
    PROJECT_ROOT, DATA_DIR = find_project_root_and_data(BASE_DIR)
    
    # Assetsは data と同階層にあると仮定、なければ探索
    ASSETS_DIR = PROJECT_ROOT / "assets"
    if not ASSETS_DIR.exists():
        # assetsも探す場合の簡易ロジック（必要ならdata同様にする）
        print(f"[Warning] assets dir not found at {ASSETS_DIR}, trying to rely on defaults or manual set.")

except NameError:
    # __file__ がない対話実行時などはカレントディレクトリ基準
    PROJECT_ROOT, DATA_DIR = find_project_root_and_data(Path.cwd())
    ASSETS_DIR = PROJECT_ROOT / "assets"

print(f"[INFO] Asset Configuration:")
print(f"  - Project Root: {PROJECT_ROOT}")
print(f"  - Data Dir    : {DATA_DIR}")
print(f"  - Assets Dir  : {ASSETS_DIR}")

ROBOT_USD  = str(ASSETS_DIR / "porcaro.usd")
DRUM_USD   = str(ASSETS_DIR / "sneadrum.usd")

# ★ 変更箇所: ファイルが見つからない場合は例外を発生させ、サイレントなフォールバックを阻止する
FORCE_MAP_CSV_PATH = str(DATA_DIR / "pam_force_map.csv")
if not os.path.exists(FORCE_MAP_CSV_PATH):
    raise FileNotFoundError(f"[CRITICAL] Force Map CSV NOT found at: {FORCE_MAP_CSV_PATH}")
FORCE_MAP_CSV = FORCE_MAP_CSV_PATH

H0_MAP_CSV_PATH    = str(DATA_DIR / "pam_force_0_map.csv")
if not os.path.exists(H0_MAP_CSV_PATH):
    print(f"[Warning] H0 Map CSV NOT found at: {H0_MAP_CSV_PATH}. Using None.")
    H0_MAP_CSV = None
else:
    H0_MAP_CSV = H0_MAP_CSV_PATH
# =========================================================================

# --- アセットCFG定義 ---

ROBOT_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot", # PrimPath は Cfg クラス内で上書き
    spawn=sim_utils.UsdFileCfg(
        usd_path=ROBOT_USD,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
            contact_offset=0.02,  # 20mm: これを小さくしないと打撃感がフワフワします
            rest_offset=0.0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.11348, -0.34041, 0.0),
        rot=quat_from_euler_zyx(yaw_deg=30, pitch_deg=0, roll_deg=0),
        joint_pos={
            ".*Base_link_Wrist_joint": WRIST_J0,
            ".*Hand_link_Grip_joint":  GRIP_J0,
        },
    ),
    actuators={
        "wrist": ImplicitActuatorCfg(
            joint_names_expr=[".*Base_link_Wrist_joint"],
            stiffness=0.0,
            damping=0.2,
            effort_limit_sim=500.0,
            friction=0.05,
        ),
        "grip": ImplicitActuatorCfg(
            joint_names_expr=[".*Hand_link_Grip_joint"],
            stiffness=0.0,
            damping=0.01,
            effort_limit_sim=500.0,
            friction=0.05,
        ),
    },
)

DRUM_CFG = RigidObjectCfg(
    prim_path="/World/envs/env_.*/Drum", # PrimPath は Cfg クラス内で上書き
    spawn=sim_utils.UsdFileCfg(
        usd_path=DRUM_USD,
        mass_props=MassPropertiesCfg(mass=1.0e2),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=False,
            disable_gravity=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
            contact_offset=0.02,  # 20mm: これを小さくしないと打撃感がフワフワします
            rest_offset=0.0,
        ),
        activate_contact_sensors=True,
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.74573, 0.04029, 0.0),
        rot=quat_from_euler_zyx(yaw_deg=0, pitch_deg=0, roll_deg=0),
    ),
)
