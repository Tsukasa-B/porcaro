import numpy as np
import torch

# --- ユーザー実験結果 (Summaryより転記) ---
# [Angle, Pressure]
DATA = {
    "PAMDF": {"angle": -2.88,  "pressure": 0.0919}, # Neutral付近
    "PAMF":  {"angle": -84.06, "pressure": 0.2066}, # 屈曲限界 (緩んでいる)
    "PAMG":  {"angle": -39.78, "pressure": 0.0775}, # 把持限界
}

# --- ロボットの幾何学パラメータ (Configと同じにする) ---
r = 0.014   # プーリ半径 [m]
L0 = 0.150  # 自然長 [m]

# 各筋肉のターゲット角度 (pam.py/controller_cfg.pyより)
# これが「幾何学的に最短になる角度」の基準
THETA_T = {
    "PAMDF": 0.0,
    "PAMF": -90.0,
    "PAMG": -45.0
}

def h0_from_pressure(P, Pmax=0.6):
    """圧力から収縮率h0を計算 (pam.py準拠)"""
    # h0 = 0.25 * (P / Pmax)
    return 0.25 * (P / Pmax)

def calculate_l_geo(theta_deg, theta_t_deg, r, L0):
    """現在の角度における幾何学的長さ"""
    theta_rad = np.deg2rad(theta_deg)
    theta_t_rad = np.deg2rad(theta_t_deg)
    
    # pam.py: delta_geo = r * abs(theta - theta_t)
    delta_geo = r * np.abs(theta_rad - theta_t_rad)
    
    # L_geo = L0 - delta_geo
    return L0 - delta_geo

def main():
    print("=== Calculating Wire Slack Offsets ===")
    print("NOTE: Negative offset = Slack (Dead Zone)")
    print("-" * 40)
    
    offsets = []
    
    # 順序: DF, F, G
    order = ["PAMDF", "PAMF", "PAMG"]
    
    for name in order:
        vals = DATA[name]
        theta_curr = vals["angle"]
        p_engage = vals["pressure"]
        theta_t = THETA_T[name]
        
        # 1. 動き出し圧力での「筋肉の収縮長」 (Force=0の境界点)
        # L_muscle = L0 * (1 - h0)
        h0 = h0_from_pressure(p_engage)
        l_muscle_engage = L0 * (1.0 - h0)
        
        # 2. その角度での「幾何学的長さ」 (プーリ上の距離)
        l_geo_curr = calculate_l_geo(theta_curr, theta_t, r, L0)
        
        # 3. オフセットの計算
        # 条件: L_muscle + offset == l_geo_curr
        # Forceが出る条件は h0 > epsilon
        # つまり L_eff < L_muscle
        # pam.py実装上、Slack(不感帯)を作るには、
        # 「見かけの長さ(L_eff)を短く」して「要求収縮率(epsilon)を高く」する必要がある
        # L_eff = L_geo + offset
        # offset = L_muscle - L_geo
        
        offset = l_muscle_engage - l_geo_curr
        
        offsets.append(offset)
        
        print(f"[{name}]")
        print(f"  Engagement Pressure : {p_engage:.4f} MPa -> h0 = {h0:.4f}")
        print(f"  Muscle Length       : {l_muscle_engage*1000:.2f} mm")
        print(f"  Geo Length (@{theta_curr:.1f}): {l_geo_curr*1000:.2f} mm")
        print(f"  => Calculated Offset: {offset*1000:.2f} mm")
        print("-" * 20)

    print("\n" + "="*50)
    print("COPY THIS TO 'actuator_cfg.py' -> PamGeometricCfg")
    print("="*50)
    print(f"wire_slack_offsets: tuple[float, ...] = ({offsets[0]:.5f}, {offsets[1]:.5f}, {offsets[2]:.5f})")
    print("="*50)

if __name__ == "__main__":
    main()