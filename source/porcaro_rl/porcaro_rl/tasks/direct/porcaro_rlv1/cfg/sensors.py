# sensors.py
from __future__ import annotations
from isaaclab.sensors import ContactSensorCfg

# --- センサ定義 ---

contact_forces_stick_at_drum_cfg = ContactSensorCfg(
    prim_path="/World/envs/env_.*/Robot/Stick_link",
    update_period=0.0,
    history_length=32,
    debug_vis=True,
    track_air_time=True,
    track_contact_points=False,
    filter_prim_paths_expr=["/World/envs/env_.*/Drum"],
)

drum_vs_stick_cfg = ContactSensorCfg(
    prim_path="/World/envs/env_.*/Drum",
    update_period=0.0,
    history_length=32,
    debug_vis=True,
    track_air_time=True,
    track_contact_points=False,
    filter_prim_paths_expr=["/World/envs/env_.*/Robot/Stick_link"],
)