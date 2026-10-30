"""The bridge wire: how the bimanual-YAM contract maps onto Dream's YAM station keys. Pure: stdlib + NumPy only.

Both sides import this module: the policy inside the Isaac process, and the bridge inside the Dream venv (whose
NumPy 2 and websockets 17 cannot share the Isaac env). It must stay free of `inspect_robots` and `dream` imports.

Dream-1 on a YAM reads each hand's pose beside the joints. Dream's own simulated-YAM example
(`dream_examples/yam_mujoco/dream_control.py`) sends such a model the YAM station keys, so the serve computes the
poses from the joints in the frame the model was trained in: `state.arm_<l|r>.joint_pos` (6 joints, rad),
`state.arm_<l|r>.gripper_pos` (0 closed, 1 open), `image_window.<top|left|right>` (one 320x176 JPEG per tick) and
the `robot.*` keys below. Its replies are `q_l_chunk` / `q_r_chunk`, `[rows, 7]`: six joint targets in radians,
then the gripper.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

# Mirrors inspect_robots_yam.packing (left then right, `[j0..j5, gripper]` per arm); tests pin the two together.
ARM_DOF = 6
ARM_WIDTH = ARM_DOF + 1
TOTAL_DIM = 2 * ARM_WIDTH
LEFT = slice(0, ARM_WIDTH)
RIGHT = slice(ARM_WIDTH, TOTAL_DIM)

# Embodiment camera -> Dream station frame key (the top camera is the model's `overhead` role).
FRAME_KEYS: Mapping[str, str] = {
    "top_cam": "image_window.top",
    "left_cam": "image_window.left",
    "right_cam": "image_window.right",
}
# Every frame the model sees: width x height, and the JPEG quality, as Dream's YAM station and sim example send.
FRAME_SIZE = (320, 176)
JPEG_QUALITY = 85

# Station keys telling the serve the arm is a YAM running the model's joint head, gripper open at 1. A joint
# speed cap of 0 means the arm spec's own caps (Dream's sim example sends the same).
STATION_ROBOT: Mapping[str, Any] = {
    "robot.arm": "yam",
    "robot.exec_head": "joint",
    "robot.max_joint_vel_rad_s": 0.0,
    "robot.gripper_open": 1.0,
}
SLOTS = (("arm_l", LEFT), ("arm_r", RIGHT))
CHUNK_KEYS = {"q_l_chunk": LEFT, "q_r_chunk": RIGHT}

# Rows in one Dream-1 chunk; the client never returns more.
MAX_ROWS = 48
# Dream-1's rows are 30 Hz actions. Dream's YAM station plays them at 20 Hz (two thirds speed) by default.
DREAM_ROW_HZ = 30.0
STATION_HZ = 20.0
# Frames the Dream client keeps per camera, and per second of observation timestamps (its obs_window, obs_fps).
OBS_WINDOW = 9
OBS_FPS = 5.0
# A gripper reply this far outside 0..1 means a wrong contract, not a target to clamp.
GRIPPER_SLACK = 0.05

Vec = npt.NDArray[np.float64]


def as_state(vec: Any) -> Vec:
    """A finite flat 14-D float vector, or ValueError."""
    arr = np.asarray(vec, dtype=np.float64).reshape(-1)
    if arr.shape != (TOTAL_DIM,):
        raise ValueError(f"expected a {TOTAL_DIM}-D bimanual vector, got shape {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("bimanual vector has non-finite entries")
    return arr


def station_observation(state: Any, prompt: str) -> dict[str, Any]:
    """The non-frame half of one `set_observation()`: both arms in station keys, the robot keys and the prompt."""
    arr = as_state(state)
    obs: dict[str, Any] = {"prompt": prompt}
    for slot, part in SLOTS:
        arm = arr[part]
        obs[f"state.{slot}.joint_pos"] = arm[:ARM_DOF].tolist()
        obs[f"state.{slot}.gripper_pos"] = arm[ARM_DOF:].tolist()
    obs.update(STATION_ROBOT)
    return obs


def station_frames(frames: Mapping[str, str]) -> dict[str, str]:
    """Rename `{embodiment camera: base64 JPEG}` to the station frame keys; every camera must be present."""
    missing = sorted(set(FRAME_KEYS) - set(frames))
    if missing:
        raise ValueError(f"frames missing cameras {missing}")
    out = {}
    for camera, key in FRAME_KEYS.items():
        jpeg = frames[camera]
        if not isinstance(jpeg, str) or not jpeg:
            raise ValueError(f"frame {camera!r} must be a non-empty base64 JPEG string")
        out[key] = jpeg
    return out


def commanded(row: Any) -> dict[str, Vec]:
    """`get_action(commanded=...)` for the last 14-D row the loop sent: each arm's joints, then its gripper."""
    arr = as_state(row)
    return {key: arr[part].copy() for key, part in CHUNK_KEYS.items()}


def parse_chunks(reply: Mapping[str, Any]) -> Vec:
    """Validate Dream's left/right chunks and pack them into `[rows, 14]` bimanual rows.

    Joints are passed through (the embodiment clamps to its own limits); grippers are clipped to 0..1 once
    they are within `GRIPPER_SLACK` of it, since anything further out means the reply is in another convention.
    """
    arms = {}
    for key in CHUNK_KEYS:
        if key not in reply:
            raise ValueError(f"Dream reply has no {key!r} (keys: {sorted(reply)})")
        rows = np.asarray(reply[key], dtype=np.float64)
        if rows.ndim != 2 or rows.shape[1] != ARM_WIDTH or not 1 <= rows.shape[0] <= MAX_ROWS:
            raise ValueError(f"{key} must be [1..{MAX_ROWS}, {ARM_WIDTH}], got shape {rows.shape}")
        if not np.isfinite(rows).all():
            raise ValueError(f"{key} has non-finite values")
        grip = rows[:, ARM_DOF]
        if (grip < -GRIPPER_SLACK).any() or (grip > 1 + GRIPPER_SLACK).any():
            raise ValueError(f"{key} gripper outside 0..1 (min {grip.min():.3f}, max {grip.max():.3f})")
        rows[:, ARM_DOF] = np.clip(grip, 0.0, 1.0)
        arms[key] = rows
    left, right = arms["q_l_chunk"], arms["q_r_chunk"]
    if left.shape != right.shape:
        raise ValueError(f"left and right chunks differ in shape: {left.shape} vs {right.shape}")
    return np.concatenate([left, right], axis=1)


def out_of_range(rows: Vec, low: Sequence[float] | None, high: Sequence[float] | None) -> int:
    """How many entries of `rows` lie outside `[low, high]` (which the embodiment will clamp); 0 without bounds."""
    if low is None or high is None:
        return 0
    lo, hi = np.asarray(low, dtype=np.float64), np.asarray(high, dtype=np.float64)
    return int(((rows < lo) | (rows > hi)).sum())
