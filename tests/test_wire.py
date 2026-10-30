import numpy as np
import pytest

from inspect_robots_dream import wire


def _state() -> np.ndarray:
    # left joints 0.0..0.5, left grip 0.9, right joints 1.0..1.5, right grip 0.2: every slot distinguishable
    return np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 0.2])


def test_packing_matches_inspect_robots_yam() -> None:
    packing = pytest.importorskip("inspect_robots_yam.packing")
    assert wire.TOTAL_DIM == packing.TOTAL_DIM
    assert (wire.LEFT, wire.RIGHT) == (packing.LEFT, packing.RIGHT)
    assert wire.ARM_DOF == packing.ARM_DOF
    config = pytest.importorskip("inspect_robots_yam.config")
    assert tuple(wire.FRAME_KEYS) == config.DEFAULT_CAMERAS


def test_station_observation_maps_left_and_right() -> None:
    obs = wire.station_observation(_state(), "put the red cube on the plate")
    assert obs["state.arm_l.joint_pos"] == pytest.approx([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    assert obs["state.arm_l.gripper_pos"] == pytest.approx([0.9])
    assert obs["state.arm_r.joint_pos"] == pytest.approx([1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
    assert obs["state.arm_r.gripper_pos"] == pytest.approx([0.2])
    assert obs["robot.arm"] == "yam" and obs["robot.exec_head"] == "joint"
    assert obs["robot.gripper_open"] == 1.0
    assert obs["prompt"] == "put the red cube on the plate"
    assert not any(key.startswith(("video.", "image_window.")) for key in obs)


@pytest.mark.parametrize("bad", [np.zeros(13), np.full(14, np.nan)])
def test_station_observation_rejects_bad_state(bad: np.ndarray) -> None:
    with pytest.raises(ValueError):
        wire.station_observation(bad, "x")


def test_station_frames_renames_and_requires_every_camera() -> None:
    frames = {"top_cam": "T", "left_cam": "L", "right_cam": "R"}
    assert wire.station_frames(frames) == {"image_window.top": "T", "image_window.left": "L", "image_window.right": "R"}
    with pytest.raises(ValueError, match="right_cam"):
        wire.station_frames({"top_cam": "T", "left_cam": "L"})
    with pytest.raises(ValueError, match="non-empty"):
        wire.station_frames({**frames, "top_cam": ""})


def test_commanded_splits_the_last_row_per_chunk_key() -> None:
    last = wire.commanded(_state())
    assert last["q_l_chunk"] == pytest.approx(_state()[:7])
    assert last["q_r_chunk"] == pytest.approx(_state()[7:])


def _reply(rows: int = 48) -> dict:
    left = np.tile(_state()[:7], (rows, 1))
    right = np.tile(_state()[7:], (rows, 1))
    return {"q_l_chunk": left.tolist(), "q_r_chunk": right.tolist(), "per_key": {}}


def test_parse_chunks_packs_left_then_right() -> None:
    rows = wire.parse_chunks(_reply())
    assert rows.shape == (48, 14)
    assert rows[0] == pytest.approx(_state())


def test_parse_chunks_clips_grippers_within_slack() -> None:
    reply = _reply(2)
    reply["q_l_chunk"][0][6] = 1.03
    reply["q_r_chunk"][1][6] = -0.02
    rows = wire.parse_chunks(reply)
    assert rows[0, 6] == 1.0 and rows[1, 13] == 0.0


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda r: r.pop("q_r_chunk"), "q_r_chunk"),
        (lambda r: r.update(q_l_chunk=[[0.0] * 6]), r"\[1\.\.48, 7\]"),
        (lambda r: r.update(q_l_chunk=[[0.0] * 7] * 49), r"\[1\.\.48, 7\]"),
        (lambda r: r.update(q_l_chunk=[[float("inf")] * 7]), "non-finite"),
        (lambda r: r["q_l_chunk"][0].__setitem__(6, 40.0), "gripper outside"),
        (lambda r: r.update(q_r_chunk=r["q_r_chunk"][:5]), "differ in shape"),
    ],
)
def test_parse_chunks_rejects_bad_replies(mutate, match) -> None:
    reply = _reply()
    mutate(reply)
    with pytest.raises(ValueError, match=match):
        wire.parse_chunks(reply)


def test_out_of_range_counts_entries_beyond_bounds() -> None:
    rows = np.zeros((2, 14))
    rows[0, 0] = 5.0
    rows[1, 13] = -1.0
    assert wire.out_of_range(rows, np.full(14, -1.0) + 0.5, np.full(14, 1.0)) == 2
    assert wire.out_of_range(rows, None, None) == 0
