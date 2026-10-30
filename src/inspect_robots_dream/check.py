"""Ask Dream-1 for one chunk on the live ``yam_sim`` scene and inspect it without executing a single row.

Run in the Isaac env with the bridge up (``python -m inspect_robots_dream.bridge``)::

    /data/env_isaaclab/bin/python -m inspect_robots_dream.check --out /data/dream/check

It resets the scene, holds the pose through the frame-window pre-roll, requests one chunk, then reports the chunk's
shape, how far its first row is from the measured pose, its per-step joint speed, its gripper range and any entries
past the joint limits. The three camera frames are saved as Dream received them (after JPEG), so the left/right
mapping can be checked by eye. Nothing is executed after the pre-roll unless ``--execute N`` asks to play the
chunk's first N rows, which adds commanded-versus-measured tracking to the report.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path

import numpy as np


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - needs Isaac Sim and a live bridge
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--instruction", default="put the red cube on the plate")
    parser.add_argument("--out", default="dream-check")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--headless", default="true")
    parser.add_argument("--execute", type=int, default=0, help="play this many of the chunk's rows (0: none)")
    args = parser.parse_args(argv)

    import cv2
    from inspect_robots import Scene
    from yam_sim.embodiment import YamSimEmbodiment

    from inspect_robots_dream import wire
    from inspect_robots_dream.policy import DreamPolicy, encode_jpeg

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    width, height = wire.FRAME_SIZE
    emb = YamSimEmbodiment(
        headless=args.headless.lower() != "false", control_hz=wire.STATION_HZ, cam_width=width, cam_height=height
    )
    policy = DreamPolicy()
    policy.bind(emb.info)
    scene = Scene(id="dream-check", instruction=args.instruction, init_seed=args.seed)
    policy.reset(scene)
    obs = emb.reset(scene, seed=args.seed)
    for _ in range(policy.warmup_steps):
        obs = emb.step(policy.act(obs).actions[0]).observation

    measured = wire.as_state(obs.state["joint_pos"])
    first = policy.act(obs)  # the chunk request; nothing below steps the simulator
    rows = np.stack([first.actions[0].data, *policy._queue])  # the queued rows, for inspection only
    low, high = np.asarray(emb.info.action_space.low), np.asarray(emb.info.action_space.high)
    step = np.abs(np.diff(np.vstack([measured, rows]), axis=0))
    labels = [f"{side}_{part}" for side in ("left", "right") for part in (*(f"j{i}" for i in range(6)), "grip")]
    report = {
        "model": policy.info.checkpoint,
        "instruction": args.instruction,
        "rtt_s": first.inference_latency_s,
        "rows": int(rows.shape[0]),
        "measured_pose": dict(zip(labels, np.round(measured, 4).tolist())),
        "first_row_minus_measured": dict(zip(labels, np.round(rows[0] - measured, 4).tolist())),
        "last_row_minus_measured": dict(zip(labels, np.round(rows[-1] - measured, 4).tolist())),
        "max_joint_step_rad": float(step[:, [i for i in range(14) if i not in (6, 13)]].max()),
        "max_joint_speed_rad_s": float(step[:, [i for i in range(14) if i not in (6, 13)]].max() * wire.STATION_HZ),
        "gripper_range": {"left": [float(rows[:, 6].min()), float(rows[:, 6].max())],
                          "right": [float(rows[:, 13].min()), float(rows[:, 13].max())]},
        "entries_past_limits": wire.out_of_range(rows, low, high),
        "transcript": policy.transcript(),
    }
    before = obs
    if args.execute:
        from inspect_robots.types import Action

        played = rows[: args.execute]
        reached = []
        for row in played:
            obs = emb.step(Action(data=row)).observation
            reached.append(wire.as_state(obs.state["joint_pos"]))
        error = np.abs(np.stack(reached) - played)
        joints = [i for i in range(14) if i not in (6, 13)]
        report["executed"] = {
            "rows": int(len(played)),
            # Measured after each 50 ms period versus that period's target: PD lag plus anything blocking the arm.
            "max_joint_tracking_error_rad": float(error[:, joints].max()),
            "mean_joint_tracking_error_rad": float(error[:, joints].mean()),
            "final_joint_error_rad": dict(zip(labels, np.round(error[-1], 4).tolist())),
            "commanded_travel_rad": float(np.abs(played[-1] - measured)[joints].max()),
            "measured_travel_rad": float(np.abs(reached[-1] - measured)[joints].max()),
        }
        np.save(out / "reached_rows.npy", np.stack(reached))
    for name, frame_obs in (("before", before), ("after", obs)):
        rgb = frame_obs.images["top_cam"]
        cv2.imwrite(str(out / f"top_{name}.png"), cv2.cvtColor(np.asarray(rgb, np.uint8), cv2.COLOR_RGB2BGR))
    (out / "chunk.json").write_text(json.dumps(report, indent=2))
    np.save(out / "chunk_rows.npy", rows)
    tiles = []
    for camera in wire.FRAME_KEYS:
        jpeg = base64.b64decode(encode_jpeg(before.images[camera]))
        bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        cv2.imwrite(str(out / f"{camera}.png"), bgr)
        tiles.append(cv2.putText(bgr.copy(), camera, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1))
    cv2.imwrite(str(out / "dream_view.png"), np.hstack(tiles))
    print(json.dumps({k: v for k, v in report.items() if k != "transcript"}, indent=2), flush=True)
    # SimulationApp.close() at interpreter exit can hang a headless app indefinitely; everything is written, so
    # leave without it.
    os._exit(0)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
