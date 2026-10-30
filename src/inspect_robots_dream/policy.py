"""Inspect Robots policy ``dream``: Dream-1 driving the bimanual YAM through the local Dream bridge.

Dream's client keeps a window of camera frames (9 per camera, 0.2 s apart) built from every observation it is
handed, so this policy sees every control step: ``act()`` returns one action at a time, forwards each observation
to the bridge, and asks for a new chunk only when its queue runs out. That mirrors Dream's own simulated-YAM
example (``dream_examples/yam_mujoco``):

1. Pre-roll: hold the measured pose until the frame window is full (32 steps at 20 Hz).
2. Ask for a chunk while the simulator waits (the rollout blocks in ``act()``), then play ``actions`` of its rows,
   one per step, feeding every new observation. Repeat.

Timestamps are the simulated clock (steps / control rate), so the window spacing matches what the model was
trained on regardless of render speed. Each request passes the last row played as ``commanded``, so Dream's seam
join takes over exactly where the arm is headed.

The bridge runs in a separate venv (``python -m inspect_robots_dream.bridge``); this module needs only NumPy,
OpenCV (Isaac Sim's) and the stdlib. Construction is inert; ``bind()`` checks the bridge is up.
"""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
import warnings
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any

import numpy as np
from inspect_robots.policy import PolicyBase, PolicyConfig, PolicyInfo
from inspect_robots.scene import Scene
from inspect_robots.types import Action, ActionChunk, Observation
from inspect_robots_yam.config import DEFAULT_CAMERAS, action_box, observation_space
from inspect_robots_yam.packing import STATE_KEY

from inspect_robots_dream import wire

# (url, payload or None for GET, timeout_s) -> decoded JSON body. Raises BridgeUnavailable / RuntimeError.
PostFn = Callable[[str, Mapping[str, Any] | None, float], Mapping[str, Any]]
# RGB uint8 (H, W, 3) -> base64 JPEG at wire.FRAME_SIZE.
EncodeFn = Callable[[np.ndarray], str]

DEFAULT_SERVER = "http://127.0.0.1:8210"
REMEDY = (
    "start the Dream bridge in its own venv: "
    "/data/dream/venv/bin/python -m inspect_robots_dream.bridge --warm"
)


class BridgeUnavailable(ConnectionError):
    """The bridge did not answer at all (not running, wrong port)."""


def _default_post(  # pragma: no cover - real loopback transport, only against a live bridge
    url: str, payload: Mapping[str, Any] | None, timeout_s: float
) -> Mapping[str, Any]:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            decoded: Mapping[str, Any] = json.loads(response.read())
            return decoded
    except urllib.error.HTTPError as error:
        try:
            body = json.loads(error.read())
        except ValueError:
            body = {}
        raise RuntimeError(f"Dream bridge {error.code}: {body.get('message') or error.reason}") from None
    except (urllib.error.URLError, ConnectionError) as error:
        raise BridgeUnavailable(f"Dream bridge at {url} is not answering ({error}); {REMEDY}") from None


def _import_cv2() -> Any:  # pragma: no cover - Isaac Sim ships cv2; never install a second one into its env
    import cv2

    return cv2


def encode_jpeg(rgb: np.ndarray, cv2: Any = None) -> str:
    """One RGB frame as Dream wants it: resized to 320x176 (area filter), JPEG quality 85, base64 without prefix."""
    cv2 = cv2 or _import_cv2()
    width, height = wire.FRAME_SIZE
    frame = np.asarray(rgb, dtype=np.uint8)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"expected an RGB (H, W, 3) frame, got shape {frame.shape}")
    if frame.shape[:2] != (height, width):
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, wire.JPEG_QUALITY])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return base64.b64encode(np.asarray(buf).tobytes()).decode("ascii")


def _int_arg(name: str, value: Any, low: int, high: int) -> int:
    number = int(value)
    if number != float(value) or not low <= number <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}, got {value!r}")
    return number


class DreamPolicy(PolicyBase):
    """Dream-1 on the bimanual YAM, one action per ``act()``, through the loopback bridge."""

    def __init__(
        self,
        *,
        server_url: str = DEFAULT_SERVER,
        actions: int = wire.MAX_ROWS,
        warmup_steps: int | None = None,
        timeout_s: float = 180.0,
        post_fn: PostFn | None = None,
        encode_fn: EncodeFn | None = None,
    ) -> None:
        # The rollout's connection-failure hint reads `server_url` and `remedy` off the policy (duck-typed).
        self.server_url = str(server_url).rstrip("/")
        self.remedy = REMEDY
        self.actions = _int_arg("actions", actions, 1, wire.MAX_ROWS)
        self._warmup_arg = None if warmup_steps is None else _int_arg("warmup_steps", warmup_steps, 0, 10_000)
        self.timeout_s = float(timeout_s)
        self._post: PostFn = post_fn or _default_post
        self._encode: EncodeFn = encode_fn or encode_jpeg
        self._loop_hz = wire.STATION_HZ
        self._low: np.ndarray | None = None
        self._high: np.ndarray | None = None
        self.info = PolicyInfo(
            name="dream",
            action_space=action_box(),
            observation_space=observation_space(wire.FRAME_SIZE[1], wire.FRAME_SIZE[0], DEFAULT_CAMERAS),
            # Dream's YAM station rate; compat warns when the embodiment runs at another one.
            control_hz=wire.STATION_HZ,
        )
        self.config = PolicyConfig(action_horizon=wire.MAX_ROWS, replan_interval=self.actions)
        self._reset_trial_state()

    # -- lifecycle -----------------------------------------------------------------------

    def bind(self, embodiment_info: Any) -> None:
        """Adopt the embodiment's rate and limits, and learn the bridge's model (fails fast if it is down)."""
        hz = embodiment_info.control_hz
        if hz:
            self._loop_hz = float(hz)
            if abs(self._loop_hz - wire.STATION_HZ) > 1e-6:
                warnings.warn(
                    f"Dream-1 rows are 30 Hz actions that Dream's YAM station plays at {wire.STATION_HZ:g} Hz; "
                    f"this embodiment runs at {self._loop_hz:g} Hz, so motion plays at "
                    f"{self._loop_hz / wire.DREAM_ROW_HZ:.2f}x the trained speed (pass -E control_hz=20)",
                    RuntimeWarning,
                    stacklevel=2,
                )
        box = embodiment_info.action_space
        self._low = None if box.low is None else np.asarray(box.low, dtype=np.float64)
        self._high = None if box.high is None else np.asarray(box.high, dtype=np.float64)
        health = self._post(f"{self.server_url}/health", None, 10.0)
        self.info = replace(self.info, checkpoint=str(health.get("model")))

    def _reset_trial_state(self) -> None:
        self._queue: deque[np.ndarray] = deque()
        self._steps = 0
        self._last_row: np.ndarray | None = None
        self._calls: list[dict[str, Any]] = []
        self._holds = 0

    @property
    def warmup_steps(self) -> int:
        """Steps the pre-roll holds the pose: enough for a full frame window at the loop's rate."""
        if self._warmup_arg is not None:
            return self._warmup_arg
        return (wire.OBS_WINDOW - 1) * max(1, round(self._loop_hz / wire.OBS_FPS))

    def reset(self, scene: Scene) -> None:
        """Start a Dream episode with the scene's instruction; the bridge empties the frame window."""
        if not scene.instruction:
            raise ValueError("the dream policy needs a language instruction")
        self._reset_trial_state()
        self._post(f"{self.server_url}/reset", {"prompt": scene.instruction, "action_hz": self._loop_hz}, 30.0)

    # -- control -------------------------------------------------------------------------

    def act(self, observation: Observation) -> ActionChunk:
        """Feed this step's frames and state to Dream; return the next queued row (asking for a chunk if empty)."""
        if STATE_KEY not in observation.state:
            raise ValueError(f"observation missing state key {STATE_KEY!r}")
        state = wire.as_state(observation.state[STATE_KEY])
        frames = {}
        for camera in wire.FRAME_KEYS:
            image = observation.images.get(camera)
            if image is None:
                raise ValueError(f"observation missing camera {camera!r}")
            frames[camera] = self._encode(image)
        # The simulated clock, not render wall time: the window keeps one frame per 0.2 s of it.
        timestamp = self._steps / self._loop_hz
        self._post(
            f"{self.server_url}/observe", {"frames": frames, "state": state.tolist(), "timestamp": timestamp}, 30.0
        )
        step = self._steps
        self._steps += 1

        if step < self.warmup_steps:
            self._holds += 1
            return ActionChunk(actions=[Action(data=state.copy(), meta={"dream": "warmup_hold"})], control_hz=self._loop_hz)

        latency = None
        if not self._queue:
            commanded = None if self._last_row is None else self._last_row.tolist()
            reply = self._post(f"{self.server_url}/act", {"commanded": commanded}, self.timeout_s)
            rows = wire.parse_chunks(reply)
            latency = float(reply.get("rtt_s") or 0.0) or None
            self._queue.extend(rows[: self.actions])
            timing = reply.get("timing") or {}
            self._calls.append(
                {
                    "t": step,
                    "rtt_s": latency,
                    "serve_infer_ms": timing.get("serve_infer_ms"),
                    "rows": int(rows.shape[0]),
                    "played": min(self.actions, int(rows.shape[0])),
                    "attempts": reply.get("attempts"),
                    "out_of_range": wire.out_of_range(rows[: self.actions], self._low, self._high),
                    "session_id": reply.get("session_id"),
                }
            )
        row = self._queue.popleft()
        self._last_row = row
        meta = {"dream": "chunk_row", "chunk": len(self._calls) - 1}
        return ActionChunk(actions=[Action(data=row.copy(), meta=meta)], control_hz=self._loop_hz, inference_latency_s=latency)

    def transcript(self) -> Any | None:
        """The trial's Dream calls (step, round trip, rows played, clamped entries) and the run settings."""
        rtts = [c["rtt_s"] for c in self._calls if c["rtt_s"]]
        return {
            "model": self.info.checkpoint,
            "loop_hz": self._loop_hz,
            "actions_per_chunk": self.actions,
            "warmup_steps": self.warmup_steps,
            "warmup_holds": self._holds,
            "dream_calls": [dict(c) for c in self._calls],
            "mean_rtt_s": (sum(rtts) / len(rtts)) if rtts else None,
            "steps_seen": self._steps,
            "sim_seconds": self._steps / self._loop_hz if self._loop_hz else None,
        }


def dream_policy(**kwargs: Any) -> DreamPolicy:
    """Policy ``dream``: ``-P actions=N`` (rows per chunk, 1..48), ``-P server_url=...``, ``-P warmup_steps=N``."""
    return DreamPolicy(**kwargs)


__all__ = ["BridgeUnavailable", "DreamPolicy", "dream_policy", "encode_jpeg"]
