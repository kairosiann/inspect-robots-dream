import base64
import json
import warnings

import numpy as np
import pytest

pytest.importorskip("inspect_robots_yam")

from inspect_robots import Scene  # noqa: E402
from inspect_robots.embodiment import EmbodimentInfo  # noqa: E402
from inspect_robots.policy import Policy  # noqa: E402
from inspect_robots.types import Observation  # noqa: E402
from inspect_robots_yam.config import DEFAULT_CAMERAS, action_box, observation_space  # noqa: E402
from inspect_robots_yam.packing import pack  # noqa: E402

from inspect_robots_dream import bridge as bridge_mod  # noqa: E402
from inspect_robots_dream import policy as policy_mod  # noqa: E402

from test_bridge import STATE, FakeClient  # noqa: E402

READY = np.array([0.0, 1.0, 1.0, -0.6, 0.0, 0.0, 1.0])
LOW = pack(np.array([-2.6, 0, 0, -1.6, -1.5, -2.0, 0.0]), np.array([-2.6, 0, 0, -1.6, -1.5, -2.0, 0.0]))
HIGH = pack(np.array([3.1, 3.6, 3.1, 1.5, 1.5, 2.0, 1.0]), np.array([3.1, 3.6, 3.1, 1.5, 1.5, 2.0, 1.0]))


class LoopbackBridge:
    """A post_fn that routes the policy's requests to a real Bridge over a FakeClient, through JSON."""

    def __init__(self, client: FakeClient | None = None) -> None:
        self.client = client or FakeClient()
        self.bridge = bridge_mod.Bridge(lambda: self.client, sleep=lambda s: None)
        self.requests: list[tuple[str, dict | None]] = []

    def __call__(self, url, payload, timeout_s):
        path = url.rsplit("/", 1)[-1]
        body = None if payload is None else json.loads(json.dumps(payload))
        self.requests.append((path, body))
        if path == "health":
            result = self.bridge.health()
        elif path == "reset":
            result = self.bridge.reset(body["prompt"], body["action_hz"])
        elif path == "observe":
            result = self.bridge.observe(body["frames"], body["state"], body["timestamp"])
        else:
            result = self.bridge.act(body["commanded"])
        return json.loads(json.dumps(result))

    def paths(self) -> list[str]:
        return [path for path, _ in self.requests]


def emb_info(control_hz: float = 20.0) -> EmbodimentInfo:
    return EmbodimentInfo(
        name="yam_sim",
        action_space=action_box(LOW, HIGH),
        observation_space=observation_space(176, 320, DEFAULT_CAMERAS),
        control_hz=control_hz,
        is_simulated=True,
    )


def obs(state: np.ndarray | None = None) -> Observation:
    state = pack(READY, READY) if state is None else state
    images = {cam: np.zeros((176, 320, 3), np.uint8) for cam in DEFAULT_CAMERAS}
    return Observation(images=images, state={"joint_pos": state})


def make(control_hz: float = 20.0, **kwargs) -> tuple[policy_mod.DreamPolicy, LoopbackBridge]:
    loop = LoopbackBridge()
    p = policy_mod.DreamPolicy(post_fn=loop, encode_fn=lambda rgb: "jpeg", **kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p.bind(emb_info(control_hz))
    p.reset(Scene(id="s", instruction="put the red cube on the plate"))
    return p, loop


def test_construction_is_inert_and_conforms() -> None:
    calls = []
    p = policy_mod.DreamPolicy(post_fn=lambda *a: calls.append(a) or {})
    assert calls == []
    assert isinstance(p, Policy)
    assert p.info.name == "dream" and p.info.action_space.dim == 14
    assert p.info.observation_space.camera_names == frozenset(DEFAULT_CAMERAS)
    assert p.config.replan_interval == 48


@pytest.mark.parametrize("bad", [{"actions": 0}, {"actions": 49}, {"actions": 2.5}, {"warmup_steps": -1}])
def test_constructor_validates(bad) -> None:
    with pytest.raises(ValueError):
        policy_mod.DreamPolicy(**bad)


def test_bind_reads_model_and_warns_off_station_rate() -> None:
    loop = LoopbackBridge()
    p = policy_mod.DreamPolicy(post_fn=loop)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        p.bind(emb_info(20.0))
    assert p.info.checkpoint == "dreamlabs/dream-1-beta"
    with pytest.warns(RuntimeWarning, match="0.33x the trained speed"):
        p.bind(emb_info(10.0))


def test_warmup_holds_then_plays_whole_chunk_and_replans() -> None:
    p, loop = make()
    assert p.warmup_steps == 32  # 8 gaps of 0.2 s at 20 Hz fill the 9-frame window
    measured = pack(READY, READY)
    for _ in range(32):
        chunk = p.act(obs(measured))
        assert len(chunk) == 1 and chunk.actions[0].meta["dream"] == "warmup_hold"
        np.testing.assert_allclose(chunk.actions[0].data, measured)
    assert "act" not in loop.paths()

    first = p.act(obs())
    assert loop.paths()[-1] == "act" and loop.requests[-1][1] == {"commanded": None}
    assert first.inference_latency_s is not None and first.control_hz == 20.0
    for _ in range(47):
        chunk = p.act(obs())
        assert chunk.inference_latency_s is None
    assert loop.paths().count("act") == 1

    p.act(obs())  # queue empty: a second request, continuing from the last row played
    assert loop.paths().count("act") == 2
    commanded = loop.client.calls[-1][1]
    np.testing.assert_allclose(commanded["q_l_chunk"], _row0()[:7])
    np.testing.assert_allclose(commanded["q_r_chunk"], _row0()[7:])


def _row0() -> np.ndarray:
    # FakeClient answers every row of q_l_chunk with STATE[:7] and of q_r_chunk with STATE[7:]
    return np.asarray(STATE, dtype=np.float64)


def test_rows_are_packed_left_then_right() -> None:
    p, _ = make(warmup_steps=0)
    np.testing.assert_allclose(p.act(obs()).actions[0].data, _row0())


def test_actions_option_replans_early() -> None:
    p, loop = make(warmup_steps=0, actions=8)
    for _ in range(24):
        p.act(obs())
    assert loop.paths().count("act") == 3


def test_every_step_is_observed_on_the_simulated_clock() -> None:
    p, loop = make(control_hz=10.0)
    assert p.warmup_steps == 16  # 0.2 s is two steps at 10 Hz
    for _ in range(5):
        p.act(obs())
    stamps = [body["timestamp"] for path, body in loop.requests if path == "observe"]
    assert stamps == pytest.approx([0.0, 0.1, 0.2, 0.3, 0.4])
    reset = [body for path, body in loop.requests if path == "reset"][0]
    assert reset == {"prompt": "put the red cube on the plate", "action_hz": 10.0}


def test_reset_starts_a_fresh_episode() -> None:
    p, loop = make(warmup_steps=0)
    p.act(obs())
    p.reset(Scene(id="s", instruction="again"))
    assert p.transcript()["dream_calls"] == []
    p.act(obs())
    assert loop.requests[-1] == ("act", {"commanded": None})
    with pytest.raises(ValueError, match="instruction"):
        p.reset(Scene(id="s", instruction=""))


def test_bad_reply_and_missing_inputs_raise() -> None:
    p, loop = make(warmup_steps=0)
    loop.client.get_action = lambda commanded=None: {"q_l_chunk": [[0.0] * 7], "q_r_chunk": [[0.0] * 6]}
    with pytest.raises(ValueError, match="q_r_chunk"):
        p.act(obs())
    p, _ = make(warmup_steps=0)
    bad = obs()
    with pytest.raises(ValueError, match="right_cam"):
        p.act(Observation(images={k: v for k, v in bad.images.items() if k != "right_cam"}, state=bad.state))
    with pytest.raises(ValueError, match="joint_pos"):
        p.act(Observation(images=bad.images, state={}))


def test_transcript_records_calls_and_clamped_entries() -> None:
    p, loop = make(warmup_steps=2)
    row = np.asarray(STATE[:7])
    beyond = row.copy()
    beyond[3] = 2.0  # j3 past its 1.5 limit: the embodiment will clamp it
    loop.client.get_action = lambda commanded=None: {"q_l_chunk": [beyond] * 10, "q_r_chunk": [row] * 10}
    for _ in range(3):
        p.act(obs())
    t = p.transcript()
    assert t["model"] == "dreamlabs/dream-1-beta" and t["warmup_holds"] == 2
    (call,) = t["dream_calls"]
    assert call["t"] == 2 and call["rows"] == 10 and call["played"] == 10
    assert call["out_of_range"] == 10
    json.dumps(t)


def test_encode_jpeg_matches_dream_frames() -> None:
    cv2 = pytest.importorskip("cv2")
    rgb = np.zeros((360, 640, 3), np.uint8)
    rgb[:, :320, 0] = 255  # left half red
    data = base64.b64decode(policy_mod.encode_jpeg(rgb, cv2=cv2))
    decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == (176, 320, 3)
    assert decoded[88, 80, 2] > 200 and decoded[88, 80, 0] < 50  # still red after the RGB->BGR round trip
