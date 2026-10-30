import json
import threading
import urllib.error
import urllib.request
from http.server import HTTPServer

import numpy as np
import pytest

from inspect_robots_dream import bridge as bridge_mod
from inspect_robots_dream import wire

FRAMES = {"top_cam": "T", "left_cam": "L", "right_cam": "R"}
# Every slot distinguishable, grippers inside 0..1: left joints 0.0..0.5 grip 0.9, right joints 1.0..1.5 grip 0.2.
STATE = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 0.2]


class DreamServerError(Exception):
    def __init__(self, reason: str, retryable: bool, retry_after_s: float | None = None) -> None:
        super().__init__(f"DA-2000 {reason}")
        self.code, self.reason, self.retryable, self.retry_after_s = "DA-2000", reason, retryable, retry_after_s


class DreamSessionExpired(Exception):
    pass


class FakeClient:
    """Records calls like dream.DreamClient; `errors` are raised by get_action() before it answers."""

    def __init__(self, errors: list[Exception] | None = None) -> None:
        self.calls: list[tuple] = []
        self.errors = list(errors or [])
        self.last_timing = {"rtt_ms": 400.0, "serve_infer_ms": np.int64(410)}
        self.session_id = "sess-1"

    def policy(self, config):
        self.calls.append(("policy", dict(config)))

    def reset_episode(self):
        self.calls.append(("reset_episode",))

    def set_observation(self, frames, observation, *, timestamp=None):
        self.calls.append(("set_observation", dict(frames), dict(observation), timestamp))

    def get_action(self, *, commanded=None):
        self.calls.append(("get_action", commanded))
        if self.errors:
            raise self.errors.pop(0)
        left, right = np.tile(np.asarray(STATE[:7]), (48, 1)), np.tile(np.asarray(STATE[7:]), (48, 1))
        return {"q_l_chunk": left, "q_r_chunk": right, "per_key": {}}

    def close(self):
        self.calls.append(("close",))


class Clock:
    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def make(client: FakeClient, **kwargs) -> tuple[bridge_mod.Bridge, Clock]:
    clock = Clock()
    b = bridge_mod.Bridge(lambda: client, sleep=clock.sleep, clock=clock, **kwargs)
    return b, clock


def test_reset_opens_policy_once_then_resets_episodes() -> None:
    client = FakeClient()
    b, _ = make(client)
    b.reset("put the red cube on the plate", 20)
    b.reset("put the red cube on the plate", 20)
    b.reset("again", 30)  # a new row rate needs a new policy (action_hz is fixed at policy())
    kinds = [c[0] for c in client.calls]
    assert kinds == ["policy", "reset_episode", "policy"]
    config = client.calls[0][1]
    assert config["checkpoint"] == "dreamlabs/dream-1-beta"
    assert config["action_hz"] == 20.0 and config["_record"] is False and config["strict"] is False
    assert config["obs_window"] == 9 and config["obs_fps"] == 5.0


def test_record_flag_reaches_the_policy_config() -> None:
    client = FakeClient()
    b, _ = make(client, record=True)
    b.reset("x", 20)
    assert client.calls[0][1]["_record"] is True


@pytest.mark.parametrize("prompt, hz", [("", 20), ("x", 0), ("x", None), ("x", True)])
def test_reset_validates(prompt, hz) -> None:
    b, _ = make(FakeClient())
    with pytest.raises(bridge_mod.BridgeError):
        b.reset(prompt, hz)


def test_observe_sends_station_keys_with_prompt() -> None:
    client = FakeClient()
    b, _ = make(client)
    b.reset("put the red cube on the plate", 20)
    b.observe(FRAMES, STATE, 1.25)
    _, frames, obs, timestamp = client.calls[-1]
    assert frames == {"image_window.top": "T", "image_window.left": "L", "image_window.right": "R"}
    assert obs["prompt"] == "put the red cube on the plate"
    assert obs["state.arm_r.joint_pos"] == pytest.approx(STATE[7:13])
    assert timestamp == 1.25


def test_observe_and_act_need_a_reset_first() -> None:
    b, _ = make(FakeClient())
    with pytest.raises(bridge_mod.BridgeError, match="before reset"):
        b.observe(FRAMES, STATE, 0.0)
    with pytest.raises(bridge_mod.BridgeError, match="before reset"):
        b.act()


@pytest.mark.parametrize("timestamp", [float("nan"), "1", True])
def test_observe_rejects_bad_timestamps(timestamp) -> None:
    b, _ = make(FakeClient())
    b.reset("x", 20)
    with pytest.raises(bridge_mod.BridgeError):
        b.observe(FRAMES, STATE, timestamp)


def test_act_returns_chunks_and_passes_commanded() -> None:
    client = FakeClient()
    b, _ = make(client)
    b.reset("x", 20)
    b.observe(FRAMES, STATE, 0.0)
    out = b.act(STATE)
    commanded = client.calls[-1][1]
    assert commanded["q_l_chunk"] == pytest.approx(STATE[:7])
    assert commanded["q_r_chunk"] == pytest.approx(STATE[7:])
    assert np.asarray(out["q_l_chunk"]).shape == (48, 7)
    assert out["attempts"] == 1 and out["session_id"] == "sess-1"
    assert out["timing"]["serve_infer_ms"] == 410
    json.dumps(out)  # the handler must be able to send it
    assert wire.parse_chunks(out).shape == (48, 14)


def test_act_retries_retryable_refusals_within_budget() -> None:
    client = FakeClient([DreamServerError("pool_warming", True, 30.0), DreamSessionExpired("new session")])
    b, clock = make(client)
    b.reset("x", 20)
    b.observe(FRAMES, STATE, 0.0)
    out = b.act()
    assert out["attempts"] == 3
    assert clock.slept == [30.0, 0.0]


def test_act_gives_up_on_non_retryable_and_after_budget() -> None:
    client = FakeClient([DreamServerError("max_serves", False)])
    b, _ = make(client)
    b.reset("x", 20)
    b.observe(FRAMES, STATE, 0.0)
    with pytest.raises(bridge_mod.BridgeError) as info:
        b.act()
    assert info.value.status == 502 and info.value.detail["reason"] == "max_serves"

    client = FakeClient([DreamServerError("pool_warming", True, 50.0)] * 5)
    b, clock = make(client, retry_budget_s=120.0)
    b.reset("x", 20)
    b.observe(FRAMES, STATE, 0.0)
    with pytest.raises(bridge_mod.BridgeError, match="pool_warming"):
        b.act()
    assert sum(clock.slept) == pytest.approx(120.0)


def test_http_round_trip() -> None:
    client = FakeClient()
    b, _ = make(client)
    server = HTTPServer(("127.0.0.1", 0), bridge_mod.make_handler(b))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"

    def call(path, payload=None):
        data = None if payload is None else json.dumps(payload).encode()
        with urllib.request.urlopen(urllib.request.Request(base + path, data=data), timeout=5) as response:
            return json.loads(response.read())

    try:
        assert call("/health")["model"] == "dreamlabs/dream-1-beta"
        assert call("/reset", {"prompt": "x", "action_hz": 20})["action_hz"] == 20.0
        assert call("/observe", {"frames": FRAMES, "state": STATE, "timestamp": 0.0}) == {"ok": True}
        assert len(call("/act", {"commanded": None})["q_r_chunk"]) == 48
        with pytest.raises(urllib.error.HTTPError) as info:
            call("/observe", {"frames": {"top_cam": "T"}, "state": STATE, "timestamp": 0.1})
        assert info.value.code == 400
        assert "right_cam" in json.loads(info.value.read())["message"]
    finally:
        server.shutdown()
        server.server_close()
