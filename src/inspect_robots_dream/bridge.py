"""Local HTTP bridge to Dream-1: runs `dream.DreamClient` in its own venv and serves it to the Isaac process.

The Dream library needs NumPy 2 and websockets 17 while Isaac Sim 5.1 pins NumPy 1.26 and websockets 12, so the
client cannot live in the Isaac env. This bridge holds one client and one policy session; the `dream` policy in
the Isaac process talks to it over loopback JSON:

    GET  /health    model, versions, whether a session is open
    POST /reset     {"prompt", "action_hz"}: open the policy (or reset_episode()) for a new trial
    POST /observe   {"frames": {camera: b64 jpeg}, "state": [14], "timestamp"}: one set_observation()
    POST /act       {"commanded": [14] | null}: one get_action(); returns q_l_chunk / q_r_chunk

Run it in the Dream venv (`--warm` sends one synthetic request first and waits while the model starts):

    /data/dream/venv/bin/python -m inspect_robots_dream.bridge --warm

Simulated episodes are not recorded to the Dream account (`_record: False`, as Dream's own sim example does);
`--record` turns recording on. Only stdlib + NumPy at import; `dream` is imported when the bridge starts.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from collections.abc import Callable, Mapping
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import numpy as np

from inspect_robots_dream import wire

DEFAULT_MODEL = "dreamlabs/dream-1-beta"
DEFAULT_PORT = 8210
# Errors worth asking again for, besides any DreamServerError the client marks `retryable`.
_RETRY_TYPES = frozenset({"DreamSessionExpired", "DreamTimeout"})


class BridgeError(Exception):
    """A request the bridge cannot serve; `status` is the HTTP status it answers with."""

    def __init__(self, message: str, status: int = 400, **detail: Any) -> None:
        super().__init__(message)
        self.status = status
        self.detail = detail


class Bridge:
    """One Dream client and its policy session, driven one request at a time.

    `client_factory()` returns a `dream.DreamClient`-like object; `sleep` and `clock` are injectable for tests.
    """

    def __init__(
        self,
        client_factory: Callable[[], Any],
        *,
        model: str = DEFAULT_MODEL,
        timeout_ms: int = 45000,
        record: bool = False,
        retry_budget_s: float = 120.0,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        log: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self._client_factory = client_factory
        self._client: Any = None
        self.model = model
        self.timeout_ms = timeout_ms
        self.record = record
        self.retry_budget_s = retry_budget_s
        self._sleep = sleep
        self._clock = clock
        self._log = log or (lambda entry: None)
        self._action_hz: float | None = None
        self._prompt = ""
        self.calls = 0

    def _config(self, action_hz: float) -> dict[str, Any]:
        # Station keys carry no strict-mode embodiment, so strict is off; the policy validates every reply itself.
        return {
            "checkpoint": self.model,
            "timeout_ms": self.timeout_ms,
            "strict": False,
            "action_hz": action_hz,
            "obs_window": wire.OBS_WINDOW,
            "obs_fps": wire.OBS_FPS,
            "_record": self.record,
        }

    def health(self) -> dict[str, Any]:
        return {
            "ok": True,
            "model": self.model,
            "session_open": self._action_hz is not None,
            "action_hz": self._action_hz,
            "record": self.record,
            "calls": self.calls,
        }

    def reset(self, prompt: str, action_hz: float) -> dict[str, Any]:
        """Start a trial: (re)open the policy when the row rate changed or none is open, else end the episode."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise BridgeError("reset needs a non-empty prompt")
        if not isinstance(action_hz, (int, float)) or isinstance(action_hz, bool) or not action_hz > 0:
            raise BridgeError(f"reset needs a positive action_hz, got {action_hz!r}")
        if self._client is None:
            self._client = self._client_factory()
        if self._action_hz != float(action_hz):
            self._client.policy(self._config(float(action_hz)))
            self._action_hz = float(action_hz)
        else:
            self._client.reset_episode()
        self._prompt = prompt
        return {"ok": True, "model": self.model, "action_hz": self._action_hz}

    def observe(self, frames: Mapping[str, str], state: Any, timestamp: float) -> dict[str, Any]:
        """Hand the client one tick: three JPEG frames and the 14-D measured state, at `timestamp` seconds."""
        if self._client is None or self._action_hz is None:
            raise BridgeError("observe before reset")
        try:
            keyed = wire.station_frames(frames)
            obs = wire.station_observation(state, self._prompt)
        except ValueError as error:
            raise BridgeError(str(error)) from None
        if not isinstance(timestamp, (int, float)) or isinstance(timestamp, bool) or not np.isfinite(timestamp):
            raise BridgeError(f"timestamp must be a finite number, got {timestamp!r}")
        self._client.set_observation(keyed, obs, timestamp=float(timestamp))
        return {"ok": True}

    def act(self, commanded: Any = None) -> dict[str, Any]:
        """One get_action(), asked again for retryable refusals within `retry_budget_s`; returns both chunks."""
        if self._client is None or self._action_hz is None:
            raise BridgeError("act before reset")
        last = None if commanded is None else wire.commanded(commanded)
        started = self._clock()
        attempts = 0
        while True:
            attempts += 1
            sent = self._clock()
            try:
                reply = self._client.get_action(commanded=last)
                break
            except Exception as error:  # duck-typed: the dream error types live only in the Dream venv
                name = type(error).__name__
                retryable = name in _RETRY_TYPES or bool(getattr(error, "retryable", False))
                waited = self._clock() - started
                if not retryable or waited >= self.retry_budget_s:
                    self._log({"event": "act_error", "error": name, "message": str(error), "attempts": attempts})
                    raise BridgeError(
                        f"{name}: {error}",
                        status=502,
                        error=name,
                        code=getattr(error, "code", None),
                        reason=getattr(error, "reason", None),
                        retryable=retryable,
                    ) from None
                pause = getattr(error, "retry_after_s", None) or 1.0
                if name == "DreamSessionExpired":
                    pause = 0.0
                self._log({"event": "act_retry", "error": name, "reason": getattr(error, "reason", None), "pause_s": pause})
                print(f"dream bridge: {name} {getattr(error, 'reason', '') or ''}, retrying in {pause:.0f}s", flush=True)
                self._sleep(min(pause, max(self.retry_budget_s - waited, 0.0)))
        rtt_s = self._clock() - sent
        self.calls += 1
        out: dict[str, Any] = {key: np.asarray(reply[key]).tolist() for key in wire.CHUNK_KEYS if key in reply}
        out["keys"] = sorted(reply)
        out["rtt_s"] = rtt_s
        out["attempts"] = attempts
        out["timing"] = _jsonable(getattr(self._client, "last_timing", None))
        out["session_id"] = getattr(self._client, "session_id", None)
        self._log({k: out[k] for k in ("rtt_s", "attempts", "timing", "session_id", "keys")} | {"event": "act"})
        return out

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
        self._client = None
        self._action_hz = None


def _jsonable(value: Any) -> Any:
    """A JSON-safe copy of a timing dict (NumPy scalars become floats); None stays None."""
    if value is None:
        return None
    return json.loads(json.dumps(value, default=lambda v: np.asarray(v).tolist()))


def make_handler(bridge: Bridge) -> type[BaseHTTPRequestHandler]:
    """A request handler serving `bridge`; the single-threaded server keeps to one Dream call at a time."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - the stdlib's name
            pass  # one line per tick would drown the terminal; the JSONL log has the Dream calls

        def _send(self, status: int, body: Mapping[str, Any]) -> None:
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802 - the stdlib's name
            if self.path == "/health":
                self._send(200, bridge.health())
            else:
                self._send(404, {"error": "NotFound", "message": self.path})

        def do_POST(self) -> None:  # noqa: N802 - the stdlib's name
            try:
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(body, dict):
                    raise BridgeError("request body must be a JSON object")
                if self.path == "/reset":
                    result = bridge.reset(body.get("prompt", ""), body.get("action_hz"))
                elif self.path == "/observe":
                    result = bridge.observe(body.get("frames") or {}, body.get("state"), body.get("timestamp"))
                elif self.path == "/act":
                    result = bridge.act(body.get("commanded"))
                else:
                    self._send(404, {"error": "NotFound", "message": self.path})
                    return
                self._send(200, result)
            except BridgeError as error:
                self._send(error.status, {"error": "BridgeError", "message": str(error), **error.detail})
            except (json.JSONDecodeError, ValueError) as error:
                self._send(400, {"error": type(error).__name__, "message": str(error)})
            except Exception as error:  # a Dream config/observation error: report it rather than drop the socket
                self._send(500, {"error": type(error).__name__, "message": str(error)})

    return Handler


def _black_jpeg() -> str:  # pragma: no cover - needs cv2, only in the warm-up
    import cv2

    width, height = wire.FRAME_SIZE
    ok, buf = cv2.imencode(".jpg", np.zeros((height, width, 3), np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("could not encode the warm-up frame")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def warm(bridge: Bridge, max_wait_s: float) -> dict[str, Any]:
    """Send one synthetic request (black frames, home pose) so a stopped model starts; discard the reply."""
    frame = _black_jpeg()
    home = [0.0] * wire.ARM_DOF + [1.0]
    budget, bridge.retry_budget_s = bridge.retry_budget_s, max_wait_s
    try:
        bridge.reset("hold still", wire.STATION_HZ)
        bridge.observe(dict.fromkeys(wire.FRAME_KEYS, frame), home + home, 0.0)
        return bridge.act()
    finally:
        bridge.retry_budget_s = budget


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - process entry point
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="checkpoint id (DreamClient().get_models() lists them)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--timeout-ms", type=int, default=45000)
    parser.add_argument("--retry-budget-s", type=float, default=120.0, help="how long one /act retries refusals")
    parser.add_argument("--record", action="store_true", help="record episodes to the Dream account (off for sim)")
    parser.add_argument("--warm", action="store_true", help="start the model with one synthetic request first")
    parser.add_argument("--warm-max-s", type=float, default=3600.0)
    parser.add_argument("--log", default=None, help="append one JSON line per Dream call to this file")
    args = parser.parse_args(argv)

    import dream

    log_file = open(args.log, "a") if args.log else None  # noqa: SIM115 - lives as long as the server

    def log(entry: dict[str, Any]) -> None:
        if log_file is not None:
            log_file.write(json.dumps({"wall_s": time.time(), **entry}) + "\n")
            log_file.flush()

    models = {m.id: m for m in dream.DreamClient().get_models()}
    if args.model not in models:
        print(f"dream bridge: {args.model!r} is not a model this key may run: {sorted(models)}", file=sys.stderr)
        return 2
    bridge = Bridge(
        dream.DreamClient,
        model=args.model,
        timeout_ms=args.timeout_ms,
        record=args.record,
        retry_budget_s=args.retry_budget_s,
        log=log,
    )
    print(f"dream bridge: dream {dream.__version__}, model {args.model}, recording {'on' if args.record else 'off'}")
    if args.warm:
        print("dream bridge: warming the model (a stopped model can take a while to start)...", flush=True)
        reply = warm(bridge, args.warm_max_s)
        print(f"dream bridge: model answered in {reply['rtt_s']:.2f}s", flush=True)
    server = HTTPServer((args.host, args.port), make_handler(bridge))
    print(f"dream bridge: listening on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        bridge.close()
        if log_file is not None:
            log_file.close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
