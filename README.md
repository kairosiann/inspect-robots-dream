# inspect-robots-dream

Run Dream Labs' Dream-1 on the bimanual YAM sim (`yam_sim`) in place of the `agent` (Claude Haiku) policy.

Dream's library needs NumPy 2 and websockets 17, and Isaac Sim 5.1 pins NumPy 1.26 and websockets 12. So Dream runs in its own venv as a small local bridge, and the `dream` policy in the Isaac env talks to it over `127.0.0.1:8210`.

```text
yam_sim (Isaac env) -> dream policy -> HTTP 127.0.0.1:8210 -> bridge (Dream venv) -> Dream-1 API
```

## Setup (once)

Put your API key (https://platform.dreamlabs.ai/api-keys) on the instance:
```bash
mkdir -p ~/.dreamlabs && echo 'DREAM_API_KEY=dl_live_...' > ~/.dreamlabs/credentials && chmod 600 ~/.dreamlabs/credentials
```

Clone this repo onto the instance:
```bash
git clone https://github.com/kairosiann/inspect-robots-dream /data/dream/inspect-robots-dream
```

Create the Dream venv and install the bridge:
```bash
uv venv --python 3.11 /data/dream/venv
uv pip install --python /data/dream/venv/bin/python dreamlabs
uv pip install --python /data/dream/venv/bin/python --no-deps -e /data/dream/inspect-robots-dream
```

Install the policy next to the Isaac env, without touching its packages:
```bash
uv pip install --python /data/env_isaaclab/bin/python --target /data/dream/isaac-overlay --no-deps /data/dream/inspect-robots-dream
```

## Run

Start the bridge and leave it running (`--warm` waits while a stopped model starts):
```bash
/data/dream/venv/bin/python -m inspect_robots_dream.bridge --warm --log /data/dream/logs/bridge-calls.jsonl
```

In another terminal, check that Dream answers, without moving the arms:
```bash
PYTHONPATH=/data/dream/isaac-overlay python -m inspect_robots_dream.check --out ~/dream-check
```

Let Dream-1 drive the arms:
```bash
PYTHONPATH=/data/dream/isaac-overlay inspect-robots run --embodiment yam_sim --policy dream \
  -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "put the red cube on the plate" \
  -E headless=false --max-steps 1600 --store-frames --log-dir ~/yam-eval/dream

inspect-robots view ~/yam-eval/dream
```

## Options

- `-P actions=N`: rows played per Dream chunk before asking again (1–48, default 48).
- `-P warmup_steps=N`: steps that hold the pose before the first request (default 32: the 9-frame window at 20 Hz).
- `-P server_url=http://127.0.0.1:8210`: where the bridge listens.
- Bridge `--model dreamlabs/dream-1-beta-flash`: the faster model. `--record` stores episodes on your Dream account (off by default for sim).

## Notes

- Requests use Dream's YAM station keys (`state.arm_l.joint_pos`, `image_window.top`, `robot.arm=yam` ...), as Dream's own sim example does, so the serve computes hand poses in its training frame.
- Dream-1 rows are 30 Hz actions. Dream's YAM station plays them at 20 Hz, and so does this setup (`-E control_hz=20`).
- The sim pauses while Dream answers. Each chunk takes about 0.3–0.5 s to infer, plus the network round trip.
- `max-steps` counts sim steps at 20 Hz: 1600 steps is 80 s, including the 1.6 s pre-roll.
- If the command doesn't return after it prints `log: ...`, press Ctrl+C: Isaac Sim can hang while closing, and the log is already saved.
- Tests: `PYTHONPATH=src:tests /data/env_isaaclab/bin/python -m pytest tests`.
