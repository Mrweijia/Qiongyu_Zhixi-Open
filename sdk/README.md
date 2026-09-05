# Qiongyu Zhixi Python SDK

This directory contains a small Python client and CLI for an already deployed
Qiongyu Zhixi API. It is an API integration tool, not a distribution of the
prediction source code or model weights.

## Install

From this directory:

```powershell
py -3.13 -m pip install .
```

For local development, use `py -3.13 -m pip install -e .`. The package has no
runtime dependency beyond Python 3.9+.

To distribute a clean wheel, build from a clean checkout (or remove local
`build/`, `dist/` and `*.egg-info/` first), then install the generated wheel
into a fresh environment:

```powershell
py -3.13 -m build --wheel
py -3.13 -m venv $env:TEMP\qiongyu-sdk-check
& $env:TEMP\qiongyu-sdk-check\Scripts\python.exe -m pip install --no-deps .\dist\qiongyu_zhixi_sdk-0.1.0-py3-none-any.whl
& $env:TEMP\qiongyu-sdk-check\Scripts\qiongyu.exe --help
```

`dist/` is deliberately not versioned. The automated wheel check builds in a
temporary clean source tree, rejects cache files and local paths in the wheel,
and runs both the installed CLI and client outside this repository.

## Configuration

Set configuration in the environment rather than putting credentials in a
script or shell history:

```powershell
$env:QIONGYU_BASE_URL = "http://localhost:5000"
$env:QIONGYU_API_TOKEN = "your-token"
```

An explicit `base_url` or `token` passed to `QiongyuClient` takes precedence.
Tokens are sent as `Authorization: Bearer <token>` and are not printed by the
client or CLI. Do not commit `.env` files or tokens.

## Python usage

```python
from qiongyu_sdk import QiongyuClient

client = QiongyuClient(timeout=30)
print(client.health())
print(client.models())

upload = client.upload("pollution.csv", "weather.csv")
prediction = client.predict(upload["session_id"])
client.download(upload["session_id"], "predictions.csv")

window = client.observation_window()
pushed = client.push_observations([
    {"time": "2026-09-03T08:00:00+08:00", "station": "1335A",
     "pollutants": {"PM2.5": 20.0}, "weather": {"tmp_C": 25.0}},
])
window_prediction = client.predict_from_window()

job = client.train("training.csv", target="pm25", horizons=[1, 2, 3])
status = client.train_status(job["job_id"])
client.download_model(job["job_id"], "model.zip")
```

`download()` returns response bytes when no destination is provided, or writes
the bytes and returns a `pathlib.Path` when a destination is supplied.

## CLI

```powershell
qiongyu health
qiongyu models
qiongyu predict --pollution pollution.csv --weather weather.csv
qiongyu train --file training.csv --target pm25 --horizons 1 2 3
qiongyu train-status JOB_ID
qiongyu window
qiongyu push-observations observations.json
qiongyu window-predict
qiongyu download SESSION_ID --output predictions.csv
qiongyu download-model JOB_ID --output model.zip
```

Every command accepts `--base-url` and `--timeout`; authentication is best
provided with `QIONGYU_API_TOKEN` (or `--token` for a one-off local invocation).
The client raises `APIError` for HTTP error responses, `SDKTimeoutError` for a
timeout, `NetworkError` for connection failures, and `InvalidResponseError` for
malformed JSON.

On the CLI, a completed command exits with `0`; SDK/API/configuration failures
are written to stderr without echoing the token and exit with `1`. Argument
syntax errors use argparse's standard nonzero exit code.

The API paths used are `GET /api/health`, `GET /api/models`, `POST
/api/upload`, `POST /api/predict`, `GET /api/download/<session_id>`, `POST
/api/train`, `GET /api/train/<job_id>`, and `GET
/api/train/<job_id>/download`, `POST /api/v1/observations`, `GET
/api/v1/observations/window` and `POST /api/v1/observations/predict`. Observation pushes and window predictions use the bearer token. Pushes validate stations, known fields, numeric bounds and timestamps and return rejected rows with reasons; full per-station field completeness is a known server-side gap and must not be inferred from `ready_for_prediction` until P0-08 is fixed.

Window prediction, observation push and every other command accept
`--base-url`/`--token`/`--timeout` both before and after the subcommand; their
argument parsing, token forwarding and exit codes are covered by
`tests/test_sdk_cli.py`. Training submissions require the server
contract's multipart `file`, `target`, and `horizons` fields. `download()` is
for prediction CSVs; use `download_model()` for a completed training model.
