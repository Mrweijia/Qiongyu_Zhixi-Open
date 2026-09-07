"""Command-line interface for the Qiongyu Zhixi SDK."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Optional, Sequence

from .client import QiongyuClient
from .errors import SDKError


def _common_parser(default: Any = argparse.SUPPRESS) -> argparse.ArgumentParser:
    """Options accepted both before and after the subcommand.

    The README promises every command accepts --base-url/--token/--timeout, so
    the same options must parse on the main parser (before the subcommand) and
    on every subparser (after it).  The default is SUPPRESS on every parser so
    an unset key is simply absent from the namespace and main() applies the
    fallback once, instead of a subparser default clobbering a value the main
    parser already consumed.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--base-url", default=default, help="API base URL (default: QIONGYU_BASE_URL or http://localhost:5000)")
    parser.add_argument("--token", default=default, help="API token; prefer QIONGYU_API_TOKEN for shell history safety")
    parser.add_argument("--timeout", type=float, default=default, help="HTTP timeout in seconds")
    return parser


def _parser() -> argparse.ArgumentParser:
    common = _common_parser()
    parser = argparse.ArgumentParser(prog="qiongyu",
                                     description="Qiongyu Zhixi API developer tools",
                                     parents=[common])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("health", parents=[common], help="check service and model readiness")
    sub.add_parser("models", parents=[common], help="list available models")

    predict = sub.add_parser("predict", parents=[common], help="upload inputs if needed and run a prediction")
    predict.add_argument("--session-id", help="existing upload session")
    predict.add_argument("--pollution", help="pollution CSV to upload")
    predict.add_argument("--weather", help="weather CSV to upload")

    train = sub.add_parser("train", parents=[common], help="create a training job")
    train.add_argument("--file", required=True, help="training CSV accepted by the online training API")
    train.add_argument("--target", required=True, help="target column in the training CSV")
    train.add_argument("--horizons", nargs="+", type=int, required=True, help="forecast horizons, for example: --horizons 1 2 3")

    status = sub.add_parser("train-status", parents=[common], help="check a training job")
    status.add_argument("job_id")

    sub.add_parser("window", parents=[common],
                   help="show rolling-window coverage, contract and readiness")
    sub.add_parser("window-predict", parents=[common],
                   help="forecast T+1..T+3 from the rolling window (needs 12 recent complete hours)")

    push = sub.add_parser("push-observations", parents=[common],
                          help="append hourly readings from a JSON file into the rolling window")
    push.add_argument("file", help='JSON file holding an observation list or {"observations": [...]}')

    download = sub.add_parser("download", parents=[common], help="download a prediction CSV")
    download.add_argument("identifier", help="prediction session id or /api/download/... path")
    download.add_argument("-o", "--output", required=True, help="destination file")
    download_model = sub.add_parser("download-model", parents=[common], help="download a completed training model")
    download_model.add_argument("job_id", help="training job id")
    download_model.add_argument("-o", "--output", required=True, help="destination file")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    # Subparsers suppress the common defaults so they never clobber a value the
    # main parser already consumed; restore the defaults for keys left unset.
    base_url = getattr(args, "base_url", None)
    token = getattr(args, "token", None)
    timeout = getattr(args, "timeout", 30.0)
    try:
        client = QiongyuClient(base_url=base_url, token=token, timeout=timeout)
        if args.command == "health":
            result = client.health()
        elif args.command == "models":
            result = client.models()
        elif args.command == "predict":
            if args.session_id:
                if args.pollution or args.weather:
                    raise SDKError("use --session-id or --pollution/--weather, not both")
                result = client.predict(args.session_id)
            elif args.pollution and args.weather:
                upload = client.upload(args.pollution, args.weather)
                result = client.predict(upload["session_id"])
            else:
                raise SDKError("predict requires --session-id or both --pollution and --weather")
        elif args.command == "train":
            result = client.train(file=args.file, target=args.target, horizons=args.horizons)
        elif args.command == "train-status":
            result = client.train_status(args.job_id)
        elif args.command == "window":
            result = client.observation_window()
        elif args.command == "window-predict":
            result = client.predict_from_window()
        elif args.command == "push-observations":
            try:
                payload = json.loads(Path(args.file).read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise SDKError(f"cannot read observation file {args.file}: {exc}") from exc
            observations = payload.get("observations") if isinstance(payload, dict) else payload
            if not isinstance(observations, list) or not observations:
                raise SDKError("observation file must hold a non-empty list or an observations key")
            result = client.push_observations(observations)
        elif args.command == "download-model":
            result = client.download_model(args.job_id, args.output)
            print(str(result))
            return 0
        else:
            result = client.download(args.identifier, args.output)
            print(str(result))
            return 0
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except SDKError as exc:
        print(f"qiongyu: {exc}", file=sys.stderr)
        return 1


__all__ = ["main"]
