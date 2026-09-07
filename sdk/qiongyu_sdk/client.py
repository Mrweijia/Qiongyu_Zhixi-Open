"""Small, dependency-free HTTP client for the Qiongyu Zhixi API."""

from __future__ import annotations

import json
import mimetypes
import os
from pathlib import Path
import secrets
from typing import Any, BinaryIO, Mapping, Optional, Sequence, Tuple, Union
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import Request, build_opener

from .errors import (
    APIError,
    ConfigurationError,
    HTTPStatusError,
    InvalidResponseError,
    NetworkError,
    SDKTimeoutError,
)

FileInput = Union[str, os.PathLike[str], BinaryIO, Tuple[str, Union[bytes, bytearray, BinaryIO, str, os.PathLike[str]]]]


class QiongyuClient:
    """Call the prediction and training API.

    Parameters can be passed explicitly or read from ``QIONGYU_BASE_URL`` and
    ``QIONGYU_API_TOKEN``.  Explicit values take precedence.  A token, when
    configured, is sent as ``Authorization: Bearer ...`` and is never included
    in exceptions or normal output.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        token: Optional[str] = None,
        timeout: float = 30.0,
        *,
        opener: Any = None,
    ) -> None:
        raw_url = base_url if base_url is not None else os.environ.get("QIONGYU_BASE_URL", "http://localhost:5000")
        if not isinstance(raw_url, str) or not raw_url.strip():
            raise ConfigurationError("base_url must be a non-empty HTTP(S) URL")
        parsed = urlsplit(raw_url.strip())
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ConfigurationError("base_url must be a non-empty HTTP(S) URL")
        if parsed.username is not None or parsed.password is not None:
            raise ConfigurationError("base_url must not contain embedded credentials")
        if parsed.query or parsed.fragment:
            raise ConfigurationError("base_url must not contain a query string or fragment")
        try:
            timeout_value = float(timeout)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError("timeout must be a positive number") from exc
        if timeout_value <= 0:
            raise ConfigurationError("timeout must be a positive number")
        self.base_url = raw_url.rstrip("/")
        self.token = token if token is not None else os.environ.get("QIONGYU_API_TOKEN")
        self.timeout = timeout_value
        self._opener = opener or build_opener()

    def __repr__(self) -> str:
        return f"QiongyuClient(base_url={self.base_url!r}, timeout={self.timeout!r}, token_configured={bool(self.token)})"

    def health(self) -> dict[str, Any]:
        """Return service and model readiness information."""
        return self._json("GET", "/api/health")

    def models(self) -> Any:
        """Return the models exposed by the service."""
        return self._json("GET", "/api/models")

    def upload(self, pollution_file: FileInput, weather_file: FileInput) -> dict[str, Any]:
        """Upload and validate the pollution and weather CSV inputs."""
        fields, files = self._file_parts(
            (("pollution_file", pollution_file), ("weather_file", weather_file))
        )
        body, headers = self._multipart(fields, files)
        return self._json("POST", "/api/upload", body=body, headers=headers)

    def predict(self, session_id: str, **payload: Any) -> dict[str, Any]:
        """Run a prediction for an uploaded session.

        Additional keyword arguments are passed through as JSON, which keeps
        this client compatible with optional server-side prediction settings.
        """
        if not session_id or not isinstance(session_id, str):
            raise ConfigurationError("session_id must be a non-empty string")
        data = {"session_id": session_id, **payload}
        return self._json("POST", "/api/predict", json_body=data)

    def push_observations(self, observations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Append hourly readings into the server's rolling prediction window.

        The endpoint is authenticated with the configured bearer token and
        validates each record against the deployed model's station and field
        contract, so unusable rows come back in ``rejected`` with a reason
        instead of disappearing.  Nothing is raised here: a partially accepted
        batch is a normal outcome for live sensor data.
        """
        if isinstance(observations, Mapping) or not observations:
            raise ConfigurationError("observations must be a non-empty sequence of mappings")
        items = []
        for index, item in enumerate(observations):
            if not isinstance(item, Mapping):
                raise ConfigurationError(f"observations[{index}] must be a mapping")
            items.append(dict(item))
        return self._json("POST", "/api/v1/observations",
                          json_body={"observations": items})

    def observation_window(self) -> dict[str, Any]:
        """Return stored coverage, the model input contract and readiness."""
        return self._json("GET", "/api/v1/observations/window")

    def predict_from_window(self) -> dict[str, Any]:
        """Forecast T+1..T+3 from the rolling window.

        Requires a recent run of 12 consecutive complete hours; an incomplete
        window comes back as an APIError carrying the 409 coverage reasons.
        """
        return self._json("POST", "/api/v1/observations/predict", json_body={})

    def train(
        self,
        file: FileInput,
        target: str,
        horizons: Sequence[int],
    ) -> dict[str, Any]:
        """Create a training job.

        The deployed training endpoint accepts one CSV in the multipart ``file``
        field, plus the required ``target`` and ``horizons`` form fields.  These
        are the only training inputs exposed by this SDK; training configuration
        and model implementation details remain server-side.
        """
        if not isinstance(target, str) or not target.strip():
            raise ConfigurationError("target must be a non-empty string")
        if isinstance(horizons, (str, bytes, bytearray)) or not horizons:
            raise ConfigurationError("horizons must be a non-empty sequence of integers")
        if any(isinstance(item, bool) or not isinstance(item, int) or item < 1 for item in horizons):
            raise ConfigurationError("horizons must contain positive integers")
        fields = [("target", target.strip()), ("horizons", json.dumps(list(horizons)))]
        _, files = self._file_parts((("file", file),))
        body, headers = self._multipart(fields, files)
        return self._json("POST", "/api/train", body=body, headers=headers)

    def train_status(self, job_id: str) -> dict[str, Any]:
        """Return status and metadata for a training job."""
        return self._json("GET", f"/api/train/{self._identifier(job_id)}")

    # A discoverable alias for users who prefer a verb phrase.
    get_train_status = train_status

    def download(self, identifier: str, destination: Optional[Union[str, os.PathLike[str]]] = None) -> Union[bytes, Path]:
        """Download a prediction CSV by session id, returning bytes or saving to a file.

        ``identifier`` may also be an API path or a relative download URL from
        a JSON response.  Absolute URLs are restricted to this client's origin
        so a server response cannot redirect the SDK to an unexpected host.
        """
        path = self._download_path(identifier)
        content = self._bytes("GET", path)
        if destination is None:
            return content
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        return target

    def download_model(self, job_id: str, destination: Optional[Union[str, os.PathLike[str]]] = None) -> Union[bytes, Path]:
        """Download the trained model artifact associated with ``job_id``."""
        path = f"/api/train/{self._identifier(job_id)}/download"
        content = self._bytes("GET", path)
        if destination is None:
            return content
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        return target

    download_training_model = download_model

    def _identifier(self, value: str) -> str:
        if not value or not isinstance(value, str):
            raise ConfigurationError("identifier must be a non-empty string")
        return quote(value, safe="")

    def _download_path(self, identifier: str) -> str:
        if not identifier or not isinstance(identifier, str):
            raise ConfigurationError("identifier must be a non-empty string")
        if identifier.startswith("/"):
            path = identifier
        else:
            split = urlsplit(identifier)
            if split.scheme or split.netloc:
                configured = urlsplit(self.base_url)
                if (split.scheme, split.netloc) != (configured.scheme, configured.netloc):
                    raise ConfigurationError("download URL must use the client's base URL")
                path = split.path + (("?" + split.query) if split.query else "")
            else:
                path = f"/api/download/{quote(identifier, safe='')}"
        valid_download = path == "/api/download" or path.startswith("/api/download/")
        if not valid_download and not path.startswith("/api/train/"):
            raise ConfigurationError("download path must target an API download endpoint")
        return path

    def _url(self, path: str) -> str:
        if not path.startswith("/"):
            path = "/" + path
        return self.base_url + path

    def _headers(self, extra: Optional[Mapping[str, str]] = None) -> dict[str, str]:
        headers = {"Accept": "application/json", "User-Agent": "qiongyu-zhixi-sdk/0.1.0"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if extra:
            headers.update(extra)
        return headers

    def _json(
        self,
        method: str,
        path: str,
        *,
        json_body: Any = None,
        body: Optional[bytes] = None,
        headers: Optional[Mapping[str, str]] = None,
    ) -> dict[str, Any]:
        if json_body is not None:
            body = json.dumps(json_body, ensure_ascii=False).encode("utf-8")
            merged = {"Content-Type": "application/json; charset=utf-8", **(headers or {})}
        else:
            merged = dict(headers or {})
        raw = self._request(method, path, body=body, headers=merged)
        if not raw:
            return {}
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise InvalidResponseError(f"{method} {path} returned invalid JSON") from exc
        if not isinstance(parsed, dict):
            raise InvalidResponseError(f"{method} {path} returned JSON that is not an object")
        return parsed

    def _bytes(self, method: str, path: str) -> bytes:
        return self._request(method, path, headers={"Accept": "application/octet-stream, text/csv"})

    def _request(self, method: str, path: str, *, body: Optional[bytes] = None, headers: Optional[Mapping[str, str]] = None) -> bytes:
        request = Request(self._url(path), data=body, headers=self._headers(headers), method=method.upper())
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                return response.read()
        except HTTPError as exc:
            # Do not echo arbitrary response bodies: an upstream proxy may put
            # sensitive request data in an error page.
            try:
                payload = exc.read(512).decode("utf-8", "replace")
                detail = self._error_detail(payload)
            except Exception:
                detail = "server returned an error"
            if not detail:
                detail = "server returned an error"
            if self.token:
                detail = detail.replace(self.token, "[redacted]")
            raise APIError(exc.code, detail, method=method.upper(), path=path) from exc
        except (TimeoutError, URLError) as exc:
            reason = getattr(exc, "reason", exc)
            if isinstance(exc, TimeoutError) or isinstance(reason, TimeoutError):
                raise SDKTimeoutError(f"{method.upper()} {path} timed out after {self.timeout:g}s") from exc
            raise NetworkError(f"{method.upper()} {path} could not reach the server") from exc

    @staticmethod
    def _error_detail(payload: str) -> str:
        try:
            parsed = json.loads(payload)
            if isinstance(parsed, dict) and isinstance(parsed.get("error"), str):
                return parsed["error"][:300]
        except json.JSONDecodeError:
            pass
        return "server returned an error"

    @staticmethod
    def _content(value: Any) -> bytes:
        if isinstance(value, (bytes, bytearray)):
            return bytes(value)
        if isinstance(value, (str, os.PathLike)):
            return Path(value).read_bytes()
        if hasattr(value, "read"):
            data = value.read()
            return data if isinstance(data, bytes) else str(data).encode("utf-8")
        raise ConfigurationError("file input must be a path, bytes, or binary file object")

    @classmethod
    def _file_parts(cls, parts: Sequence[Tuple[str, FileInput]]) -> Tuple[list[Tuple[str, str]], list[Tuple[str, bytes, str]]]:
        fields: list[Tuple[str, str]] = []
        files: list[Tuple[str, bytes, str]] = []
        for field, value in parts:
            if isinstance(value, tuple):
                if len(value) != 2:
                    raise ConfigurationError("file tuple must be (filename, content)")
                filename, content = value
                filename = str(filename)
                data = cls._content(content)
            else:
                filename = Path(getattr(value, "name", "input.csv")).name if hasattr(value, "name") else Path(value).name if isinstance(value, (str, os.PathLike)) else "input.csv"
                data = cls._content(value)
            filename = filename.replace("\\", "/").rsplit("/", 1)[-1]
            filename = filename.replace('"', "_").replace("\r", "_").replace("\n", "_")
            if not filename:
                raise ConfigurationError("uploaded filename must not be empty")
            mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"
            files.append((field, data, filename + "\0" + mime))
        return fields, files

    @staticmethod
    def _multipart(fields: Sequence[Tuple[str, str]], files: Sequence[Tuple[str, bytes, str]]) -> Tuple[bytes, dict[str, str]]:
        boundary = "----qiongyu-" + secrets.token_hex(12)
        chunks: list[bytes] = []
        for name, value in fields:
            chunks.extend([f"--{boundary}\r\n".encode(), f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(), str(value).encode(), b"\r\n"])
        for name, data, filename_mime in files:
            filename, _, mime = filename_mime.partition("\0")
            chunks.extend([f"--{boundary}\r\n".encode(), f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'.encode(), f"Content-Type: {mime}\r\n\r\n".encode(), data, b"\r\n"])
        chunks.append(f"--{boundary}--\r\n".encode())
        return b"".join(chunks), {"Content-Type": f"multipart/form-data; boundary={boundary}"}


__all__ = ["QiongyuClient"]
