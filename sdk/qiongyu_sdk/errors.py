"""Exceptions raised by :mod:`qiongyu_sdk`.

Exception messages deliberately never include request headers.  In particular,
an API token is not copied into an exception, repr, or log message.
"""

from __future__ import annotations


class SDKError(Exception):
    """Base class for all errors raised by the SDK."""


class ConfigurationError(SDKError):
    """The client was configured with an invalid URL or option."""


class SDKTimeoutError(SDKError):
    """The server did not respond within the configured timeout."""


class NetworkError(SDKError):
    """The request could not reach the server."""


class InvalidResponseError(SDKError):
    """The server response did not have the format expected by the SDK."""


class HTTPStatusError(SDKError):
    """The server returned an HTTP error status."""

    def __init__(self, status_code: int, message: str, *, method: str, path: str):
        self.status_code = status_code
        self.method = method
        self.path = path
        # The path is caller supplied but never contains the Authorization header.
        super().__init__(f"{method} {path} failed with HTTP {status_code}: {message}")


class APIError(HTTPStatusError):
    """Alias for a non-success API response, useful for callers catching API errors."""
