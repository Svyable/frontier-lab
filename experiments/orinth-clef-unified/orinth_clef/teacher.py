"""Small synchronous client for the Mac-local Clef MLX SystemOne server."""

import json
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .schema import SchemaError, validate_request, validate_response


class TeacherError(RuntimeError):
    pass


class LocalTeacher:
    def __init__(self, base_url: str = "http://127.0.0.1:8001", timeout: float = 120):
        parsed = urlparse(base_url)
        if parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("the teacher must be a localhost-only HTTP server")
        if not 1 <= (parsed.port or 80) <= 65535 or parsed.username or parsed.password:
            raise ValueError("invalid teacher URL")
        if parsed.path not in ("", "/") or parsed.query or parsed.fragment:
            raise ValueError("teacher URL must be an origin, not a path")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.origin = base_url.rstrip("/")
        self.timeout = timeout

    def health(self) -> dict:
        try:
            with urlopen(self.origin + "/health", timeout=10) as response:  # nosec B310 (loopback)
                return json.load(response)
        except (HTTPError, URLError, TimeoutError, ValueError) as exc:
            raise TeacherError(f"local Clef health check failed: {exc}") from exc

    def predict(self, request: dict) -> dict:
        request = validate_request(request)
        payload = json.dumps({**request, "truncate": False}, allow_nan=False).encode("utf-8")
        wire = Request(self.origin + "/v1/systemone", data=payload, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urlopen(wire, timeout=self.timeout) as stream:  # nosec B310 (loopback)
                response = json.load(stream)
        except HTTPError as exc:
            raise TeacherError(f"Clef HTTP {exc.code}; 413 means the input exceeded context") from exc
        except (URLError, TimeoutError, ValueError) as exc:
            raise TeacherError(f"local Clef request failed: {exc}") from exc
        try:
            return validate_response(request, response)
        except SchemaError as exc:
            raise TeacherError(f"unexpected Clef response contract: {exc}") from exc
