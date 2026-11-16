"""Local, non-streaming vLLM reasoning-field compatibility gateway.

The frozen native CLM client consumes ``reasoning_content``; vLLM 0.30
returns ``reasoning``. This gateway changes only that response field name.
It never joins reasoning into content, changes generation parameters, or
changes token usage. The original baseline therefore retains its own
content-only history behavior. No request or response bodies are logged.
"""

from __future__ import annotations

import argparse
import http.client
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

PROTOCOL_VERSION = "reasoning-alias-v1"
MAX_BODY_BYTES = 64 * 1024 * 1024
HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "content-length",
}


class ReasoningAliasError(ValueError):
    """The response cannot be represented by one unambiguous legacy field."""


def _reject_constant(value: str):
    raise ValueError(f"Non-finite JSON constant: {value}")


def normalize_response(body: bytes) -> bytes:
    """Rename reasoning without duplicating it in stored message histories."""
    try:
        response = json.loads(body, parse_constant=_reject_constant)
    except (ValueError, UnicodeError) as exc:
        raise ReasoningAliasError("Upstream returned invalid JSON") from exc
    if not isinstance(response, dict) or not isinstance(response.get("choices"), list):
        raise ReasoningAliasError("Upstream returned no choices list")
    for choice in response["choices"]:
        if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
            raise ReasoningAliasError("Upstream returned an invalid choice message")
        message = choice["message"]
        current = message.get("reasoning")
        legacy = message.get("reasoning_content")
        if any(value is not None and not isinstance(value, str) for value in (current, legacy)):
            raise ReasoningAliasError("Upstream returned a non-string reasoning field")
        if current is not None and legacy is not None and current != legacy:
            raise ReasoningAliasError("Upstream returned conflicting reasoning aliases")
        if current is not None:
            message["reasoning_content"] = current
        message.pop("reasoning", None)
    return json.dumps(response, ensure_ascii=False, allow_nan=False).encode("utf-8")


def upstream_port(upstream: str) -> int:
    """Accept only a literal loopback HTTP origin, never arbitrary proxy targets."""
    try:
        value = urlsplit(upstream)
        port = value.port
    except ValueError as exc:
        raise ValueError("Upstream must be http://127.0.0.1:PORT") from exc
    if (
        value.scheme != "http"
        or value.hostname != "127.0.0.1"
        or value.username is not None
        or value.password is not None
        or value.path not in ("", "/")
        or value.query
        or value.fragment
        or port is None
        or not 1 <= port <= 65535
    ):
        raise ValueError("Upstream must be http://127.0.0.1:PORT")
    return port


class GatewayServer(ThreadingHTTPServer):
    """A node-local adapter with one independent connection per request."""

    daemon_threads = True
    allow_reuse_address = False

    def __init__(self, port: int, upstream: str, timeout: float = 1800):
        if not 0 <= port <= 65535 or not 0 < timeout < float("inf"):
            raise ValueError("Invalid gateway port or timeout")
        self.upstream_port = upstream_port(upstream)
        self.upstream_timeout = timeout
        if port == self.upstream_port:
            raise ValueError("Gateway and upstream ports must differ")
        super().__init__(("127.0.0.1", port), GatewayHandler)


class GatewayHandler(BaseHTTPRequestHandler):
    """Forward the small API surface used by the frozen CLM experiment."""

    server_version = "CLMReasoningAlias/1"
    sys_version = ""

    def log_message(self, format, *args):
        # Never log request paths, credentials, prompts, or model replies.
        pass

    def setup(self):
        super().setup()
        self.connection.settimeout(self.server.upstream_timeout)

    def _send(self, status: int, body: bytes, headers=()):
        self.send_response_only(status)
        connection_tokens = set()
        for key, value in headers:
            if key.lower() == "connection":
                connection_tokens.update(part.strip().lower() for part in value.split(","))
        for key, value in headers:
            if key.lower() not in HOP_HEADERS | connection_tokens:
                self.send_header(key, value)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _error(self, status: int, message: str):
        body = json.dumps({"error": {"message": message, "type": PROTOCOL_VERSION}}).encode()
        self._send(status, body, [("Content-Type", "application/json")])

    def do_GET(self):
        if self.path not in ("/health", "/v1/models"):
            self._error(404, "Unsupported gateway endpoint")
            return
        self._forward(None)

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self._error(404, "Unsupported gateway endpoint")
            return
        if self.headers.get("Transfer-Encoding") is not None:
            self._error(400, "Chunked requests are not supported")
            return
        lengths = self.headers.get_all("Content-Length", [])
        if len(lengths) != 1 or not lengths[0].isdigit():
            self._error(411, "One Content-Length header is required")
            return
        size = int(lengths[0])
        if size > MAX_BODY_BYTES:
            self._error(413, "Request body exceeds the gateway limit")
            return
        try:
            body = self.rfile.read(size)
            request = json.loads(body, parse_constant=_reject_constant)
        except (ValueError, UnicodeError):
            self._error(400, "Request body must be a JSON object")
            return
        except (TimeoutError, OSError):
            self._error(408, "Request body could not be read")
            return
        if len(body) != size or not isinstance(request, dict):
            self._error(400, "Request body must be a complete JSON object")
            return
        if request.get("stream") not in (None, False):
            self._error(400, "Streaming is not supported by reasoning-alias-v1")
            return
        # Forward exactly the bytes received: sampling and history are untouched.
        self._forward(body)

    def _forward(self, body: bytes | None):
        connection_tokens = {
            token.strip().lower()
            for value in self.headers.get_all("Connection", [])
            for token in value.split(",")
        }
        excluded = HOP_HEADERS | connection_tokens | {"host", "accept-encoding"}
        headers = {key: value for key, value in self.headers.items() if key.lower() not in excluded}
        headers["Accept-Encoding"] = "identity"
        connection = http.client.HTTPConnection(
            "127.0.0.1",
            self.server.upstream_port,
            timeout=self.server.upstream_timeout,
        )
        try:
            connection.request(self.command, self.path, body=body, headers=headers)
            response = connection.getresponse()
            result = response.read(MAX_BODY_BYTES + 1)
            if len(result) > MAX_BODY_BYTES:
                self._error(502, "Upstream response exceeds the gateway limit")
                return
            response_headers = response.getheaders()
            if 200 <= response.status < 300 and self.command == "POST":
                result = normalize_response(result)
                response_headers = [
                    (key, value)
                    for key, value in response_headers
                    if key.lower() not in {"content-encoding", "content-md5", "etag"}
                ]
            # In particular, upstream HTTP error statuses and bodies are unchanged.
            self._send(response.status, result, response_headers)
        except ReasoningAliasError:
            self._error(502, "Upstream returned invalid or conflicting reasoning fields")
        except TimeoutError:
            self._error(504, "Upstream request timed out")
        except (OSError, http.client.HTTPException):
            self._error(502, "Upstream request failed")
        finally:
            connection.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--upstream", required=True, help="http://127.0.0.1:PORT")
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    try:
        server = GatewayServer(args.port, args.upstream, args.timeout)
    except ValueError as exc:
        parser.error(str(exc))
    with server:
        print(f"{PROTOCOL_VERSION} listening on 127.0.0.1:{args.port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
