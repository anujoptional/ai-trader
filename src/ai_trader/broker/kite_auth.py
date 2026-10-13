"""Bounded browser authentication with an in-memory Kite session."""

from __future__ import annotations

import hmac
import secrets
import time
import webbrowser
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlencode, urlsplit

from ai_trader.broker.kite import (
    KiteAuthenticationError,
    KiteBroker,
    _request,
)
from ai_trader.config import KiteLoginSettings

CALLBACK_HOST = "127.0.0.1"
CALLBACK_PORT = 8765
CALLBACK_PATH = "/kite/callback"
REDIRECT_URL = f"http://{CALLBACK_HOST}:{CALLBACK_PORT}{CALLBACK_PATH}"
# A scheduled run opens the login at 08:30 and may wait for the user until 16:30.
MAX_LOGIN_WAIT_SECONDS = 8 * 3600


def login_browser(
    settings: KiteLoginSettings,
    *,
    timeout_seconds: float = 300,
    callback_port: int = CALLBACK_PORT,
    open_browser: Callable[[str], bool] | None = None,
) -> KiteBroker:
    if not 0 < timeout_seconds <= MAX_LOGIN_WAIT_SECONDS:
        raise ValueError(
            "Kite browser login wait must be between zero seconds and 8 hours."
        )
    try:
        from kiteconnect import KiteConnect
    except ImportError:
        raise KiteAuthenticationError(
            "Install the project's declared Kite Connect dependency."
        ) from None
    client = _request(
        lambda: KiteConnect(
            api_key=settings.api_key.get_secret_value(), timeout=30, debug=False
        )
    )
    state = secrets.token_urlsafe(32)
    accepted: dict[str, str] = {}

    class CallbackHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            pass

        def do_GET(self) -> None:
            parsed = urlsplit(self.path)
            query = parse_qs(parsed.query, keep_blank_values=True, max_num_fields=12)
            expected_host = f"{CALLBACK_HOST}:{self.server.server_port}"
            valid = (
                self.headers.get("Host") == expected_host
                and parsed.path == CALLBACK_PATH
                and len(query.get("state", [])) == 1
                and hmac.compare_digest(query["state"][0], state)
                and query.get("status") == ["success"]
                and query.get("action") == ["login"]
                and len(query.get("request_token", [])) == 1
                and 0 < len(query["request_token"][0]) <= 512
            )
            if valid:
                accepted["request_token"] = query["request_token"][0]
            body = (
                b"Kite sign-in received. You can close this tab."
                if valid
                else b"Invalid sign-in callback. Return to the terminal."
            )
            self.send_response(200 if valid else 400)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'none'")
            self.end_headers()
            self.wfile.write(body)

    class CallbackServer(HTTPServer):
        def get_request(self):
            connection, address = super().get_request()
            connection.settimeout(min(10, timeout_seconds))
            return connection, address

        def handle_error(self, request, client_address) -> None:
            pass

    try:
        server = CallbackServer((CALLBACK_HOST, callback_port), CallbackHandler)
    except OSError:
        raise KiteAuthenticationError(
            "Kite callback port is unavailable; close its other listener before login."
        ) from None
    with server:
        login_url = client.login_url()
        parsed_login = urlsplit(login_url)
        if parsed_login.scheme != "https" or parsed_login.netloc != "kite.zerodha.com":
            raise KiteAuthenticationError("Unexpected Kite login endpoint.")
        login_url += "&" + urlencode({"redirect_params": urlencode({"state": state})})
        opener = open_browser or (lambda url: webbrowser.open(url, new=1))
        try:
            opened = opener(login_url)
        except Exception:
            opened = False
        if not opened:
            raise KiteAuthenticationError(
                "Could not open the system browser for Kite sign-in."
            )
        deadline = time.monotonic() + timeout_seconds
        while not accepted:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise KiteAuthenticationError(
                    "Kite browser sign-in timed out; rerun when ready."
                )
            server.timeout = remaining
            server.handle_request()
    request_token = accepted.pop("request_token")
    try:
        session = _request(
            lambda: client.generate_session(
                request_token, api_secret=settings.api_secret.get_secret_value()
            )
        )
        token = session.get("access_token")
        if not isinstance(token, str) or not token:
            raise ValueError
        client.set_access_token(token)
    except Exception:
        raise KiteAuthenticationError(
            "Kite token exchange failed; sign in again."
        ) from None
    return KiteBroker(client)
