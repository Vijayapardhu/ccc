#!/usr/bin/env python3
"""ISAPI client for the XMEye / Hikvision-lineage campus cameras.

Implements the authentication protocol recovered from the device's own web UI
(see ``XMEYE_WEB_API.md``), plus SADP discovery for finding a unit on the LAN.

Why this exists: the capture pipeline in ``src/campus`` needs credentials that
work, and a 400-camera fleet needs a way to verify and reconfigure devices
without a browser. This is that tool. It does *not* guess credentials.

    pip install requests

    python tools/isapi_camera.py discover
    python tools/isapi_camera.py verify   --host 10.0.0.9 --user admin
    python tools/isapi_camera.py login    --host 10.0.0.9 --user admin
    python tools/isapi_camera.py info     --host 10.0.0.9 --user admin
    python tools/isapi_camera.py get      --host 10.0.0.9 --user admin \
        --path /ISAPI/Security/capabilities

The password is read from the ``CAM_PASS`` environment variable when ``--pass``
is not supplied, so it never has to appear in shell history.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import socket
import sys
import time
import uuid
import xml.etree.ElementTree as ET

try:
    import requests
    from requests.auth import HTTPDigestAuth
except ImportError:  # discovery needs no third-party dependency
    requests = None
    HTTPDigestAuth = None

SADP_PORT = 37020
SADP_ADDRS = ("239.255.255.250", "255.255.255.255")
SADP_TIMEOUT = 3.0

# Constant baked into the device firmware; AES key material salt.
AES_KEY_SALT = "AaBbCcDd1234!@#$"

AUTH_USERCHECK = 2
AUTH_SESSION_V1 = 3
AUTH_SESSION_V2 = 4


# -- primitives -----------------------------------------------------------


def sha256_hex(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def md5_hex(value: str) -> str:
    return hashlib.md5(value.encode("utf-8")).hexdigest()


def encode_pwd(
    password: str,
    user: str,
    salt: str,
    challenge: str,
    iterations: int,
    irreversible: bool,
) -> str:
    """`utils.encodePwd` — the digest the device expects in SessionLogin.

    SHA256 is applied `iterations` times in total. `irreversible` only changes
    the seed: the salt is mixed in before the challenge, rather than after.
    """
    if irreversible:
        h = sha256_hex(f"{user}{salt}{password}")
        h = sha256_hex(h + challenge)
        start = 2
    else:
        h = sha256_hex(password) + challenge
        start = 1
    for _ in range(start, iterations):
        h = sha256_hex(h)
    return h


def device_aes_key(
    password: str, user: str, salt: str, key_iterate: int, irreversible: bool
) -> str:
    """`webAuth._strToAESKey` — AES key used for `security=N` payloads."""
    k = sha256_hex(f"{user}{salt}{password}") if irreversible else password
    k = sha256_hex(k + AES_KEY_SALT)
    for _ in range(1, key_iterate):
        k = sha256_hex(k)
    return k[:32]


def _text(root: ET.Element, tag: str, default: str = "") -> str:
    node = root.find(f".//{tag}")
    return node.text.strip() if node is not None and node.text else default


def _bool(value: str) -> bool:
    return value.strip().lower() == "true"


# -- transport ------------------------------------------------------------


class Device:
    def __init__(self, host: str, user: str, password: str, port: int = 80) -> None:
        if requests is None:
            raise RuntimeError("requests is required: pip install requests")
        self.host = host
        self.user = user
        self.password = password
        self.base = f"http://{host}:{port}"
        self.session = requests.Session()
        self.session.auth = HTTPDigestAuth(user, password)
        self.session.headers["If-Modified-Since"] = "0"
        self.session_id = ""
        self.key_iterate = 0
        self.secret = ""

    def _url(self, path: str) -> str:
        return f"{self.base}{path}?timeStamp={int(time.time() * 1000)}"

    # -- auth ------------------------------------------------------------

    def capabilities(self, path: str = "/ISAPI/Security/sessionLogin/capabilities"):
        """Negotiate the auth type. Returns (auth_type, cap_element|None, resp)."""
        resp = self.session.get(self._url(path), params={"username": self.user})
        if resp.status_code != 200:
            return AUTH_USERCHECK, None, resp
        root = ET.fromstring(resp.text)
        version = _text(root, "sessionIDVersion")
        return (AUTH_SESSION_V2 if version == "2" else AUTH_SESSION_V1), root, resp

    def error_info(self, resp) -> dict:
        """`webAuth._getErrorInfo` — server-side status and lockout state."""
        try:
            root = ET.fromstring(resp.text)
        except ET.ParseError:
            return {"http": resp.status_code, "body": resp.text[:200]}
        info = {
            "http": resp.status_code,
            "statusCode": _text(root, "statusCode") or _text(root, "statusValue"),
            "subStatusCode": _text(root, "subStatusCode"),
            "statusString": _text(root, "statusString"),
            "lockStatus": _text(root, "lockStatus"),
            "retryLoginTime": _text(root, "retryLoginTime"),
            "unlockTime": _text(root, "unlockTime"),
        }
        return {k: v for k, v in info.items() if v}

    def verify(self) -> tuple[bool, dict]:
        """Single `userCheck` probe. One credential attempt. No session."""
        resp = self.session.get(self._url("/ISAPI/Security/userCheck"))
        return resp.status_code == 200, self.error_info(resp)

    def login(self) -> tuple[bool, dict]:
        """Full session login per the recovered protocol."""
        auth_type, cap, resp = self.capabilities()
        if auth_type == AUTH_USERCHECK:
            ok, info = self.verify()
            return ok, {"authType": AUTH_USERCHECK, **info}

        challenge = _text(cap, "challenge")
        salt = _text(cap, "salt")
        session_id = _text(cap, "sessionID")
        iterations = int(_text(cap, "iterations", "100") or 100)
        irreversible = _bool(_text(cap, "isIrreversible"))
        version = _text(cap, "sessionIDVersion")
        long_term = _text(cap, "isSessionIDValidLongTerm") or "false"

        digest = encode_pwd(
            self.password, self.user, salt, challenge, iterations, irreversible
        )
        body = (
            "<SessionLogin>"
            f"<userName>{self.user}</userName>"
            f"<password>{digest}</password>"
            f"<sessionID>{session_id}</sessionID>"
            f"<isSessionIDValidLongTerm>{long_term}</isSessionIDValidLongTerm>"
            f"<sessionIDVersion>{version}</sessionIDVersion>"
            "</SessionLogin>"
        )
        resp = self.session.post(
            self._url("/ISAPI/Security/sessionLogin"),
            data=body,
            headers={"Content-Type": "application/xml"},
        )
        if resp.status_code != 200:
            return False, {"authType": auth_type, **self.error_info(resp)}
        root = ET.fromstring(resp.text)
        self.session_id = _text(root, "sessionID")
        self.session.cookies.set("WebSession", self.session_id)
        self.cred = {"authType": auth_type, "sessionID": self.session_id}
        return True, {**self.cred, "iterations": iterations, "irreversible": irreversible}

    def security_capabilities(self) -> dict:
        """`/ISAPI/Security/capabilities` — keyIterateNum, salt, isIrreversible."""
        resp = self.session.get(
            self._url("/ISAPI/Security/capabilities"), params={"username": self.user}
        )
        if resp.status_code != 200:
            return {}
        root = ET.fromstring(resp.text)
        self.key_iterate = int(_text(root, "keyIterateNum", "0") or 0)
        salt = _text(root, "salt")
        self.secret = device_aes_key(
            self.password, self.user, salt, self.key_iterate, _bool(_text(root, "isIrreversible"))
        )
        return {
            "keyIterateNum": self.key_iterate,
            "securityVersion": _text(root, "securityVersion"),
            "isIrreversible": _bool(_text(root, "isIrreversible")),
        }

    # -- requests --------------------------------------------------------

    def get(self, path: str):
        return self.session.get(self._url(path))

    def put(self, path: str, body: str):
        return self.session.put(
            self._url(path), data=body, headers={"Content-Type": "application/xml"}
        )


# -- SADP discovery -------------------------------------------------------


def sadp_packet() -> bytes:
    return (
        '<?xml version="1.0" encoding="utf-8"?>'
        f"<Probe><Uuid>{str(uuid.uuid4()).upper()}</Uuid><Types>inquiry</Types></Probe>"
    ).encode("utf-8")


def discover(timeout: float = SADP_TIMEOUT) -> list[dict]:
    """Broadcast the SADP inquiry and collect device replies on UDP/37020."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.settimeout(timeout)
    packet = sadp_packet()
    for addr in SADP_ADDRS:
        try:
            sock.sendto(packet, (addr, SADP_PORT))
        except OSError as exc:
            print(f"  ! could not send to {addr}: {exc}", file=sys.stderr)
    found: dict[str, dict] = {}
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            data, (src, _) = sock.recvfrom(65535)
        except socket.timeout:
            break
        try:
            root = ET.fromstring(data.decode("utf-8", "replace"))
        except ET.ParseError:
            continue
        info = {child.tag: (child.text or "").strip() for child in root}
        info.setdefault("SourceIP", src)
        key = info.get("MAC") or info.get("Uuid") or src
        found[key] = info
    sock.close()
    return list(found.values())


# -- CLI ------------------------------------------------------------------


def _password(args) -> str:
    value = args.password or os.environ.get("CAM_PASS", "")
    if not value:
        print("error: supply --pass or set CAM_PASS", file=sys.stderr)
        raise SystemExit(2)
    return value


def cmd_discover(args) -> int:
    print(f"Broadcasting SADP inquiry on UDP/{SADP_PORT} ({args.timeout:.0f}s)...")
    devices = discover(args.timeout)
    if not devices:
        print("No devices answered. Check you are on the device's LAN/VLAN.")
        return 1
    for dev in devices:
        print(f"\n{dev.get('IPv4Address', dev.get('SourceIP', '?'))}")
        for key, value in sorted(dev.items()):
            if value:
                print(f"    {key:<22} {value}")
    return 0


def cmd_verify(args) -> int:
    dev = Device(args.host, args.user, _password(args), args.port)
    ok, info = dev.verify()
    print("OK - credential accepted" if ok else "REJECTED")
    if info:
        print(f"  {info}")
        if info.get("lockStatus") == "lock":
            print("  ! account is LOCKED - stop and wait for unlockTime")
    return 0 if ok else 1


def cmd_login(args) -> int:
    dev = Device(args.host, args.user, _password(args), args.port)
    ok, info = dev.login()
    print("login ok" if ok else "login failed")
    print(f"  {info}")
    if ok:
        print(f"  security: {dev.security_capabilities()}")
    return 0 if ok else 1


def cmd_get(args) -> int:
    dev = Device(args.host, args.user, _password(args), args.port)
    ok, info = dev.login()
    if not ok:
        print(f"login failed: {info}")
        return 1
    resp = dev.get(args.path)
    print(f"HTTP {resp.status_code}")
    print(resp.text)
    return 0 if resp.status_code == 200 else 1


def cmd_info(args) -> int:
    args.path = "/ISAPI/System/deviceInfo"
    return cmd_get(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def with_target(p):
        p.add_argument("--host", required=True)
        p.add_argument("--user", default="admin")
        p.add_argument("--pass", dest="password", default=None)
        p.add_argument("--port", type=int, default=80)
        return p

    p = sub.add_parser("discover", help="find devices on the LAN via SADP")
    p.add_argument("--timeout", type=float, default=SADP_TIMEOUT)
    p.set_defaults(func=cmd_discover)

    with_target(sub.add_parser("verify", help="one userCheck, no session")).set_defaults(
        func=cmd_verify
    )
    with_target(sub.add_parser("login", help="full session login")).set_defaults(
        func=cmd_login
    )
    with_target(sub.add_parser("info", help="device info (needs valid creds)")).set_defaults(
        func=cmd_info
    )
    get_p = with_target(sub.add_parser("get", help="GET an ISAPI path"))
    get_p.add_argument("--path", required=True)
    get_p.set_defaults(func=cmd_get)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())