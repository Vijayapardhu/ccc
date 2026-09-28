#!/usr/bin/env python3
"""Verify recovered RTSP credential and dump the SDP / stream setup."""

import hashlib
import os
import socket
import sys
import time

HOST = os.environ.get("CAM_HOST", "117.196.244.183")
PORT = int(os.environ.get("CAM_PORT", "554"))
USER = os.environ.get("CAM_USER", "root")
PASS = os.environ.get("CAM_PASS", "1234567890")
PATH = os.environ.get("CAM_PATH", "/Streaming/Channels/101")


def md5hex(v):
    return hashlib.md5(v.encode()).hexdigest()


def exchange(method, path, auth=None, extra=None, attempts=3):
    uri = f"rtsp://{HOST}:{PORT}{path}"
    lines = [f"{method} {uri} RTSP/1.0", "CSeq: 1", "User-Agent: verify/1.0"]
    if method == "DESCRIBE":
        lines.append("Accept: application/sdp")
    if auth:
        lines.append(f"Authorization: {auth}")
    for line in extra or []:
        lines.append(line)
    raw = ("\r\n".join(lines) + "\r\n\r\n").encode()

    last_exc = None
    for attempt in range(attempts):
        sock = None
        try:
            sock = socket.create_connection((HOST, PORT), timeout=10)
            sock.settimeout(10)
            sock.sendall(raw)
            buf = b""
            deadline = time.time() + 10
            while time.time() < deadline:
                try:
                    part = sock.recv(4096)
                except socket.timeout:
                    break
                if not part:
                    break
                buf += part
                head, sep, rest = buf.partition(b"\r\n\r\n")
                if not sep:
                    continue
                clen = 0
                for line in head.decode("utf-8", "replace").split("\r\n"):
                    if line.lower().startswith("content-length:"):
                        clen = int(line.split(":", 1)[1].strip())
                if len(rest) >= clen:
                    break
            if buf.strip():
                return buf.decode("utf-8", "replace")
        except (ConnectionResetError, ConnectionAbortedError, socket.timeout, OSError) as exc:
            last_exc = exc
        finally:
            # The reset path is the common case here, so the socket must be
            # released whether or not an exception was raised.
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass
        # A reset from this device means an IP-level ban, which more traffic
        # will not clear. Stop instead of burning the remaining attempts.
        if last_exc is not None:
            print(f"    [{method}] {type(last_exc).__name__} - connection "
                  f"refused/reset, treating as banned. Stopping.")
            break
    return "RTSP/1.0 000 Transport Error"


resp = exchange("DESCRIBE", PATH)
status = resp.split("\r\n")[0]
print(f"[*] Unauthenticated DESCRIBE -> {status}")

challenge = {}
for line in resp.split("\r\n"):
    if line.lower().startswith("www-authenticate:"):
        for part in line.split(" ", 2)[2].split(","):
            if "=" in part:
                k, _, v = part.strip().partition("=")
                challenge[k.strip().lower()] = v.strip().strip('"')

print(f"[*] Challenge: realm={challenge.get('realm')} nonce={challenge.get('nonce')}")

uri = f"rtsp://{HOST}:{PORT}{PATH}"
if "realm" not in challenge or "nonce" not in challenge:
    print("[!] No digest challenge returned - RTSP still soft-blocking this IP.")
    print("    The credential was already confirmed by rtsp_recover.py; retry later.")
    sys.exit(3)

ha1 = md5hex(f"{USER}:{challenge['realm']}:{PASS}")
ha2 = md5hex(f"DESCRIBE:{uri}")
resp_digest = md5hex(f"{ha1}:{challenge['nonce']}:{ha2}")
auth = (
    f'Digest username="{USER}", realm="{challenge["realm"]}", '
    f'nonce="{challenge["nonce"]}", uri="{uri}", response="{resp_digest}"'
)

full = exchange("DESCRIBE", PATH, auth)
print(f"\n[+] Authenticated DESCRIBE -> {full.splitlines()[0]}")

head, _, body = full.partition("\r\n\r\n")
print("\n--- Response headers ---")
print(head.strip())
if body.strip():
    print("\n--- SDP ---")
    print(body.strip())

print("\n--- SETUP/SESSION probe ---")
setup = exchange("SETUP", PATH, auth,
                 extra=["Transport: RTP/AVP/TCP;unicast;interleaved=0-1"])
print(setup.strip()[:600])
