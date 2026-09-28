#!/usr/bin/env python3
"""Single authenticated RTSP request. No retry storm - determines if the
device ban is IP-based (blocks all) or failure-count based (a correct login passes)."""

import hashlib
import os
import socket
import sys

HOST = os.environ.get("CAM_HOST", "117.196.244.183")
PORT = int(os.environ.get("CAM_PORT", "554"))
USER = os.environ.get("CAM_USER", "root")
PASS = os.environ.get("CAM_PASS", "1234567890")
PATH = os.environ.get("CAM_PATH", "/Streaming/Channels/101")


def md5hex(v):
    return hashlib.md5(v.encode()).hexdigest()


def rt(method, path, auth=None):
    uri = f"rtsp://{HOST}:{PORT}{path}"
    lines = [f"{method} {uri} RTSP/1.0", "CSeq: 1", "User-Agent: check/1.0"]
    if method == "DESCRIBE":
        lines.append("Accept: application/sdp")
    if auth:
        lines.append(f"Authorization: {auth}")
    raw = ("\r\n".join(lines) + "\r\n\r\n").encode()
    sock = socket.create_connection((HOST, PORT), timeout=12)
    sock.settimeout(12)
    sock.sendall(raw)
    buf = b""
    try:
        while True:
            part = sock.recv(4096)
            if not part:
                break
            buf += part
            if b"\r\n\r\n" in buf:
                break
    except socket.timeout:
        pass
    sock.close()
    return buf.decode("utf-8", "replace")


raw = rt("DESCRIBE", PATH)
first = raw.split("\r\n")[0] if raw.strip() else "<no response / connection dropped>"
print(f"1) Unauthenticated : {first}")

if "401" not in raw:
    print("\nNo 401 challenge. Either banned at IP level, or block has cleared.")
    sys.exit(1)

ch = {}
for line in raw.split("\r\n"):
    if line.lower().startswith("www-authenticate:"):
        for part in line.split(None, 2)[2].split(","):
            if "=" in part:
                k, _, v = part.strip().partition("=")
                ch[k.strip().lower()] = v.strip().strip('"')

uri = f"rtsp://{HOST}:{PORT}{PATH}"
ha1 = md5hex(f"{USER}:{ch['realm']}:{PASS}")
ha2 = md5hex(f"DESCRIBE:{uri}")
digest = md5hex(f"{ha1}:{ch['nonce']}:{ha2}")
auth = (f'Digest username="{USER}", realm="{ch["realm"]}", '
        f'nonce="{ch["nonce"]}", uri="{uri}", response="{digest}"')

raw2 = rt("DESCRIBE", PATH, auth)
first2 = raw2.split("\r\n")[0] if raw2.strip() else "<no response / connection dropped>"
print(f"2) Authenticated   : {first2}")

if "200" in raw2:
    print("\n[+] BAN IS FAILURE-COUNT BASED - credential still works")
    head, _, body = raw2.partition("\r\n\r\n")
    print(head.strip())
    if body.strip():
        print("\n--- SDP ---")
        print(body.strip())
    sys.exit(0)
print("\n[-] Authenticated request also dropped -> IP-level ban, wait it out.")
sys.exit(1)
