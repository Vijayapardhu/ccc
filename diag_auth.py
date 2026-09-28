#!/usr/bin/env python3
"""Diagnose the RTSP digest handshake: print challenge, computed values, and
try both same-connection and new-connection authentication."""

import hashlib
import os
import socket
import time

HOST = os.environ.get("CAM_HOST", "117.196.244.183")
PORT = int(os.environ.get("CAM_PORT", "554"))
USER = os.environ.get("CAM_USER", "root")
PASS = os.environ.get("CAM_PASS", "1234567890")
PATH = os.environ.get("CAM_PATH", "/Streaming/Channels/101")


def md5hex(v):
    return hashlib.md5(v.encode()).hexdigest()


def read_response(sock, timeout=8):
    sock.settimeout(timeout)
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
    return buf.decode("utf-8", "replace")


def build(method, cseq, auth=None):
    uri = f"rtsp://{HOST}:{PORT}{PATH}"
    lines = [f"{method} {uri} RTSP/1.0", f"CSeq: {cseq}",
             "User-Agent: diag/1.0", "Accept: application/sdp"]
    if auth:
        lines.append(f"Authorization: {auth}")
    return ("\r\n".join(lines) + "\r\n\r\n").encode()


def parse(raw):
    ch = {}
    for line in raw.split("\r\n"):
        if line.lower().startswith("www-authenticate:"):
            for part in line.split(None, 2)[2].split(","):
                if "=" in part:
                    k, _, v = part.strip().partition("=")
                    ch[k.strip().lower()] = v.strip().strip('"')
    return ch


uri = f"rtsp://{HOST}:{PORT}{PATH}"

print(f"target {HOST}:{PORT}{PATH}")
print(f"credential {USER}:{PASS}\n")

# --- Test A: same connection ------------------------------------------
print("=== Test A: challenge + auth on ONE connection ===")
try:
    s = socket.create_connection((HOST, PORT), timeout=10)
    s.sendall(build("DESCRIBE", 1))
    raw = read_response(s)
    print(f"  step1 -> {raw.splitlines()[0] if raw.strip() else '<empty>'}")
    ch = parse(raw)
    print(f"  challenge: realm={ch.get('realm')!r} nonce={ch.get('nonce')!r} "
          f"qop={ch.get('qop')!r} opaque={ch.get('opaque')!r}")

    if "realm" in ch:
        ha1 = md5hex(f"{USER}:{ch['realm']}:{PASS}")
        ha2 = md5hex(f"DESCRIBE:{uri}")
        dg = md5hex(f"{ha1}:{ch['nonce']}:{ha2}")
        auth = (f'Digest username="{USER}", realm="{ch["realm"]}", '
                f'nonce="{ch["nonce"]}", uri="{uri}", response="{dg}"')
        print(f"  HA1={ha1}")
        print(f"  HA2={ha2}")
        print(f"  response={dg}")
        s.sendall(build("DESCRIBE", 2, auth))
        raw2 = read_response(s)
        print(f"  step2 -> {raw2.splitlines()[0] if raw2.strip() else '<empty>'}")
        if "200" in raw2:
            _, _, body = raw2.partition("\r\n\r\n")
            print("\n  *** SAME-CONNECTION AUTH SUCCEEDED ***")
            if body.strip():
                print("  --- SDP ---")
                print("  " + body.strip().replace("\r\n", "\n  "))
    s.close()
except Exception as exc:
    print(f"  error: {type(exc).__name__}: {exc}")

# --- Test B: fresh connection, reuse nonce from A ----------------------
print("\n=== Test B: reuse that nonce on a NEW connection ===")
try:
    s2 = socket.create_connection((HOST, PORT), timeout=10)
    s2.sendall(build("DESCRIBE", 1, auth))
    raw3 = read_response(s2)
    print(f"  -> {raw3.splitlines()[0] if raw3.strip() else '<empty>'}")
    s2.close()
except Exception as exc:
    print(f"  error: {type(exc).__name__}: {exc}")

# --- Test C: fresh challenge then auth, back to back -------------------
print("\n=== Test C: fresh challenge, auth on new conn (current script flow) ===")
try:
    s3 = socket.create_connection((HOST, PORT), timeout=10)
    s3.sendall(build("DESCRIBE", 1))
    r1 = read_response(s3)
    s3.close()
    ch3 = parse(r1)
    print(f"  fresh nonce={ch3.get('nonce')!r} realm={ch3.get('realm')!r}")
    ha1 = md5hex(f"{USER}:{ch3.get('realm','')}:{PASS}")
    ha2 = md5hex(f"DESCRIBE:{uri}")
    dg = md5hex(f"{ha1}:{ch3.get('nonce','')}:{ha2}")
    a = (f'Digest username="{USER}", realm="{ch3.get("realm","")}", '
         f'nonce="{ch3.get("nonce","")}", uri="{uri}", response="{dg}"')
    s4 = socket.create_connection((HOST, PORT), timeout=10)
    s4.sendall(build("DESCRIBE", 1, a))
    r2 = read_response(s4)
    print(f"  -> {r2.splitlines()[0] if r2.strip() else '<empty>'}")
    if "200" in r2:
        print("  *** CROSS-CONNECTION AUTH SUCCEEDED ***")
    s4.close()
except Exception as exc:
    print(f"  error: {type(exc).__name__}: {exc}")
