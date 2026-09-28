#!/usr/bin/env python3
"""
Camera streaming helper for the XiongMai/XMEye (EZVIZ) unit at 117.196.244.183.

Modes:
    wait      Poll gently until the RTSP ban lifts, then report.
    probe     DESCRIBE and print the SDP (codec, resolution, framerate).
    snap      Save a single JPEG frame.
    record N  Record N seconds of video to out/recording.mp4 (no re-encode).
    play      Open the live stream in ffplay.
    view      Open the stream in VLC.

The device currently enforces an IP-level RTSP ban triggered by burst traffic,
so every mode here is deliberately low-rate: one connection at a time, and
`wait` polls slowly rather than hammering the socket.
"""

import hashlib
import os
import socket
import subprocess
import sys
import time

HOST = os.environ.get("CAM_HOST", "117.196.244.183")
PORT = int(os.environ.get("CAM_PORT", "554"))
USER = os.environ.get("CAM_USER", "root")
PASS = os.environ.get("CAM_PASS", "1234567890")
PATH = os.environ.get("CAM_PATH", "/Streaming/Channels/101")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
FFMPEG = "ffmpeg"
FFPLAY = "ffplay"
VLC = r"C:\Program Files\VideoLAN\VLC\vlc.exe"

os.makedirs(OUT, exist_ok=True)


def md5hex(v):
    return hashlib.md5(v.encode()).hexdigest()


def rtsp_request(method, path, auth=None, timeout=12):
    uri = f"rtsp://{HOST}:{PORT}{path}"
    lines = [f"{method} {uri} RTSP/1.0", "CSeq: 1", "User-Agent: cam-helper/1.0"]
    if method == "DESCRIBE":
        lines.append("Accept: application/sdp")
    if auth:
        lines.append(f"Authorization: {auth}")
    raw = ("\r\n".join(lines) + "\r\n\r\n").encode()

    sock = socket.create_connection((HOST, PORT), timeout=timeout)
    sock.settimeout(timeout)
    try:
        sock.sendall(raw)
        buf = b""
        deadline = time.time() + timeout
        while time.time() < deadline:
            part = sock.recv(4096)
            if not part:
                break
            buf += part
            if b"\r\n\r\n" in buf:
                head, _, rest = buf.partition(b"\r\n\r\n")
                clen = 0
                for line in head.decode("utf-8", "replace").split("\r\n"):
                    if line.lower().startswith("content-length:"):
                        clen = int(line.split(":", 1)[1].strip())
                if len(rest) >= clen:
                    break
    finally:
        sock.close()
    return buf.decode("utf-8", "replace")


def describe():
    """Return (status, sdp) or ('banned', '') if the IP ban is active."""
    try:
        raw = rtsp_request("DESCRIBE", PATH)
    except (ConnectionResetError, ConnectionAbortedError, socket.timeout, OSError):
        return "banned", ""
    if not raw.strip():
        return "banned", ""
    if "401" not in raw:
        return raw.split("\r\n")[0], ""
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
    raw2 = rtsp_request("DESCRIBE", PATH, auth)
    _, _, body = raw2.partition("\r\n\r\n")
    return raw2.split("\r\n")[0], body


def cmd_wait(interval=180, max_minutes=240):
    print(f"[*] Polling {HOST}:{PORT} every {interval}s, up to {max_minutes} min.")
    print("[*] This is intentionally slow to avoid extending the ban.\n")
    deadline = time.time() + max_minutes * 60
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        stamp = time.strftime("%H:%M:%S")
        status, sdp = describe()
        if sdp.strip() or ("200" in status and "banned" not in status):
            print(f"[{stamp}] attempt {attempt}: {status} - UNBANNED, credential works")
            if sdp.strip():
                print(sdp.strip())
            return 0
        print(f"[{stamp}] attempt {attempt}: still banned")
        time.sleep(interval)
    print("[-] Timed out, ban still active. Try again later.")
    return 1


def cmd_probe():
    status, sdp = describe()
    if sdp.strip():
        print(f"[+] {status}\n")
        print(sdp.strip())
        return 0
    if status == "banned":
        print("[-] RTSP is IP-banning this host. Wait, then retry.")
        return 1
    print(f"[-] unexpected: {status}")
    return 1


def cmd_snap():
    dest = os.path.join(OUT, "snapshot.jpg")
    url = f"rtsp://{USER}:{PASS}@{HOST}:{PORT}{PATH}"
    proc = subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-rtsp_transport", "tcp",
         "-i", url, "-frames:v", "1", "-q:v", "2", "-y", dest],
        capture_output=True, text=True, timeout=60)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"[+] Snapshot saved: {dest} ({os.path.getsize(dest)} bytes)")
        return 0
    print(f"[-] Snapshot failed: {proc.stderr.strip()[:300]}")
    return 1


def cmd_record(seconds=10):
    dest = os.path.join(OUT, "recording.mp4")
    url = f"rtsp://{USER}:{PASS}@{HOST}:{PORT}{PATH}"
    print(f"[*] Recording {seconds}s -> {dest}")
    proc = None
    try:
        proc = subprocess.run(
            [FFMPEG, "-hide_banner", "-loglevel", "error", "-rtsp_transport", "tcp",
             "-i", url, "-t", str(seconds), "-c", "copy", "-y", dest],
            capture_output=True, text=True, timeout=int(seconds) + 45)
    except subprocess.TimeoutExpired:
        # proc is never assigned when the timeout fires; treat it as a
        # transport failure rather than dereferencing it below.
        print(f"[-] ffmpeg timed out after {seconds + 45}s. The camera likely "
              f"is not delivering the stream.")
        return 1
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        size = os.path.getsize(dest)
        print(f"[+] Saved {dest} ({size/1024:.0f} KB)")
        if size < 10000:
            print("[!] File is tiny - stream may not have delivered data.")
        return 0
    detail = proc.stderr.strip()[:300] if proc and proc.stderr else "no error output"
    print(f"[-] Record failed: {detail}")
    return 1


def cmd_play():
    url = f"rtsp://{USER}:{PASS}@{HOST}:{PORT}{PATH}"
    print(f"[*] Launching ffplay on {url}")
    subprocess.run([FFPLAY, "-rtsp_transport", "tcp", url])
    return 0


def cmd_view():
    url = f"rtsp://{USER}:{PASS}@{HOST}:{PORT}{PATH}"
    print(f"[*] Launching VLC on {url}")
    subprocess.run([VLC, "--rtsp-tcp", url])
    return 0


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "probe"
    if mode == "wait":
        interval = int(sys.argv[2]) if len(sys.argv) > 2 else 180
        return cmd_wait(interval)
    if mode == "probe":
        return cmd_probe()
    if mode == "snap":
        return cmd_snap()
    if mode == "record":
        secs = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        return cmd_record(secs)
    if mode == "play":
        return cmd_play()
    if mode == "view":
        return cmd_view()
    print(__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(main())
