#!/usr/bin/env python3
"""Concurrent TCP port scan of the camera.

Port 80 and 554 were the only ports examined so far. Other camera services
(ONVIF, alternate RTSP ports, management daemons) may expose a different
credential store that is not subject to the RTSP global lockout, so
discovering them is worth doing before concluding the device is closed off.
"""

import socket
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

HOST = "117.196.244.183"
PORTS = list(range(1, 10001)) + [
    20000, 34567, 37777, 50000, 50001, 55400, 9000, 8000, 8888, 8899, 10000
]
PORTS = sorted(set(PORTS))
TIMEOUT = 0.35
WORKERS = 700

known = {80: "web/UI (already known)", 554: "RTSP (already known, locked out)"}


def probe(port):
    try:
        with socket.create_connection((HOST, port), timeout=TIMEOUT):
            return port
    except OSError:
        return None


def main():
    print(f"[*] scanning {HOST} ports 1-10000 + high range, {WORKERS} workers")
    open_ports = []
    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(probe, p): p for p in PORTS}
        for fut in as_completed(futures):
            done += 1
            port = fut.result()
            if port:
                open_ports.append(port)
            if done % 2000 == 0:
                print(f"    ...{done}/{len(PORTS)} ports checked")

    open_ports.sort()
    print(f"\n[*] {len(open_ports)} open port(s):\n")
    for p in open_ports:
        tag = known.get(p, "")
        try:
            with socket.create_connection((HOST, p), timeout=1.5) as s:
                s.settimeout(1.5)
                try:
                    s.sendall(b"\r\n")
                    data = s.recv(256)
                except OSError:
                    data = b""
                banner = data.split(b"\r\n")[0][:80] if data else b""
        except OSError:
            banner = b""
        extra = f"  {tag}" if tag else ""
        if banner:
            extra += f"  banner={banner!r}"
        print(f"  {p:<6}{extra}")

    return open_ports


if __name__ == "__main__":
    main()
