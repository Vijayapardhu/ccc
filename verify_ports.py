#!/usr/bin/env python3
"""Verify which ports are genuinely serving versus VPN-tunnel artifacts.

A connect succeeding does not prove a service exists. This reads an actual
application-level response (banner or HTTP status) to confirm reality.
"""

import socket
import ssl

HOST = "117.196.244.183"

# mix of expected-real, expected-closed controls, and a few probes
PORTS = [21, 22, 23, 25, 53, 80, 110, 143, 443, 445, 554, 993, 995,
         3306, 3389, 5900, 8000, 8080, 8443, 31337, 49152, 65000]


def banner(port, timeout=4):
    """Try to elicit a real application response."""
    try:
        s = socket.create_connection((HOST, port), timeout=timeout)
    except OSError as e:
        return f"CONNECT-FAIL {e.__class__.__name__}"
    s.settimeout(timeout)
    try:
        if port == 80:
            s.sendall(b"HEAD / HTTP/1.0\r\nHost: x\r\n\r\n")
        elif port == 443:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            raw = socket.create_connection((HOST, port), timeout=timeout)
            s = ctx.wrap_socket(raw, server_hostname=HOST)
            s.sendall(b"HEAD / HTTP/1.0\r\nHost: x\r\n\r\n")
        else:
            s.sendall(b"\r\n")
        data = s.recv(200)
        s.close()
        if not data:
            return "<connect ok, no banner>"
        return data.split(b"\r\n")[0][:90].decode("utf-8", "replace")
    except OSError as e:
        try:
            s.close()
        except OSError:
            pass
        return f"CONNECT-OK but no banner ({e.__class__.__name__})"


print(f"{'port':<7} {'what the service actually says'}")
print("-" * 72)
for p in PORTS:
    print(f"{p:<7} {banner(p)}")
