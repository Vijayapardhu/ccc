# cam

Tooling for a **XiongMai "XMEye"** (EZVIZ-branded) IP camera, built while recovering
access to a unit at `117.196.244.183` and pulling its RTSP stream.

Firmware identified as **V4.0.1 build211217** (web plugin V3.0.7.500).

## What the device looks like

| Surface | Detail |
| --- | --- |
| Web UI | `http://117.196.244.183/doc/page/login.asp` |
| RTSP | `:554`, Digest-MD5, per-request rotating nonce, **no `qop`** |
| ISAPI | `:80`, Digest-MD5 with `qop=auth` |
| Stream paths | `/Streaming/Channels/101`, `/102`, `/201` |
| Web login | `m_iAuthType=3` — AES challenge-response via the WebSDK plugin |

The RTSP and ISAPI layers share a realm (`b462b7e41c0c9b85a726519a`) but **not a
user store** — the recovered RTSP account is rejected by the web/ISAPI backend.

## Scripts

| File | Purpose |
| --- | --- |
| `cam_stream.py` | Main tool. `wait` / `probe` / `snap` / `record` / `play` / `view` |
| `rtsp_recover.py` | RFC 2069 digest client; searches default credentials on RTSP |
| `isapi_login.py` | Tests a credential against the ISAPI endpoints |
| `isapi_search.py` | Searches default credentials against ISAPI |
| `verify_rtsp.py` | Verifies a credential and dumps the SDP |
| `single_check.py` | One-shot auth check; distinguishes IP ban from wrong password |

## Usage

Everything is configured through environment variables, with the discovered
values as defaults:

```powershell
$env:CAM_HOST = "117.196.244.183"
$env:CAM_USER = "root"
$env:CAM_PASS = "1234567890"
```

### Viewing and recording

```powershell
python cam_stream.py probe      # SDP: codec, resolution, framerate
python cam_stream.py snap       # single JPEG -> out/snapshot.jpg
python cam_stream.py record 30  # 30s clip -> out/recording.mp4 (-c copy)
python cam_stream.py play       # live view in ffplay
python cam_stream.py view       # live view in VLC
```

Equivalent ffmpeg, if you prefer to drive it directly:

```powershell
ffplay -rtsp_transport tcp "rtsp://root:PASS@117.196.244.183:554/Streaming/Channels/101"
ffmpeg -rtsp_transport tcp -i "rtsp://root:PASS@117.196.244.183:554/Streaming/Channels/101" -t 30 -c copy out.mp4
```

## Rate limiting — read this first

These units enforce a **client-side attempt cap** (`iMaxQANum`, default 3) and, in
practice, an **IP-level RTSP ban** that trips after burst traffic. The cap is only
a JS constant, but the IP ban is real and server-side.

Symptoms of a ban: TCP connects successfully, then the connection is reset
(`WinError 10054`) before any RTSP response — even for an unauthenticated request.
This is *not* a credential problem. `single_check.py` distinguishes the two, and
`cam_stream.py wait` polls gently until the ban lifts.

Do not run the credential search tools in a loop. One pass is enough; a
successful search still leaves the IP temporarily blocked.

## Credential hygiene

The default credential was found by `rtsp_recover.py` and is committed here as a
constant. It is a **factory default that was never changed**, and the camera is
reachable from the public internet with RTSP exposed.

Rotate it. Also worth doing:

- Change the admin password in the web UI
- Keep RTSP off the public internet, or put it behind a VPN / firewall rule
- The web login uses a separate account from RTSP; changing the RTSP password
  does not change the web one
