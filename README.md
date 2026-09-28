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
| `setup.bat` | One-click bootstrap: install/verify FFmpeg + VLC |
| `start_stream.bat` | One-click live stream in VLC |
| `cam_stream.py` | Main tool. `wait` / `probe` / `snap` / `record` / `play` / `view` |
| `rtsp_recover.py` | RFC 2069 digest client; searches default credentials on RTSP |
| `isapi_login.py` | Tests a credential against the ISAPI endpoints |
| `isapi_search.py` | Searches default credentials against ISAPI |
| `verify_rtsp.py` | Verifies a credential and dumps the SDP |
| `single_check.py` | One-shot auth check; distinguishes IP ban from wrong password |

## Quick start on a new machine

Two double-clicks and you are watching the stream.

1. **`setup.bat`** — verifies Python, installs FFmpeg and VLC via winget,
   writes `camera.env` from the template. Safe to re-run; anything already
   present is skipped.
2. **`start_stream.bat`** — reads `camera.env` and opens the live stream in
   VLC. Falls back to `ffplay` if VLC is missing.

`setup.bat` needs `winget` (present on Windows 10 1809+ / Windows 11) for the
FFmpeg and VLC installs. Everything else is detected, not downloaded.

### Dependencies

| Tool | Why | Installed by |
| --- | --- | --- |
| Python 3.10+ | runs the scripts | pre-existing; verified by setup |
| FFmpeg | `snap`, `record`, `play` | winget `Gyan.FFmpeg` |
| VLC | GUI playback | winget `VideoLAN.VLC` |

**No pip packages are required.** Every script uses only the Python standard
library and shells out to FFmpeg. `requirements.txt` exists but is entirely
comments - it lists optional packages (PyAV, OpenCV, NumPy) that you would only
need if you extend the scripts to decode frames in Python rather than
delegating to FFmpeg.

### Configuration

`setup.bat` copies `camera.env.example` to `camera.env` on first run. Edit
`camera.env` to point at a different camera:

```bat
CAM_HOST=117.196.244.183
CAM_PORT=554
CAM_USER=root
CAM_PASS=1234567890
CAM_PATH=/Streaming/Channels/101
```

`camera.env` is gitignored, since it is per-device. The same keys work as
environment variables for the Python scripts:

```powershell
$env:CAM_HOST = "117.196.244.183"
python cam_stream.py snap
```

## Manual usage

Everything is also configurable through environment variables, with the
discovered values as defaults:

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

A **different machine on a different network is not affected** by an existing
ban, since the block is per source IP. Setting up a second device is therefore
usually the fastest way around a ban on the first.

## Credential hygiene

The default credential was found by `rtsp_recover.py` and is committed here as a
constant. It is a **factory default that was never changed**, and the camera is
reachable from the public internet with RTSP exposed.

Rotate it. Also worth doing:

- Change the admin password in the web UI
- Keep RTSP off the public internet, or put it behind a VPN / firewall rule
- The web login uses a separate account from RTSP; changing the RTSP password
  does not change the web one
