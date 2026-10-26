# XMEye / XiongMai web API — reverse-engineering notes

Derived from a live campus unit (firmware `V4.0.1build211217`, web UI
last-modified 2021-12-17) by reading its own shipped JavaScript. Not a guess from
vendor docs. Host addresses and credentials are redacted; nothing below depends
on them.

Device fingerprint:

| Field | Value |
| --- | --- |
| Web UI | `http://<host>/doc/page/login.asp` (the `?_<ms>` suffix is a cache-buster) |
| Front-end | AngularJS 1.x + SeaJS module loader |
| `seajs.web_version` | `V4.0.1build211217` |
| `seajs.plugin_version` | `V3.0.7.500` |
| ISAPI XML namespace | `http://www.hikvision.com/ver20/XMLSchema` (XiongMai ISAPI compat layer) |
| RTSP | `554`, `/Streaming/Channels/101` (main), `.../102` (sub) |

Key front-end modules (all under `/doc/script/`):

| Path | Role |
| --- | --- |
| `login.js` | page controller; calls `isapi/webAuth` |
| `isapi/webAuth.js` | auth-type negotiation + login |
| `isapi/websdk.js` | `WebSDK.WSDK_Request` / `WSDK_SetDeviceConfig` |
| `lib/utils.js` | `encodePwd`, `encodeAES`, `decodeAES` |
| `lib/encryption/AES.js`, `crypto.min.js` | CryptoJS + global `SHA256`/`MD5` |

---

## 1. Authentication

The login is **ISAPI**, not the older `/RPC2` JSON RPC. All ISAPI endpoints sit
behind **HTTP Digest** (`WWW-Authenticate: Digest realm=<device-id>, qop="auth",
algorithm=MD5`). Plain Basic returns `401`; the browser SDK relies on `xhr.open`'s
built-in Digest handling.

### 1.1 Negotiate the auth type

```
GET /ISAPI/Security/sessionLogin/capabilities?username=<user>
```

Response (`SessionLoginCap`):

```xml
<sessionID>fd2b9f07...235fc7</sessionID>
<challenge>35233ae6b8d38847093e1093403d6b09</challenge>
<iterations>100</iterations>
<isIrreversible>true</isIrreversible>
<salt>9eaba42e...db4650</salt>
<isSessionIDValidLongTerm opt="true,false">false</isSessionIDValidLongTerm>
<sessionIDVersion>2</sessionIDVersion>
```

`sessionIDVersion == 2` selects auth type **4**, otherwise type **3**. If the
request errors, the SDK falls back to type **2**.

### 1.2 The three auth types

| Type | Method | Endpoint |
| --- | --- | --- |
| 2 | `GET` | `/ISAPI/Security/userCheck?timeStamp=<ms>` — 200 means OK |
| 3 | `POST` | `/ISAPI/Security/sessionLogin?timeStamp=<ms>` (v1 body) |
| 4 | `POST` | `/ISAPI/Security/sessionLogin?timeStamp=<ms>` (v2 body) |

Type 4 body adds two fields:

```xml
<SessionLogin>
  <userName>%s</userName>
  <password>%s</password>
  <sessionID>%s</sessionID>
  <isSessionIDValidLongTerm>false</isSessionIDValidLongTerm>
  <sessionIDVersion>2</sessionIDVersion>
</SessionLogin>
```

Success returns a `sessionID`; the SDK also sets cookie `WebSession=<sessionID>`.

### 1.3 Digest password — `utils.encodePwd`

This is the load-bearing algorithm (`lib/utils.js:227`):

```js
encodePwd(password, {userName, salt, challenge, iIterate}, isIrreversible) {
  let h;
  if (isIrreversible) {
    h = SHA256(userName + salt + password);   // hex, lowercase
    h = SHA256(h + challenge);
    for (let n = 2;  iIterate > n; n++) h = SHA256(h);
  } else {
    h = SHA256(password) + challenge;          // hex string concatenated with challenge
    for (let n = 1;  iIterate > n; n++) h = SHA256(h);
  }
  return h;
}
```

`iIterate` is `iterations` from the capabilities response (100 here).
Both branches apply SHA256 `iIterate` times in total; `isIrreversible` only
changes the seed. `SHA256` returns a lowercase hex string, and each round hashes
that hex string (not raw bytes).

### 1.4 AES key / payload protection

`GET /ISAPI/Security/capabilities?username=<user>` yields `keyIterateNum`,
`isIrreversible`, `salt`, `securityVersion`. Then:

```js
irreversibleKey = isIrreversible ? SHA256(userName + salt + password) : password;
k = SHA256(irreversibleKey + "AaBbCcDd1234!@#$");
for (let n = 1; keyIterateNum > n; n++) k = SHA256(k);
szAESKey = k.substring(0, 32);
```

Commands declared as `security<N>` get `?security=N&iv=<MD5(ms)>&key=<szAESKey>`
appended, and the sensitive body fields are AES-encrypted:

- **AES-128-CBC**, PKCS7
- key = hex-decoded `szAESKey` (32 hex chars)
- IV = hex-decoded per-request `iv` (default `6cd9616beb39d4034fdebe107df9a399`)

### 1.5 Server-side login feedback

`userCheck` / `sessionLogin` failures return:

```xml
<userCheck>
  <statusValue>401</statusValue>
  <statusString>Unauthorized</statusString>
  <lockStatus>unlock</lockStatus>
  <unlockTime>0</unlockTime>
  <retryLoginTime>3</retryLoginTime>
</userCheck>
```

The web UI branches on `lockStatus`, `unlockTime`, `retryLoginTime`. **Failed
logins are counted and the account locks.** Treat `retryLoginTime` as a hard
budget, not a suggestion.

---

## 2. Password recovery (from `pwdReset.asp` / `pwdReset.js`)

The web reset page offers exactly three methods, each gated by
`/ISAPI/Security/extern/capabilities` (`securityExtCap`):

| Method | Request | Gated by |
| --- | --- | --- |
| GUID file | `POST /ISAPI/Security/GUIDFileData?resetpassword=<RSA>` (`application/x-pem-file`) | `isSupportWithGUIDFileData` |
| Security questions | `PUT /ISAPI/Security/questionConfiguration` — XML `<SecurityQuestion>`; answers + new password RSA-encrypted | `isSupportWithSecurityQuestion` |
| Email code | `PUT` `emailCertification` — `{EmailCertification:{securityCode,resetPassword}}` | `isSupportWithSecurityEmail` |

`GET /ISAPI/Security/questionConfiguration` returns `statusCode 6`,
`subStatusCode WANNotSupport` — the firmware classifies WAN clients and refuses
recovery to them.

## 3. Observed on this device

| Probe | Result |
| --- | --- |
| `GET /SDK/activateStatus` | `Activated=true` |
| `GET /ISAPI/Security/extern/capabilities` | `statusCode 4 / notSupport` — no GUID/QA/email reset |
| `GET /ISAPI/Security/questionConfiguration` | `statusCode 6 / WANNotSupport` |
| `GET /ISAPI/Security/userCheck` (Digest, `root:<redacted>`) | `401`, `lockStatus=unlock`, `retryLoginTime=3` |
| `GET /ISAPI/Security/userCheck` (Digest, `admin:<redacted>`) | `401`, `lockStatus=unlock`, `retryLoginTime=4` |
| `GET /ISAPI/Security/sessionLogin/capabilities?username=root` | `SessionLoginCap` (v2, 100 iters, irreversible) |
| RTSP `DESCRIBE` authed (`root:<redacted>`, digest) | `401` (unauth probe gives a normal `401` challenge → not an IP ban) |
| `POST /onvif/device_service` | `404` — no ONVIF |

**Conclusion:** the credential in `camera.env` (`CAM_USER=root`, password
redacted) is invalid for *both* ISAPI and RTSP, and matches the value in the
pre-`38ddce5` git history — so it is stale (rotated), not a transcription error.
Neither default account name (`root`, `admin`) accepts that password, and the
device is `Activated`, so the password was changed. The sibling units were left
untouched, and `rtsp_authgrinder` was not used.

`retryLoginTime` is **per account**, capped at 5: `root` read 3 after two prior
failures, `admin` read 5 then 4. Guessing further will lock an account.

`/Streaming/Channels/101` and the ISAPI XML namespace are **Hikvision** lineage,
not classic XiongMai — which decides the recovery tooling (SADP / iVMS, or a
LAN-side reset).

## 4. Open questions

- Correct username/password for the unit (and whether the sibling units share it).
- Whether the device is Hikvision-branded or an OEM clone — determines whether
  SADP/iVMS reset applies.
- Whether the RTSP breakage is the Concord-style missing SPS/PPS, or an encoder
  setting reachable through ISAPI. Cannot be assessed without valid auth.

## 5. Factory reset and re-provisioning

Remote self-service reset is unavailable, so a factory reset is the remaining
path. Relevant endpoints:

| Action | Request |
| --- | --- |
| Activation state | `GET /SDK/activateStatus` (no auth) |
| Activate (set new password) | `PUT /ISAPI/System/activate` — `<ActivateInfo><password>RSA</password></ActivateInfo>` |
| Activate IPC behind a DVR | `PUT /ISAPI/ContentMgmt/InputProxy/channels/activate` |
| Device info | `GET /ISAPI/System/deviceInfo` (auth) |

Hardware procedure is model-specific: power on, hold the recessed reset button
(10–20 s) until the status LED changes / it reboots. Many Hikvision-lineage units
then come up **inactive**, and the web UI shows the activation dialog
(`activateStatus` → `bNeedAction`), where a new admin password is set.

**Do this on the LAN, not over the internet.** A reset commonly reverts network
settings (Hikvision default `192.168.1.64`, or DHCP), so the existing port-forward
to the public IP can stop resolving and the device becomes unreachable remotely.
Budget on-site access to re-provision before resetting.

### Discovery (SADP)

`Search Active Devices Protocol`, UDP **37020**, sent to `239.255.255.250` and the
broadcast address:

```xml
<?xml version="1.0" encoding="utf-8"?>
<Probe><Uuid>...</Uuid><Types>inquiry</Types></Probe>
```

Each device replies with a `ProbeMatch` containing `IPv4Address`, `MAC`,
`DeviceModel`, `DeviceVersion`, `SerialNumber`, `Activated`, etc. This is how you
find the unit after a reset moved it to a default address.

### Activation encryption

Setting the admin password after a reset encrypts it, via `lib/encryption/encryption.js`:

1. `POST /ISAPI/Security/challenge` with `<PublicKey><key>base64(rsa_pub)</key></PublicKey>`
   (client generates a 1024-bit RSA keypair locally);
2. the device returns `key` = an AES key encrypted to that RSA public key;
3. the client decrypts it and encrypts the password under it (ECB, per
   `encodeAES(..., "ecb")`), sending `base64(block + ciphertext)`.

Deliberately **not** implemented in the tool — the vendor SADP GUI does
activation correctly, and a wrong implementation here would silently produce a
device you cannot re-enter.

## 6. Tooling

`tools/isapi_camera.py` implements the recovered protocol (auth-type negotiation,
`encodePwd`, the session login, `security=N` key derivation) plus SADP discovery:

```
python tools/isapi_camera.py discover                     # find units on the LAN
python tools/isapi_camera.py verify --host H --user admin  # one userCheck, no session
python tools/isapi_camera.py login  --host H --user admin  # full session login
python tools/isapi_camera.py info   --host H --user admin  # /ISAPI/System/deviceInfo
python tools/isapi_camera.py get    --host H --user admin --path /ISAPI/Security/capabilities
```

Password comes from `--pass` or `$CAM_PASS`. `verify` is the safe way to test a
candidate: one request, and it prints `lockStatus` / `retryLoginTime` so you stop
before a lockout. It does not iterate any wordlist.