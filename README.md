# campus

Student presence tracking across 400+ campus CCTV cameras.

Dense-crowd face detection, quality gating, ArcFace embedding, FAISS identity
search, and multi-frame temporal verification — with DPDP 2023 consent,
purpose limitation, and erasure built into the data path rather than bolted on.

**Read [ARCHITECTURE.md](ARCHITECTURE.md) first.** It contains the sizing math,
the tiering recommendation, and the failure modes. The short version: the design
is organised around one fact — **face size is the binding constraint**, and at
1080p/720p it falls below what ArcFace can use beyond ~12m / ~8m. So the system
crops instead of resizing, never trusts a single frame, spends its compute
headroom on frame rate rather than on resolution, and does not run biometrics on
cameras that do not need them.

---

## Status

| Area | State |
| --- | --- |
| Tiling, quality gate, alignment | Complete, tested |
| Temporal verification | Complete, tested |
| Gallery index (FAISS + NumPy) | Complete, tested |
| ByteTrack | Complete, tested |
| DPDP consent / purpose / erasure | Complete, tested |
| Worker pipeline, bus, API, CLI | Complete |
| SCRFD / ArcFace wrappers | Written, **untested — needs model weights and a GPU** |
| Postgres schema | Written, **unapplied** |
| Cross-camera visit linking | Not started (deliberately out of scope) |

151 tests pass on CPU. The inference wrappers are the honest gap: they encode
real SCRFD DFL decoding and ArcFace preprocessing, but nothing has run them.
Validate them on real footage before trusting them.

---

## Install

```powershell
python -m pip install -e ".[dev,services,inference,index]"
```

Model weights (~500MB, not vendored):

```bash
bash deploy/fetch-models.sh
```

---

## Quick start

```powershell
# 1. Check the config before touching any hardware. Exits 2 on problems.
campus validate --config configs/system.yaml --cameras configs/cameras.yaml

# 2. Dev stack: Postgres + Redis + one CPU worker + identity + API
docker compose -f deploy/docker-compose.yml up

# 3. Size the pipeline on your own footage. This is the only number that counts.
campus bench footage/canteen.mkv --frames 200
```

GPU production profile:

```bash
docker compose -f deploy/docker-compose.gpu.yml up -d
```

### Commands

| Command | Purpose |
| --- | --- |
| `campus validate` | Config and model check, exit 2 on problems |
| `campus bench VIDEO` | Per-frame latency and sustainable cameras/node |
| `campus worker` | Run a capture worker over a slice of cameras |
| `campus identity` | Run the identity service |
| `campus api` | Serve the HTTP API |
| `campus enroll STU:DIR ...` | Enroll students from photos |
| `campus index` | Rebuild and report the gallery index |

`--config` and `--cameras` work before or after the subcommand.

---

## The pipeline

```
1080p frame
   -> motion gate
   -> tile plan (8 tiles @ 640px, 25% overlap, 1:1)  crop, never resize
   -> SCRFD 2.5G, one batched pass over live tiles
   -> NMS merge into frame coordinates
   -> quality gate (size/blur/pose/illumination)  drops ~70-85%
   -> Umeyama alignment to 112x112
   -> ArcFace, batched, L2-normalised 512-d
   -> ByteTrack, one id per person per camera
   -> FAISS top-5
   -> temporal verification
   -> commit
```

---

## Layout

```
src/campus/
  types.py            domain types; the capture/identity plane boundary
  config.py           pydantic config, validated once at startup
  cli.py              entry point
  imaging/            tiling, quality gate, alignment
  temporal/           verifier  <- the important one
  index/              FAISS + NumPy gallery, centroiding
  track/              ByteTrack
  models/             SCRFD, ArcFace ONNX wrappers
  capture/            RTSP ingest, NVDEC, health
  worker/             capture worker + Redis bus
  identity/           identity service
  consent/            DPDP consent, purpose limitation, erasure
  enroll/             enrollment service
  store/              Postgres
  api/                FastAPI
sql/schema.sql        schema, partitions, append-only audit trigger
configs/              system.yaml, cameras.yaml
deploy/               Dockerfiles, compose, fetch-models.sh
tests/                151 tests, CPU-only, no GPU or weights required
```

---

## Three things to know before changing anything

**1. `temporal/verifier.py` is load-bearing.** The margin over the runner-up is
what makes a multi-frame commitment meaningful, and it is computed from rank 2.
If the verifier ever records only the top-1 candidate, every track looks
uncontested, the margin becomes a constant, and the system degrades to
coin flips *while every test still passes*. `test_temporal.py` guards this.

**2. Alignment has three guards that return `None` rather than a fallback crop.**
Unaligned, degenerate, and mirrored landmarks all produce embeddings that look
valid and match nobody. Never substitute a fallback — it converts a loud
failure into a silent one.

**3. Postgres is the source of truth; FAISS is disposable.** A gallery that
lives only in GPU memory is unrecoverable after a node failure, impossible to
audit, and makes DPDP erasure unimplementable.

---

## Testing

```bash
python -m pytest -q          # 151 tests, no GPU, no weights, ~4s
```

Synthetic throughout. `campus bench` on real footage is what validates
*accuracy*; the suite validates that the logic is *correct*. The two are not
interchangeable, and tuning thresholds against synthetic data will mislead you.

---

## Security

- Camera passwords live in environment variables, never in `cameras.yaml`
  (the config is committed and spans departments).
- `CAMPUS_API_KEY` is required; `campus api` refuses to start without it unless
  explicitly passed `--insecure`.
- Containers run as a non-root uid.
- Redis is `noeviction` so backpressure is visible rather than silent.
- `audit_log` rejects `UPDATE` and `DELETE` at the database level.
