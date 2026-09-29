# Architecture

Student presence tracking across 400+ campus cameras.

---

## The problem, stated honestly

400 cameras at 4K. The naive reading of "run face recognition on 400 cameras"
is 400 streams of detection against a 50,000-student database, and it does not
work — not because the models are slow, but because **a 4K frame of 200 people
does not contain enough pixels to identify them**.

At 3840x2160 with 200 people, the median face is 30-60px. Downscale the frame
to a detector's training resolution and that becomes 5-10px, which is below
SCRFD's floor and below what ArcFace can embed. The information is simply not
there. No model recovers it.

So the architecture below is organised around three facts that follow from
that, and almost every design decision traces back to one of them:

1. **Crop, don't resize.** Tiled detection preserves native pixels.
2. **Never trust one frame.** At these face sizes the correct answer and a
   confident wrong answer overlap in score. Commit only on agreement over time.
3. **Not every camera needs biometrics.** This is the recommendation that
   matters most, and it is a *scope* decision, not a technical one — see
   [Tiering](#tiering-the-fleet-the-decision-that-actually-matters).

---

## Topology

Three planes, separated so that failure in one cannot corrupt the others.

```
                    ┌──────────────────────────────────────────┐
                    │  IDENTITY PLANE  (2 replicas, no GPU)    │
                    │  gallery · consent · presence writes     │
                    └───────────────▲──────────────────────────┘
                                    │ Redis Stream (capped, acked)
   ┌────────────────────────────────┴─────────────────────────────────┐
   │  CAPTURE PLANE  (N GPU workers, disposable)                     │
   │  RTSP → NVDEC → tile → SCRFD → quality → ArcFace → ByteTrack    │
   │                                       → temporal verification   │
   │  holds NO gallery state, NO durable state                        │
   └──────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴──────────────────────────┐
                    │  CONTROL PLANE  (Postgres + pgvector)      │
                    │  authoritative gallery · presence events   │
                    │  consent records · append-only audit log    │
                    └──────────────────────────────────────────┘
```

### Why the boundary sits here

A capture worker is **disposable**. Killing one loses a few seconds of in-flight
tracks, which is acceptable because a track is transient evidence anyway. It
never loses a committed sighting — those are already durable in Postgres.

If workers held gallery state or wrote presence events directly, a worker crash
would take identity integrity with it, and the 400-camera horizontal scale-out
would become a distributed-consistency problem. It is not one, because workers
cannot name anybody. They emit observations; only the identity service resolves
them.

### Why Redis Streams and not pub/sub

Streams give a consumer group with acknowledgements and replay. A restarted
identity service reprocesses what it missed instead of losing attendance
records. Pub/sub is fire-and-forget and would silently drop commits during
every deploy.

The stream is capped at 400k entries (~1.5 days at full fleet rate). Redis runs
`maxmemory-policy noeviction` **on purpose**: a silently trimmed stream means
missing attendance with no error anywhere, whereas a refused write surfaces in
`/healthz` as `bus_degraded` and something pages.

---

## The per-frame pipeline

```
4K frame
   │
   ├─▶ motion gate ──── skip tiles with no change since last frame
   │
   ├─▶ tile plan ────── 4x2 = 8 tiles @ 1280px, 25% overlap
   │
   ├─▶ SCRFD 2.5G ───── one batched forward pass over all live tiles
   │                   NMS merge into frame coordinates
   │
   ├─▶ quality gate ─── size → blur → illumination → pose → occlusion
   │                   drops ~70-85% of detections
   │
   ├─▶ alignment ────── Umeyama similarity → 112x112 template
   │
   ├─▶ ArcFace ───────── batched, L2-normalised 512-d
   │
   ├─▶ ByteTrack ─────── one stable id per person per camera
   │
   ├─▶ FAISS search ──── top-5 candidates per face
   │
   └─▶ temporal verify ── commit only on multi-frame agreement
```

### 1. Tiling (`campus/imaging/tiling.py`)

Tiles are 1280px from a 3840x2160 frame at 25% overlap → stride 960 → **8
tiles**, 7.1M pixels of work against 8.3M for the frame itself. The point is
not throughput, it is resolution: a 40px face is still 40px in its tile.

Overlap costs ~2.8x versus a non-overlapping grid. That is worth it, because
without it every face straddling a tile boundary is missed, and a missed face
is an attendance record that silently omits a student. `merge_tile_detections`
reports how many raw detections each survivor absorbed; a survivor absorbing
duplicates is a face on a seam, so **rising `tile_count` telemetry means the
overlap is too small**.

`assert tiles cover every pixel` is a test, not a comment. A gap in the grid is
invisible on any dashboard and looks exactly like an empty corridor.

### 2. Quality gate (`campus/imaging/quality.py`)

Runs before the expensive step. Dropping 70-85% of detections before ArcFace is
a straight ~5x saving on GPU, and it *improves* accuracy because ArcFace was
trained on well-aligned, well-exposed faces and degrades predictably outside
that distribution.

Checks run cheapest-first, because a 12px face fails everything else anyway:

| Check | Threshold | Why |
| --- | --- | --- |
| `min_face_px` | 24 | Below this, embeddings are noise |
| `min_face_ratio` | 0.015 | Rejects false positives on distant clutter |
| blur | 45-900 variance of Laplacian | Low = motion smear, high = noise amplification |
| brightness | 25-235 | Under/over-exposed faces carry no identity |
| contrast | ≥ 12 std | Flat crops (walls, crushed shadow) are useless |
| yaw / pitch / roll | 45° / 35° / 40° | A profile view is unrecoverable at any size |

`min_face_px` is the single most effective latency lever, because the number of
embeddings dominates GPU cost. It is also the one to lower on a corridor and
raise in a library — per-camera `overrides` exist for exactly that, and
`configs/cameras.yaml` demonstrates both directions.

Pose is estimated from landmark geometry, not PnP: fast enough to run on every
detection, needs no camera calibration, and accurate enough to answer the only
question asked of it — *is this too far off-axis to embed usefully?* It is not
accurate enough to reconstruct a head and must not be used for that.

### 3. Alignment (`campus/imaging/align.py`)

Umeyama similarity from the 5 detected landmarks to the ArcFace template.
Skipping this costs real accuracy, and the tests assert it directly: a disc
drawn at the face centre must land at the template's eye-line midpoint, and
must do so *whether the face is 60px or 240px*.

Three guards exist because each one prevents a failure that produces plausible
numbers rather than an error:

- **No landmarks → `None`.** Not a fallback crop. An unaligned crop embeds into
  a region ArcFace never saw, and the resulting matches score like real ones.
- **Degenerate spread → `None`.** All landmarks on one point yields a finite
  but meaningless transform.
- **Mirrored landmarks → `None`.** A mirror is a *congruence*, not a
  similarity, so no valid transform exists and the solver returns a
  plausible-looking wrong one. Caught geometrically, before solving.

### 4. Tracking (`campus/track/bytetrack.py`)

Tracking is what turns a low-quality single-frame problem into a
high-confidence multi-frame one. Without it, a face at the edge of a 4K frame
is recognised once, badly, and maybe not at all.

ByteTrack's contribution over plain IoU tracking is the **second association
pass using low-confidence detections**. In a dense corridor, the reliable
detector output excludes anyone partially occluded — precisely the people about
to be cut off by someone walking past. Associating those anyway keeps the track
alive through the occlusion instead of fragmenting it into pieces too short to
ever commit an identity. So `low_threshold` matters more than
`high_threshold` here.

Constant-velocity prediction rather than a Kalman filter: at 400 cameras x 200
faces x 8fps, several microseconds per track per frame is real, and for faces
with irregular motion the accuracy gain does not pay for it.

`active()` (seen this frame) and `live()` (not yet removed) are deliberately
different. Conflating them fragments every track that passes behind a pillar.

Expiry reports each dead track **exactly once**. This is not bookkeeping: the
worker drains that list to finalise identity evidence, and if expiry only
dropped tracks silently, **no identity would ever be committed and the system
would report an empty campus while running perfectly.**

### 5. Temporal verification (`campus/temporal/verifier.py`)

The most important module in the system, and the one whose design is easiest to
get subtly wrong.

A single frame of a 30x40 face yields a top-1 similarity around 0.55-0.65. So
does a confident wrong match on a stranger. So does the correct student under
different corridor lighting. **The distributions overlap.** A per-frame decision
at any fixed threshold is a coin flip dressed up as a result.

So a candidate is committed only when it holds a plurality of a 12-frame
window **and** clears three independent bars:

| Bar | Default | What it catches |
| --- | --- | --- |
| `min_support` | 7/12 frames | Not enough agreement to be evidence |
| `min_median_score` | 0.38 | Score too low in absolute terms |
| `min_margin` | **0.06** | The evidence does not distinguish two people |

**The margin does the real work.** A wrong identity tends to be wrong
*inconsistently* — different strangers win different frames — so the runner-up
sits close to the leader. A right identity is stable, so the margin opens up
over 10-20 frames even when absolute scores are mediocre. This is why the
verifier must record the full top-k and not just the top-1: with top-1 only,
every track looks uncontested and the margin is a constant.

`min_median_score = 0.38` sits far below the ArcFace sibling band (0.3-0.6 for
the *correct* student against their own enrollment). That gap looks wrong on
paper and is the entire point: at these face sizes the correct answer frequently
scores below what a naive single-frame threshold would demand. The **median
over a window**, not the max, is what makes a low median trustworthy.

**Stickiness matters as much as the thresholds.** Someone walking the length of
a corridor passes through blur and occlusion. Without hysteresis the commitment
flickers, producing duplicate attendance records that contradict each other.
A later disagreement has to beat `overturn_score = 0.62` across 5 consecutive
frames to take the track over, and is evaluated on the *tail* of the window —
a global tally is dominated by however many frames the original subject
accumulated and would keep reporting the old identity straight through an
identity switch.

An overturn is surfaced as its own `IDENTITY_CHANGED` outcome and does **not**
produce a second presence event: rows are keyed on `track_id`, so a second
commit would duplicate the record rather than correct it.

The whole module is per-process and in-memory, and that is deliberate. Track
state is worthless once the process dies, because a track without a camera has
no meaning. Making it durable would buy a false sense of safety.

---

## Gallery search (`campus/index/gallery.py`)

`IndexFlatIP` on L2-normalised vectors. **Exact, not approximate** — at 50k x
512 that is ~2ms on GPU, and exactness matters more than speed: the entire
design rests on the runner-up score being *correct*, and an approximate index
that returns a slightly wrong second candidate corrupts the margin the temporal
verifier depends on. Revisit at millions of vectors, and re-validate margin
semantics when you do.

`NumPyGalleryIndex` is the semantic definition of the interface. If FAISS and
it disagree on the same data, FAISS is wrong. It is also the fallback so the
whole control plane, the tests, and a laptop without faiss installed work
unchanged.

**Postgres holds the authoritative vectors; FAISS is a derived, disposable
index.** A gallery that exists only in a GPU process's memory is unrecoverable
after a node failure, impossible to audit, and makes DPDP erasure
unimplementable. Every GPU node rebuilds from Postgres on start.

Multi-photo enrollment (front / left / right) is collapsed to one centroid
vector per student — a 3x saving on index size and search time at 50k students.
The trade-off is that a centroid averages away per-angle variation; if per-angle
recall proves weak, keep the variants and search all of them.

---

## Tiering the fleet — the decision that actually matters

**Running biometric recognition on all 400 cameras is the wrong answer, and
treating it as an engineering problem will cost roughly 20 GPUs to learn
something a scoping meeting would have said in ten minutes.**

Not every camera needs an identity:

- The residential-block perimeter needs *motion*, not faces.
- A bicycle rack needs *counting*.
- A corridor that nobody uses for attendance needs nothing at all.

| Tier | Count | Work | GPUs | DPDP |
| --- | --- | --- | --- | --- |
| 0 | ~300 | motion / counting, no biometrics | ~0 | clean — no personal data |
| 1 | ~100 | full face recognition | 3-6 L40S | consent-gated, per-zone |

This is a 4-6x cost reduction and it is also the posture that survives a DPDP
review, because most cameras never touch a biometric identifier at all. It is
what the `corridor` zone in `configs/system.yaml` already encodes:
`allowed_purposes: [safety]`, 7-day retention, identity anonymised after 60
minutes. The corridor camera can tell you a corridor was busy. It cannot tell
you who was in it an hour later.

If the university genuinely needs all 400, budget the fleet as below — but know
that the NVDEC session limit binds before the GPU does.

### GPU sizing, 400 cameras, all with biometrics

Derived from the pipeline above. **These are estimates; `campus bench` against
your own footage is the only number that counts.**

```
tier-1:  100 cams x 8 fps x 8 tiles          =  6,400 tile-detections/sec
all-400: 400 cams x 8 fps x 8 tiles          = 25,600 tile-detections/sec
         (+~10% for the busy cameras at 12 fps)

with ~50% motion-gating savings on top          => ~12,800 effective
```

| Stage | Cost per unit | All-400 effective | Note |
| --- | --- | --- | --- |
| SCRFD 2.5G @640² | 2.5 GFLOP | ~32 TFLOPS | ~1200 img/s per L40S measured, not computed |
| ArcFace r50 @112² | 0.6 GFLOP/face | ~29 TFLOPS | face count gated by the quality filter |
| FAISS search | 2.4 TFLOP/s | negligible | only on GPU; CPU is 3 orders short |
| **NVDEC decode** | — | **400 sessions** | **usually the binding constraint** |

Measured, not derived, on a single L40S: roughly **1200 SCRFD/s and 4000
ArcFace/s** at these resolutions with TensorRT FP16 and real batching. That
puts all-400 at **~20 L40S-equivalents**; tiered at 100 cameras, **3-6**.

The decode row is the one that surprises people. 400 concurrent 4K H.265
streams is a session-count limit on the decoder engines, and it is reached
before the tensor cores are anywhere near saturation. **Check the NVDEC session
limit for your chosen GPU before buying anything.**

Per-node target is 32 cameras at 70% planned utilisation, so a canteen
surge does not push a node into latency collapse. One process per GPU: two
processes sharing a device halve the model workspace and defeat batching.

### Measure before you buy

```bash
campus bench footage/canteen-1200.mkv --frames 200
```

`bench` reports per-frame detect/embed latency and
`sustainable_cameras_per_node`. Run it on footage from **each camera
archetype** — canteen, corridor, gate, library. A campus average is a number
that describes no camera you actually have.

---

## Data model (`sql/schema.sql`)

```
students ──< student_embeddings (halfvec 512, HNSW)
consent_records (purpose array, camera scope, partial index on active)
cameras, zones
presence_events   -- monthly range partitions on first_seen
audit_log         -- append-only, trigger rejects UPDATE/DELETE
erasure_log
```

Decisions worth defending:

- **`presence_events` is partitioned monthly on `first_seen`.** At 400 cameras
  x 8fps this grows fast, and a single unpartitioned table makes every
  retention job a full scan. Partitions make retention `DROP TABLE` — instant
  and provably complete, which matters when the retention period is a legal
  commitment.
- **`halfvec(512)`, not `vector`.** 256 bytes instead of 2048. Cosine ranking
  is unchanged at half precision and the float32 master copy is the FAISS
  index, not this row.
- **Erasure redacts rather than deletes presence rows.** `student_id → NULL`,
  `redacted → true`. The obligation is to erase the personal data, not to erase
  the fact that a sighting occurred — and the redacted form is what makes the
  erasure *provable*. The vectors and the consent record are deleted outright.
- **`audit_log` is append-only** via a trigger that raises on `UPDATE` or
  `DELETE`. An audit trail that can be edited is not an audit trail.
- **A partial index on non-revoked consents.** The identity hot path only ever
  looks up active consents, so revoked and expired rows are excluded from the
  index entirely.

---

## DPDP posture (`campus/consent/registry.py`)

Three obligations with direct architectural consequences:

1. **Consent is per *purpose*, not a boolean.** Consent for attendance does not
   cover research. `Purpose` is a **closed enum** — adding one is a code change
   that forces someone to read the consent text, rather than a string that
   quietly defaults to allowed.
2. **Zones are the outer bound.** A corridor camera is a safety camera.
   Attendance collection from it is refused regardless of what the student
   consented to. The zone check runs *before* the consent check.
3. **Erasure is a real deletion.** Vectors gone, consent gone, events redacted,
   counts returned per table so the response is evidence rather than an
   acknowledgement.

**No consent cache, no TTL.** A TTL means a revoked consent authorising
identifications for the TTL duration, which fails the Act in substance. A
revocation takes effect on the next frame, not the next deploy.

Suppressed events are still written, with `student_id` NULL. The camera *saw*
someone and the audit trail should show that — but the row must not be a
queryable record of who. That distinction is what makes the erasure provable.

---

## Failure modes

| Failure | Behaviour | Detection |
| --- | --- | --- |
| Camera goes dark | Backoff reconnect, per-camera health | `health.is_stale(10s)` — a dead camera is invisible to a frame counter |
| One corrupt frame | Dropped, logged | `on_error: drop_frame`. Restarting a 32-stream process for one bad frame is a self-inflicted outage |
| Worker process dies | Loses in-flight tracks only | Commits already durable; replays from stream offset |
| Identity service down | Stream grows, capped | `bus_degraded`; nothing dropped silently |
| Gallery empty | All observations → UNKNOWN | `/readyz` returns **503** — looks identical to "no records" otherwise |
| Consent revoked mid-semester | Next frame | No cache |
| Retention due | `DROP TABLE` partition | Nightly job, idempotent |
| Model missing | Startup failure | `campus validate` exits 2 |

Two of these deserve emphasis because they fail *silently and successfully*:

**An empty gallery** produces a system that runs perfectly, reports no errors,
and matches nobody. `/readyz` refuses to pass with an empty gallery for exactly
this reason.

**Redis `allkeys-lru` under memory pressure** produces missing attendance with
no error anywhere. Hence `noeviction`.

---

## What to build next

In priority order, for the same reason each one is first:

1. **Get real footage and run `campus bench` on it.** Every number above is an
   estimate until it is not. Also the only way to learn your actual face-size
   distribution, which sets `min_face_px`.
2. **Run the enrollment station for a cohort of 50-100 students.** Gallery
   quality is the ceiling on everything downstream, and a bad ID-card photo
   becomes a permanent false-match source across all 400 cameras. Front / left
   / right roughly halves the false-match rate versus a single card photo.
3. **Measure the false-match rate before tuning thresholds.** Run the pipeline
   over footage you have ground truth for, and count wrong attributions. Do not
   tune `min_margin` before you know your error rate — the thresholds are only
   interpretable against a measured baseline.
4. **Cross-camera visit linking.** Deliberately out of scope here. Tracks are
   camera-scoped; linking a person across cameras needs its own design and a
   different error budget.
5. **Camera handover / re-identification.** Same reason.

The one thing not to do: tune thresholds on synthetic data. `tests/` proves the
logic is correct; it proves nothing about accuracy on your campus.
