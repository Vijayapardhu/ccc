# Architecture

Student presence tracking across 400+ campus cameras.

---

## The problem, stated honestly

400 cameras. The naive reading of "run face recognition on 400 cameras" is 400
streams of detection against a 50,000-student database, and it does not work —
not because the models are slow, but because **at the real resolutions, there
is not enough information in the pixels to identify people at range**.

The campus fleet is 1080p and 720p. Face size scales with resolution, and face
size is the binding constraint on everything:

```
face_px  ~=  250 / distance_metres    at 1080p
face_px  ~=  167 / distance_metres    at 720p
```

(~75° HFOV, 0.20m face height. A narrower lens extends the range
proportionally.)

Measured on the real 391-photo enrollment set, against a real 358-student
gallery — same photo downscaled, then back through the whole chain
(detect → gate → align → embed → search):

| Actual face px | 54 | 36 | 27 | 20 | <20 |
| --- | --- | --- | --- | --- | --- |
| Top-1 accuracy | 100% | 100% | 80% | 33% | 0% — nothing survives detection |

| Distance | 1080p | 720p | Verdict |
| --- | --- | --- | --- |
| 4m | 62px | 42px | reliable |
| 6m | 42px | 28px | reliable / usable |
| 8m | 31px | 21px | usable / marginal |
| 10m | 25px | 17px | marginal / unreliable |
| 12m | 21px | 14px | unreliable |
| 15m | 17px | 11px | unreliable |

**A 720p camera cannot cover what a 1080p one does**, and with
`min_face_px = 24` the useful range is roughly **10m at 1080p and 7m at 720p**.
Below ~20px no threshold rescues the match rate — the camera has to move
closer or take a longer lens. That is a placement decision, not a tuning
decision, and it belongs in the camera inventory review.

**These are upper bounds.** The query and the gallery entry came from the same
frontal studio photo, so only resolution and JPEG varied; real sightings differ
in pose, expression and illumination, and the real gallery is 50,000 students
rather than 358. Treat the table as the best case and **measure on real
footage before quoting any accuracy to the university.**

Three consequences shape the design:

1. **Crop, don't resize.** Tiled detection preserves native pixels. This matters
   more at 1080p/720p than at 4K, because a 1280px tile on a 1080p frame is a
   0.5x downscale — precisely the lossy path the whole design exists to avoid.
2. **Never trust one frame.** At these face sizes the correct answer and a
   confident wrong answer overlap in score. Commit only on agreement over time.
   And since per-frame compute is now cheap, **spend the headroom on frame
   rate** — 12–15fps instead of 8 — because frames are what substitute for
   pixels.
3. **Not every camera needs biometrics.** See
   [Tiering](#tiering-the-fleet-the-decision-that-actually-matters). This is
   now *more* true, not less: the 720p cameras at the edge of campus cannot
   identify anyone usefully, and pretending otherwise buys a dashboard that
   reports confident nonsense.

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
1080p frame  (1920x1080; 720p is 6 tiles)
   │
   ├─▶ motion gate ──── skip tiles with no change since last frame
   │
   ├─▶ tile plan ────── 4x2 = 8 tiles @ 640px, 25% overlap, 1:1 scale
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

**`tile_size` must equal `input_size` (640) at 1080p and below**, so tiles reach
the detector at 1:1 and a face is exactly as many pixels to the detector as it
has in the frame.

| Source | tile_size | Tiles | Scale | Detector px/frame | vs frame |
| --- | --- | --- | --- | --- | --- |
| 1080p | 640 | 8 | **1.00** | 3.28M | 1.58x |
| 1080p | 1280 | 2 | 0.50 | 0.82M | 0.40x |
| 720p | 640 | 6 | **1.00** | 2.46M | 2.67x |
| 4K | 1280 | 8 | 0.50 | 3.28M | 0.40x |
| 4K | 640 | 40 | 1.00 | 16.4M | 1.98x |

Two things fall out of this table:

**The 4K-era default of 1280 is actively harmful here.** On a 1080p frame it
runs at 0.5x, a 20px face arrives as 10px, and the camera reports an empty
corridor rather than raising an error. The defaults are now 640.

**Tiling at native scale makes compute nearly resolution-independent.** 1080p
and 4K both cost ~3.3M detector pixels per frame; what differs is *decode*,
not inference. So the GPU story barely moves, while the NVDEC constraint —
which was the binding one — eases substantially at 1080p.

Note the overlap cost is *proportionally* worse on small frames: 1.58x at
1080p, 2.67x at 720p, because a 640px tile is a much larger fraction of those
frames. Drop `tile_overlap` to 0.125 before buying hardware for a 720p-heavy
fleet.

`merge_tile_detections` reports how many raw detections each survivor absorbed;
a survivor absorbing duplicates is a face on a seam, so **rising `tile_count`
telemetry means the overlap is too small**.

`assert tiles cover every pixel` is a test, not a comment, and it now runs for
1080p and 720p as well. A gap in the grid is invisible on any dashboard and
looks exactly like an empty corridor.

### 2. Quality gate (`campus/imaging/quality.py`)

Runs before the expensive step. Dropping 70-85% of detections before ArcFace is
a straight ~5x saving on GPU, and it *improves* accuracy because ArcFace was
trained on well-aligned, well-exposed faces and degrades predictably outside
that distribution.

Checks run cheapest-first, because a 12px face fails everything else anyway:

| Check | Threshold | Why |
| --- | --- | --- |
| `min_face_px` | 20 | Below this, embeddings are noise. Sets the ~12.5m (1080p) / ~8.3m (720p) range limit |
| `min_face_ratio` | 0.015 | Rejects false positives on distant clutter |
| blur | 45-900 variance of Laplacian | Low = motion smear, high = noise amplification |
| brightness | 25-235 | Under/over-exposed faces carry no identity |
| contrast | ≥ 12 std | Flat crops (walls, crushed shadow) are useless |
| yaw / pitch / roll | 45° / 35° / 40° | A profile view is unrecoverable at any size |

`min_face_px` is the single most effective latency lever, because the number of
embeddings dominates GPU cost. It is also the setting that quietly caps your
coverage, and it is **per-camera**. `configs/cameras.yaml` sets it from 15
(a long 1080p corridor, accepting a short range) to 28 (a quiet library
reading room, where faces are close and sharp and a strict gate costs nothing).

Getting this wrong is quiet in a specific way: too high and the far end of a
corridor reports *nobody*, which looks identical to an empty corridor. It will
not raise an error.

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
high-confidence multi-frame one. Without it, a face at the far end of a corridor
is recognised once, badly, and maybe not at all.

ByteTrack's contribution over plain IoU tracking is the **second association
pass using low-confidence detections**. In a dense corridor, the reliable
detector output excludes anyone partially occluded — precisely the people about
to be cut off by someone walking past. Associating those anyway keeps the track
alive through the occlusion instead of fragmenting it into pieces too short to
ever commit an identity. So `low_threshold` matters more than
`high_threshold` here.

Constant-velocity prediction rather than a Kalman filter: at 400 cameras x 200
faces x 12fps, several microseconds per track per frame is real, and for faces
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

### GPU sizing, 400 cameras

Derived from the pipeline above, at the real resolutions. **These are
estimates; `campus bench` against your own footage is the only number that
counts.**

```
tier-1, mixed fleet 100 cams:  60 x 1080p x 8 tiles x 12fps =  5,760 det/s
                               40 x  720p x 6 tiles x 12fps =  2,880 det/s
                                                    total  =  8,640 det/s
                                                        x ~50% motion gate
                                                            =  ~4,300 det/s

all-400, mixed fleet:          250 x 1080p + 150 x 720p x 12fps
                                                     ≈ 40,000 det/s
                                                        x ~50% motion gate
                                                            = ~20,000 det/s
```

| Stage | Cost per unit | Tier-1 (100) | All-400 |
| --- | --- | --- | --- |
| SCRFD 2.5G @640² | 2.5 GFLOP | ~11 TFLOPS | ~50 TFLOPS |
| ArcFace r50 @112² | 0.6 GFLOP/face | ~10 TFLOPS | ~45 TFLOPS |
| FAISS search | — | negligible | negligible (GPU only) |
| **NVDEC decode** | — | **100 sessions** | **400 sessions** |

Measured, not derived, on a single L40S: roughly **1200 SCRFD/s and 4000
ArcFace/s** with TensorRT FP16 and real batching. That puts tier-1 at
**~3 L40S-equivalents** and all-400 at **~15–20**.

**What 1080p actually bought you.** Inference cost is essentially unchanged —
tiling at native scale means the detector sees the same pixel budget at any
resolution. What improved is the constraint that was *actually* binding:
1080p/720p H.265 decode is far cheaper in NVDEC sessions than 4K, and session
availability is what usually gates a fleet before the GPU does. Verify the
session limit for your chosen GPU, but this moves in your favour.

**What it did not buy you: accuracy.** Face pixels halved. That is why the
recommendation to spend the headroom on frame rate matters — 12–15fps instead
of 8 gives the temporal verifier more frames per person crossing the zone, and
frames are the cheapest substitute for pixels. It partly, not fully, offsets
the loss.

### Measure before you buy

```bash
campus bench footage/canteen-1200.mkv --frames 200
```

`bench` reports per-frame detect/embed latency and
`sustainable_cameras_per_node`. Run it on footage from **each camera
archetype** — canteen, corridor, gate, library — and on both resolutions. A
campus average describes no camera you actually have, and a 720p average
describes neither of your 1080p ones.

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
  x 12fps this grows fast, and a single unpartitioned table makes every
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

1. **Audit the camera inventory against the face-size table.** For every camera,
   measure the depth of the area it is meant to cover and check it against
   `face_px ~= 250/d` (1080p) or `167/d` (720p). Any 720p camera covering more
   than ~8m, or 1080p beyond ~12m, cannot identify anybody usefully and should
   be tiered down to motion-only or re-aimed. This is a placement review, not
   a config change, and it is the highest-value hour in the whole project —
   it is the difference between a camera that works and one that confidently
   reports nobody.
2. **Get real footage and run `campus bench` on it**, per camera archetype and
   per resolution. Every number above is an estimate until it is not. Also the
   only way to learn your actual face-size distribution, which sets
   `min_face_px` per camera.
3. **Run the enrollment station for a cohort of 50-100 students.** Gallery
   quality is the ceiling on everything downstream, and a bad ID-card photo
   becomes a permanent false-match source across all 400 cameras. Front / left
   / right roughly halves the false-match rate versus a single card photo.
4. **Measure the false-match rate before tuning thresholds.** Run the pipeline
   over footage you have ground truth for, and count wrong attributions. Do not
   tune `min_margin` before you know your error rate — the thresholds are only
   interpretable against a measured baseline.
5. **Cross-camera visit linking.** Deliberately out of scope here. Tracks are
   camera-scoped; linking a person across cameras needs its own design and a
   different error budget.
6. **Camera handover / re-identification.** Same reason.

Two things not to do. Don't tune thresholds on synthetic data — `tests/` proves
the logic is correct and proves nothing about accuracy on your campus. And
don't compensate for short faces by lowering `min_face_px` further: a 14px face
embeds into a vector that will produce confident wrong answers, and the
temporal verifier will then commit that wrong answer with *more* confidence
because it is consistent across frames. Consistency is not correctness.
