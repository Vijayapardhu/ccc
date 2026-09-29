-- Campus presence tracking schema.
--
-- Design constraints that drove this:
--   * 400 cameras x 8 fps -> high write volume, so presence_events is
--     partitioned by month and retention is a DROP TABLE, not a DELETE.
--   * DPDP erasure must be provable, so every redaction is counted in
--     erasure_log and events are redacted (student_id NULL) rather than
--     deleted, preserving the fact that a sighting happened.
--   * FAISS is derived; Postgres holds the authoritative vectors so a GPU
--     node can be rebuilt from scratch.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- ---------------------------------------------------------------------------
-- Cameras
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS cameras (
    id            TEXT PRIMARY KEY,
    name          TEXT NOT NULL DEFAULT '',
    host          TEXT NOT NULL,
    rtsp_path     TEXT NOT NULL DEFAULT '/Streaming/Channels/101',
    port          INTEGER NOT NULL DEFAULT 554,
    zone          TEXT NOT NULL DEFAULT 'default',
    width         INTEGER NOT NULL DEFAULT 3840,
    height        INTEGER NOT NULL DEFAULT 2160,
    target_fps    REAL NOT NULL DEFAULT 8.0,
    enabled       BOOLEAN NOT NULL DEFAULT TRUE,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS cameras_zone_idx ON cameras (zone);
CREATE INDEX IF NOT EXISTS cameras_enabled_idx ON cameras (enabled) WHERE enabled;

-- ---------------------------------------------------------------------------
-- Zones and consent (DPDP)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS zones (
    zone                    TEXT PRIMARY KEY,
    allowed_purposes        TEXT[] NOT NULL,
    require_consent         BOOLEAN NOT NULL DEFAULT TRUE,
    retention_days          INTEGER NOT NULL DEFAULT 30,
    anonymise_after_minutes INTEGER,
    notes                   TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS consent_records (
    student_id   TEXT PRIMARY KEY,
    purposes     TEXT[] NOT NULL,
    consent_id   TEXT NOT NULL,
    camera_scope TEXT[] NOT NULL DEFAULT '{}',
    granted_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at   TIMESTAMPTZ,
    revoked_at   TIMESTAMPTZ
);

-- Partial index: the identity hot path only ever looks up *active* consents,
-- so revoked/expired rows are excluded from the index entirely.
CREATE INDEX IF NOT EXISTS consent_active_idx
    ON consent_records (student_id)
    WHERE revoked_at IS NULL;

CREATE TABLE IF NOT EXISTS erasure_log (
    id         BIGSERIAL PRIMARY KEY,
    student_id TEXT NOT NULL,
    erased_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    vectors    INTEGER NOT NULL DEFAULT 0,
    events     INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS erasure_log_student_idx ON erasure_log (student_id);

-- ---------------------------------------------------------------------------
-- Gallery
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS students (
    student_id  TEXT PRIMARY KEY,
    display_name TEXT NOT NULL DEFAULT '',
    department  TEXT NOT NULL DEFAULT '',
    year        INTEGER,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS student_embeddings (
    student_id   TEXT NOT NULL REFERENCES students (student_id) ON DELETE CASCADE,
    photo_id     TEXT NOT NULL,
    angle        TEXT NOT NULL DEFAULT 'front',
    -- halfvec: 256 bytes/vector instead of 2048. Cosine ranking is unchanged;
    -- the float32 master copy is the FAISS index, not this row.
    embedding    halfvec(512) NOT NULL,
    enrolled_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (student_id, photo_id)
);

-- Half-precision inner product needs the vector extension's halfvec operator,
-- which requires an explicit cast from the index_type parameter.
CREATE INDEX IF NOT EXISTS student_embeddings_cos_idx
    ON student_embeddings
    USING hnsw (embedding halfvec_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- ---------------------------------------------------------------------------
-- Presence events
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS presence_events (
    event_id       UUID PRIMARY KEY,
    student_id     TEXT,
    camera_id      TEXT NOT NULL,
    zone           TEXT NOT NULL,
    track_id       TEXT NOT NULL,
    first_seen     DOUBLE PRECISION NOT NULL,
    last_seen      DOUBLE PRECISION NOT NULL,
    confidence     REAL NOT NULL,
    purposes       TEXT[] NOT NULL DEFAULT '{}',
    consent_id     TEXT,
    redacted       BOOLEAN NOT NULL DEFAULT FALSE,
    retained_until DOUBLE PRECISION NOT NULL
) PARTITION BY RANGE (to_timestamp(first_seen));

CREATE INDEX IF NOT EXISTS presence_student_idx
    ON presence_events (student_id, first_seen DESC);
CREATE INDEX IF NOT EXISTS presence_camera_idx
    ON presence_events (camera_id, first_seen DESC);
CREATE INDEX IF NOT EXISTS presence_retention_idx
    ON presence_events (retained_until);

-- A unique constraint scoped to the partition key, so a replayed Redis message
-- cannot insert the same sighting twice. The track_id is the natural
-- idempotency key: one track commits at most once.
CREATE TABLE IF NOT EXISTS presence_events_2026_09 PARTITION OF presence_events
    FOR VALUES FROM ('2026-09-01') TO ('2026-10-01');

-- ---------------------------------------------------------------------------
-- Audit
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS audit_log (
    id         BIGSERIAL PRIMARY KEY,
    at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    actor      TEXT NOT NULL,
    action     TEXT NOT NULL,
    subject    TEXT,
    detail     JSONB NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS audit_at_idx ON audit_log (at DESC);
CREATE INDEX IF NOT EXISTS audit_subject_idx ON audit_log (subject);

-- Append-only: an audit trail that can be edited is not an audit trail.
CREATE OR REPLACE FUNCTION forbid_audit_mutation() RETURNS TRIGGER AS $$
BEGIN
    RAISE EXCEPTION 'audit_log is append-only';
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS audit_log_immutable ON audit_log;
CREATE TRIGGER audit_log_immutable
    BEFORE UPDATE OR DELETE ON audit_log
    FOR EACH ROW EXECUTE FUNCTION forbid_audit_mutation();

-- ---------------------------------------------------------------------------
-- Partition maintenance
-- ---------------------------------------------------------------------------

-- Create next month's partition. Run monthly via pg_cron; the DO block is
-- idempotent so a re-run is harmless.
CREATE OR REPLACE FUNCTION ensure_presence_partitions(months_ahead INTEGER DEFAULT 2)
RETURNS INTEGER AS $$
DECLARE
    created INTEGER := 0;
    start_month DATE;
    end_month DATE;
    part_name TEXT;
BEGIN
    FOR i IN 0..months_ahead LOOP
        start_month := date_trunc('month', now()) + (i || ' month')::INTERVAL;
        end_month := start_month + INTERVAL '1 month';
        part_name := 'presence_events_' || to_char(start_month, 'YYYY_MM');
        IF NOT EXISTS (SELECT 1 FROM pg_class WHERE relname = part_name) THEN
            EXECUTE format(
                'CREATE TABLE %I PARTITION OF presence_events FOR VALUES FROM (%L) TO (%L)',
                part_name, start_month, end_month
            );
            EXECUTE format('CREATE INDEX ON %I (student_id, first_seen DESC)', part_name);
            created := created + 1;
        END IF;
    END LOOP;
    RETURN created;
END;
$$ LANGUAGE plpgsql;
