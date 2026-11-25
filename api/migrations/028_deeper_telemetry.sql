-- 028_deeper_telemetry.sql — the DEEPER TELEMETRY mirror.
--
-- api/lib/telemetry.js is the source of truth and works with NO database
-- (local-first). This migration is the DURABLE mirror of the parts that must
-- survive a gateway restart:
--
--   hivv_telemetry_counters — one row per (metric, label_key, kind)
--   hivv_telemetry_spans    — recent spans, for the trace waterfall
--   hivv_bridge_receipts    — hive-to-hive receipts (open-hives/0.1 §6)
--
-- The in-memory core and this mirror are written independently on purpose:
-- if the mirror is lost, the live view keeps working; if the core is lost,
-- the mirror keeps the history. Neither is a dependency of the other.
--
-- PRIVACY: no column here can hold a prompt. Labels are an allow-listed,
-- length-capped set of low-cardinality strings; spans carry the same
-- allow-listed attributes. The allow-list lives in lib/telemetry.js
-- (LABEL_ALLOW) — this schema is the second half of that contract.
--
-- Idempotent: safe to re-run.

-- ── the metric mirror ──────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS hivv_telemetry_counters (
    id           BIGSERIAL PRIMARY KEY,
    metric       TEXT NOT NULL,
    label_key    TEXT NOT NULL DEFAULT '',
    labels       JSONB NOT NULL DEFAULT '{}'::jsonb,
    value        DOUBLE PRECISION NOT NULL DEFAULT 0,
    kind         TEXT NOT NULL DEFAULT 'counter',   -- counter | gauge
    samples      BIGINT NOT NULL DEFAULT 0,
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (metric, label_key, kind)
);

CREATE INDEX IF NOT EXISTS idx_hivv_telemetry_metric
    ON hivv_telemetry_counters (metric, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_hivv_telemetry_kind
    ON hivv_telemetry_counters (kind, updated_at DESC);

-- ── the span mirror (the trace waterfall) ──────────────────────────────────
CREATE TABLE IF NOT EXISTS hivv_telemetry_spans (
    id             BIGSERIAL PRIMARY KEY,
    trace_id       TEXT NOT NULL,
    span_id        TEXT NOT NULL,
    parent_span_id TEXT,
    name           TEXT NOT NULL,
    kind           TEXT NOT NULL DEFAULT 'internal',
    status         TEXT NOT NULL DEFAULT 'ok',
    error_class    TEXT,
    duration_ms    BIGINT NOT NULL DEFAULT 0,
    start_ms       BIGINT NOT NULL DEFAULT 0,
    attributes     JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_hivv_telemetry_span_trace
    ON hivv_telemetry_spans (trace_id, start_ms);
CREATE INDEX IF NOT EXISTS idx_hivv_telemetry_span_created
    ON hivv_telemetry_spans (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_hivv_telemetry_span_error
    ON hivv_telemetry_spans (error_class, created_at DESC)
    WHERE error_class IS NOT NULL;

-- ── hive-to-hive receipts (open-hives/0.1 §6) ─────────────────────────────
-- One row per crossing. A receipt CONFIRMS what the peer measured; our own
-- counters are the ones we trust for our own traffic.
CREATE TABLE IF NOT EXISTS hivv_bridge_receipts (
    id          BIGSERIAL PRIMARY KEY,
    peer        TEXT NOT NULL,                  -- the remote hive's domain
    request_id  TEXT,                           -- the peer's request id
    node_id     TEXT,                           -- which node answered
    model       TEXT,
    tokens      BIGINT NOT NULL DEFAULT 0,
    latency_ms  INTEGER NOT NULL DEFAULT 0,
    status      TEXT NOT NULL DEFAULT 'unknown',
    cost_usd    NUMERIC(20,10) NOT NULL DEFAULT 0,
    persona_id  TEXT,                           -- which strange mind answered
    receipt     JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (peer, request_id, node_id, created_at)
);

CREATE INDEX IF NOT EXISTS idx_hivv_bridge_receipts_peer
    ON hivv_bridge_receipts (peer, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_hivv_bridge_receipts_status
    ON hivv_bridge_receipts (status, created_at DESC);
