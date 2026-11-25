-- 029_task_queue.sql — the DISTRIBUTED TASK FOUNDATION.
--
-- api/lib/task-queue.js is the source of truth and works with NO database
-- (local-first, in-memory). This migration is its DURABLE mirror, so a
-- gateway restart does not lose work that was already submitted.
--
-- The table is deliberately SHAPE-COMPATIBLE with the protocol vocabularies
-- it serves, so a row can be read by any of them:
--
--   state          queued | running | done | failed | expired | rejected | canceled
--   kind           chat | code | tool | bridge | a2a | generic
--   interrupted    {kind, reason} — an A2A task waiting for input or approval
--   budget         {max_tokens, max_cost_usd, max_duration_ms}
--   usage          {tokens, cost_usd, duration_ms}
--
-- WHY ONE TABLE FOR ALL OF THEM
-- ─────────────────────────────
-- The relay (inference jobs to NAT'd bees), the coder bees, the hive-to-hive
-- bridge and A2A all need the same nine things (a pollable state machine,
-- idempotency, priority, a TTL, LEASES so a dead worker loses nothing,
-- retries, a budget, cancellation, and a receipt). One table, one state
-- machine, one janitor — and a peer hive can read the same vocabulary we
-- publish.
--
-- IDEMPOTENT: safe to re-run.

CREATE TABLE IF NOT EXISTS hivv_task_queue (
    id                TEXT PRIMARY KEY,
    kind              TEXT NOT NULL DEFAULT 'generic',
    state             TEXT NOT NULL DEFAULT 'queued',

    priority          INTEGER NOT NULL DEFAULT 0,
    attempts          INTEGER NOT NULL DEFAULT 0,
    max_attempts      INTEGER NOT NULL DEFAULT 3,

    -- who
    origin            TEXT,                       -- the hive / peer / user that submitted
    context_id        TEXT,                       -- groups related tasks (a conversation)
    principal         TEXT,                       -- who is accountable
    trace_id          TEXT,                       -- inherited from the caller's trace

    -- what
    payload           JSONB,
    result            JSONB,
    error             JSONB,
    interrupted       JSONB,                      -- {kind: input|auth, reason}

    -- the ledger
    usage             JSONB NOT NULL DEFAULT '{}'::jsonb,
    budget            JSONB,
    metadata          JSONB NOT NULL DEFAULT '{}'::jsonb,

    -- the lease (a lock would strand work when a bee dies behind NAT)
    lease_worker      TEXT,
    lease_expires_at_ms BIGINT,

    -- when
    created_at_ms     BIGINT NOT NULL DEFAULT 0,
    updated_at_ms     BIGINT NOT NULL DEFAULT 0,
    started_at_ms     BIGINT,
    completed_at_ms   BIGINT,
    expires_at_ms     BIGINT NOT NULL DEFAULT 0,
    not_before_ms     BIGINT NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_hivv_task_queue_state
    ON hivv_task_queue (state, priority DESC, created_at_ms);
CREATE INDEX IF NOT EXISTS idx_hivv_task_queue_kind
    ON hivv_task_queue (kind, state, created_at_ms DESC);
CREATE INDEX IF NOT EXISTS idx_hivv_task_queue_context
    ON hivv_task_queue (context_id, created_at_ms DESC);
CREATE INDEX IF NOT EXISTS idx_hivv_task_queue_origin
    ON hivv_task_queue (origin, created_at_ms DESC);
CREATE INDEX IF NOT EXISTS idx_hivv_task_queue_trace
    ON hivv_task_queue (trace_id)
    WHERE trace_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_hivv_task_queue_lease
    ON hivv_task_queue (lease_expires_at_ms)
    WHERE state = 'running';
