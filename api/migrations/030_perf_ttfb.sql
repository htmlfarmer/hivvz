-- 030_perf_ttfb.sql — route on TIME TO FIRST TOKEN, not on total completion.
--
-- The router's latency factor needs the number a user actually waits on: the
-- gap between "request sent" and "first word arrived". hivv_model_perf only
-- stored ewma_latency_ms, which for a streaming answer is the WHOLE reply —
-- a bee that answers in 200ms and then writes for 40s looked 40s slow, and
-- fast mode ranked it last. The router now records ttfb on every completion
-- (api/lib/router.js stream loop) and prefers it; this column is where the
-- EWMA lives so it survives a gateway restart.
--
-- Safe to re-run (idempotent).

ALTER TABLE hivv_model_perf ADD COLUMN IF NOT EXISTS ewma_ttfb_ms REAL NOT NULL DEFAULT 0;

-- Fast mode sorts on TTFT more than on tps once it is populated.
CREATE INDEX IF NOT EXISTS idx_hivv_model_perf_ttfb
    ON hivv_model_perf (via, ewma_ttfb_ms);
