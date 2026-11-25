-- Migration 031: identity revocation (Phase 4 of the identity layer).
--
-- A revoked Ed25519 key presenting itself at register/heartbeat is an EVENT:
-- the hub files one deduplicated `identity_revoked` incident per node so an
-- operator sees CRL hits in `hivv incidents` + the Command Center.
-- (CHECK constraints can't be ALTERed in place — same pattern as 017.)

ALTER TABLE hivv_incidents DROP CONSTRAINT IF EXISTS chk_incident_type;
ALTER TABLE hivv_incidents ADD CONSTRAINT chk_incident_type CHECK (incident_type IN (
    'node_offline', 'node_stale', 'node_restart_loop', 'node_crash',
    'port_unreachable', 'model_unavailable', 'network_partition',
    'hub_unreachable', 'auth_failure', 'config_error',
    'upgrade_suspicious', 'upgrade_failed',
    'identity_revoked'
));
