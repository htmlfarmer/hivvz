#!/usr/bin/env bash
# bridge-peer-test.sh — HIVV <-> KwaaiNet bridge smoke test.
#
# Purpose: get to know each other. Exercise the open-hives/0.1 doors in
# BOTH directions and report what each side answers, so we learn the shape
# of the other hive before we trust each other with real traffic.
#
# Scope, deliberately narrow:
#   * We call HIVV's bridge API (the thing we already serve).
#   * HIVV calls OUR KwaaiNet node, on loopback, for compute only.
#   * NO traffic flows from the KwaaiNet network to us. Nothing a KwaaiNet
#     node holds is read, fetched, or mirrored here.
#   * Requests are few and serial. A benchmark that melts the hive measures
#     nothing (open-hives 12.6) — so this is a smoke test, not a benchmark.
#
# Usage:
#   ./scripts/bridge-peer-test.sh [--local]
#     --local  target http://localhost:3011 instead of https://hivv.org
#
# Exit: 0 if every door answered, 1 if any door failed.

set -uo pipefail

HUB="https://hivv.org"
# The manifest doors are STATIC FILES served by the web server from
# public_html/, not by the API gateway. So --local (which points at the
# gateway on :3011) cannot reach them; it needs the web server's own port.
WEB="https://hivv.org"
WEB_RESOLVE=()
if [[ "${1:-}" == "--local" ]]; then
  HUB="http://localhost:3011"
  # The manifest doors are static files served by Caddy (file_server over
  # public_html/); /api/* is proxied to the gateway on :3011. So --local
  # talks to the gateway directly and to Caddy on :443 with the Host pinned
  # to hivv.org — still entirely on this box, no traffic off-host.
  WEB="https://hivv.org"
  WEB_RESOLVE=(--resolve "hivv.org:443:127.0.0.1")
fi

# Our own KwaaiNet node's door, on loopback.
NODE_MODELS="${KWAAINET_MODELS_URL:-http://localhost:18090/v1/models}"
NODE_CHAT="${KWAAINET_CHAT_URL:-http://localhost:18090/v1/chat/completions}"
PEER="kwaai"

pass=0; fail=0
ok()   { printf '  \033[32m✓\033[0m %s\n' "$1"; pass=$((pass+1)); }
bad()  { printf '  \033[31m✗\033[0m %s\n' "$1"; fail=$((fail+1)); }
note() { printf '  \033[2m·\033[0m %s\n' "$1"; }

# probe <name> <method> <url> [extra curl args...]
probe() {
  local name="$1" method="$2" url="$3"; shift 3
  local body code
  body=$(curl -s -m 20 -X "$method" "$url" -w '\n__CODE__%{http_code}' "$@" 2>/dev/null)
  code=$(printf '%s' "$body" | tail -1 | sed 's/__CODE__//')
  body=$(printf '%s' "$body" | sed '$d')
  if [[ "$code" =~ ^[23] ]]; then
    ok "$name (HTTP $code)"
    printf '%s' "$body"
  else
    bad "$name (HTTP ${code:-none})"
    printf '%s' "$body"
  fi
}

echo "══ HIVV <-> KwaaiNet bridge smoke test ══"
echo "   hub: $HUB"
echo "   web: $WEB   (static manifest doors, served by Caddy)"
echo "   node: $NODE_CHAT"
echo ""

# ── 1. OUR DOOR: the KwaaiNet node answers ───────────────────────────────
echo "[1] KwaaiNet node (our side, loopback)"

models=$(curl -s -m 15 "$NODE_MODELS" 2>/dev/null)
if printf '%s' "$models" | grep -q '"data"'; then
  m=$(printf '%s' "$models" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["data"][0]["id"])' 2>/dev/null)
  ok "node /v1/models lists: $m"
else
  bad "node /v1/models did not answer"
  m="auto"
fi

# A real answer, from the network. This is the only compute call we make.
ans=$(curl -s -m 180 -X POST "$NODE_CHAT" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$m\",\"max_tokens\":24,\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: BRIDGE OK\"}]}" \
  2>/dev/null | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])' 2>/dev/null)
if [[ -n "$ans" ]]; then
  ok "node answered: $(printf '%s' "$ans" | tr '\n' ' ' | cut -c1-70)"
else
  bad "node did not answer"
fi
echo ""

# ── 2. HIVV's DOORS: what a peer hive exposes ───────────────────────────
echo "[2] HIVV bridge API (their side)"

for route in load allowance peers problems heartbeat; do
  probe "$route" GET "$HUB/api/bridge/$route" >/dev/null 2>&1
  # re-run visibly
  code=$(curl -s -m 20 -o /dev/null -w '%{http_code}' "$HUB/api/bridge/$route")
  if [[ "$code" =~ ^[23] ]]; then ok "GET /api/bridge/$route (HTTP $code)"; else bad "GET /api/bridge/$route (HTTP ${code:-none})"; fi
done

code=$(curl -s -m 20 -o /dev/null -w '%{http_code}' "$WEB/.bridge/hivv.json" "${WEB_RESOLVE[@]}")
if [[ "$code" == "200" ]]; then ok "GET /.bridge/hivv.json (their manifest)"; else bad "GET /.bridge/hivv.json (HTTP ${code:-none})"; fi

code=$(curl -s -m 20 -o /dev/null -w '%{http_code}' "$WEB/.well-known/hivv.json" "${WEB_RESOLVE[@]}")
if [[ "$code" == "200" ]]; then ok "GET /.well-known/hivv.json (standard door)"; else bad "GET /.well-known/hivv.json (HTTP ${code:-none})"; fi
echo ""

# ── 3. THE SUBMIT DOOR: we post, they work, they receipt ────────────────
echo "[3] Submit door (the two-way work path)"

# A submission is a request for HIVV to DO something. We are only checking
# that the door opens and the bookkeeping lands — we are not asking for
# anything expensive, and HIVV is not being asked to read our node.
resp=$(curl -s -m 30 -X POST "$HUB/api/bridge/requests" \
  -H 'Content-Type: application/json' \
  -H "x-hive-origin: $PEER" \
  -H "x-hive-request-id: smoke_$(date +%s)" \
  -d '{"kind":"chat","payload":{"probe":"bridge smoke test"}}' 2>/dev/null)
code=$(printf '%s' "$resp" | python3 -c 'import json,sys
try:
    d=json.load(sys.stdin); print(d.get("status") or d.get("id") or "ok")
except Exception: print("?")' 2>/dev/null)
if [[ "$code" == "202" || "$code" == "accepted" || "$code" == "queued" ]]; then
  ok "POST /api/bridge/requests accepted ($code)"
elif printf '%s' "$resp" | grep -q 'accepted\|queued\|202'; then
  ok "POST /api/bridge/requests accepted"
else
  note "POST /api/bridge/requests -> $(printf '%s' "$resp" | head -c 120)"
fi

code=$(curl -s -m 20 -o /dev/null -w '%{http_code}' "$HUB/api/bridge/receipts")
if [[ "$code" =~ ^[23] ]]; then ok "GET /api/bridge/receipts (HTTP $code)"; else bad "GET /api/bridge/receipts (HTTP ${code:-none})"; fi
echo ""

# ── 4. THEIR MANIFEST: what do they say their chat door is? ─────────────
echo "[4] What their manifest claims"

manifest=$(curl -s -m 20 "$WEB/.bridge/hivv.json" "${WEB_RESOLVE[@]}" 2>/dev/null)
if [[ -n "$manifest" ]]; then
  printf '%s' "$manifest" | python3 -c '
import json,sys
try: m=json.load(sys.stdin)
except Exception: sys.exit(0)
print("    protocol :", m.get("protocol"))
print("    hive     :", m.get("hive"))
print("    status   :", m.get("status"))
print("    auth     :", (m.get("auth") or {}).get("scheme"))
eps=m.get("endpoints") or {}
for k in ("chat","load","allowance","state","submit","receipts"):
    if k in eps: print(f"    {k:9}: {str(eps[k])[:60]}")
' 2>/dev/null
fi
echo ""

echo "══ $pass passed, $fail failed ══"
[[ "$fail" -eq 0 ]]
