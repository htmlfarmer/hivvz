const { AIProvider, AIKey, ProviderClient, StreamCapture } = require('./providers');
const CircuitBreaker = require('./circuit-breaker');

const { PROVIDER_DEFAULTS } = require('./providers');
const { looksDegenerate } = require('./degenerate');

// ── token budgeting ────────────────────────────────────────────────────────
//
// THE PROBLEM THIS SOLVES
// ───────────────────────
// The old code hardcoded `contextLimit = 14000` and `maxTokens ≤ 2048`.
// A local model with a 32k context (gemma-4-E2B) was being told it had 14k,
// and OpenCode's coding responses were being cut at 2048 output tokens —
// which produced "Response still truncated after two output token limit
// continuations" on every non-trivial edit.
//
// THE FIX
// ───────
// 1. Use the model's REAL context window (looked up from a name-based table,
//    overridable per request via `options.contextWindow`).
// 2. When the prompt is too large, TRIM old conversation messages (keep the
//    system prompt and the most recent turns) instead of shrinking max_tokens
//    to almost nothing.
// 3. Give output a generous default (4096) and let it grow with the context
//    window, so a coding response has room to breathe.
// 4. Return a clear `truncated` flag when we had to drop messages, so callers
//    can tell the user rather than silently losing context.
//
// The token estimate is the same ~4 chars/token heuristic the rest of the
// codebase uses. It is deliberately approximate: over-estimating input is the
// safe direction (we trim early rather than overflow).

// Known context windows by model-name fragment. Ordered: most specific first.
// Anything not matched falls back to DEFAULT_CONTEXT_WINDOW.
const CONTEXT_WINDOWS = [
    [/gemma[-_ ]?4/i, 32768],
    [/gemma/i, 8192],
    [/llama[-_ ]?3[._-]?\d/i, 131072],
    [/llama[-_ ]?3/i, 8192],
    [/llama/i, 4096],
    [/qwen[-_ ]?3/i, 32768],
    [/qwen/i, 8192],
    [/mistral/i, 32768],
    [/phi[-_ ]?3/i, 131072],
    [/phi/i, 8192],
    [/deepseek/i, 65536],
    [/codellama/i, 16384],
    [/gpt-oss/i, 131072],
];
const DEFAULT_CONTEXT_WINDOW = 8192;

// How much output the model should be allowed to produce, as a function of
// its context window. Coding responses (OpenCode edits, agent tool loops)
// routinely need 2-4k tokens; a reasoning model may need more.
function outputBudgetFor(contextWindow) {
    if (contextWindow >= 131072) return 8192;
    if (contextWindow >= 32768) return 4096;
    if (contextWindow >= 16384) return 3072;
    return 2048;
}

function contextWindowFor(modelName, override) {
    if (Number.isFinite(override) && override > 0) return override;
    const name = String(modelName || '');
    for (const [re, ctx] of CONTEXT_WINDOWS) {
        if (re.test(name)) return ctx;
    }
    return DEFAULT_CONTEXT_WINDOW;
}

// Rough token estimate: ~4 chars/token, with a floor so an empty prompt is 0.
function estimateTokens(messages) {
    const chars = (messages || []).reduce(
        (sum, m) => sum + (typeof m?.content === 'string' ? m.content.length
            : JSON.stringify(m?.content ?? '').length), 0);
    return Math.ceil(chars / 4);
}

// ── CONTEXT OVERFLOW IS NOT A PROVIDER FAILURE ────────────────────────
// llama.cpp answers a prompt that is too big with a 400 and a body of
//     {"error":{"code":400,"type":"exceed_context_size_error",
//               "message":"request (18766 tokens) exceeds the
//                          available context size (16384 tokens)"}}
//
// The old code read `if (!response.ok) { … continue; }` and threw the
// body away, so EVERY provider was tried, every one failed the same
// way, and the client got "All providers unavailable" (503). That says
// "the hive is down" when the truth is "your prompt was too big" — and
// it is the single most misleading failure this router can produce,
// because the hive is perfectly healthy.
//
// Detect it so the caller can be told the truth (and so the budget
// trimmer can be trusted to have done its job).
function isContextOverflow(status, bodyText) {
    if (!bodyText) return false;
    if (!/exceed_context_size|context size|n_ctx|too many tokens|maximum context/i.test(bodyText)) {
        return false;
    }
    // A 400/413 with that text is the overflow. llama.cpp uses 400; a
    // gateway in front of it may use 413.
    return status === 400 || status === 413 || status === 422;
}

/**
 * Decide how many output tokens to allow, and whether the conversation must
 * be trimmed to fit.
 *
 * @param {Array}  messages       the conversation to send
 * @param {object} opts
 * @param {string} opts.modelName     the model being targeted
 * @param {number} [opts.requestedMax]  the client's requested max_tokens
 * @param {number} [opts.contextWindow] override the detected context window
 * @returns {{maxTokens:number, messages:Array, contextWindow:number,
 *            inputTokens:number, trimmed:boolean, dropped:number}}
 */
function _budgetTokens(messages, { modelName, requestedMax, contextWindow } = {}) {
    const msgs = Array.isArray(messages) ? messages : [];
    const ctx = contextWindowFor(modelName, contextWindow);
    // Leave 10% headroom for the model's own bookkeeping (chat template,
    // BOS/EOS, tool-call framing). Over-estimating input is the safe side.
    const usable = Math.floor(ctx * 0.9);
    const outBudget = outputBudgetFor(ctx);

    const inputTokens = estimateTokens(msgs);
    let trimmed = false;
    let dropped = 0;
    let out = msgs;

    // If the prompt plus a minimum output budget does not fit, drop the OLDEST
    // non-system messages until it does. The system prompt (persona, tool
    // definitions, identity) is never dropped — losing it breaks every tool
    // call. The most recent turns are kept so the model still has context for
    // what the user just asked.
    const MIN_OUTPUT = Math.min(1024, outBudget);
    if (inputTokens + MIN_OUTPUT > usable) {
        const system = out.filter((m) => m?.role === 'system');
        const conversation = out.filter((m) => m?.role !== 'system');
        // Keep at most the last 8 non-system messages (4 turns).
        const KEEP = 8;
        if (conversation.length > KEEP) {
            dropped = conversation.length - KEEP;
            out = [...system, ...conversation.slice(-KEEP)];
            trimmed = true;
        }
    }

    // ── A SINGLE OVERSIZED MESSAGE ──────────────────────────────────────
    // The trim above only fires when there are MORE THAN EIGHT non-system
    // messages. An agent client (OpenCode, Claude Code, …) sends ONE
    // message carrying the whole system prompt + tool definitions +
    // conversation, so `conversation.length === 1` and NOTHING was
    // trimmed — the oversized message went to the bee whole.
    //
    // The bee then answered with llama.cpp's own 400:
    //   "request (18766 tokens) exceeds the available context size
    //    (16384 tokens)"
    // and the router treated it as a generic provider failure, so every
    // provider was tried, every one failed the same way, and the client
    // got "All providers unavailable" (503) — which reads like the hive
    // being down, not like a prompt that was too big.
    //
    // So when the estimate is still over budget, TRUNCATE THE LONGEST
    // MESSAGE by characters (chars/4 tokens, so the char budget is
    // derived from the token budget). The message is kept — a truncated
    // prompt still answers; a dropped one answers nothing.
    const afterTurnTrim = estimateTokens(out);
    if (afterTurnTrim + MIN_OUTPUT > usable) {
        // Characters we may keep in the whole conversation.
        const charBudget = Math.max(2000, (usable - MIN_OUTPUT) * 4);
        let chars = 0;
        const kept = [];

        // THE SYSTEM PROMPT IS RESERVED FIRST.
        //
        // It carries the identity and the tool definitions, and the tool
        // definitions are what make a tool call possible at all — a
        // conversation without them cannot use tools, and a system
        // message trimmed away entirely breaks every tool call. So its
        // share of the budget is taken OFF THE TOP, before any
        // conversation message competes for the rest.
        //
        // The order matters: the previous version walked newest-first and
        // let a large newest message eat the whole budget, which left
        // NO room for the system prompt and silently dropped it.
        const systemMsgs = out.filter((m) => m?.role === 'system');
        const convMsgs = out.filter((m) => m?.role !== 'system');
        const sysChars = systemMsgs.reduce((sum, m) => sum + (typeof m?.content === 'string' ? m.content.length
            : JSON.stringify(m?.content ?? '').length), 0);
        // The system prompt gets at most half the budget, and at least
        // enough to carry an identity and a tool list.
        const sysBudget = Math.min(sysChars, Math.max(4000, Math.floor(charBudget / 2)));
        const convBudget = Math.max(0, charBudget - sysBudget);

        // The system messages, truncated to their reserved share.
        const keptSystem = [];
        let sysUsed = 0;
        for (const m of systemMsgs) {
            const len = (typeof m?.content === 'string') ? m.content.length
                : JSON.stringify(m?.content ?? '').length;
            const room = Math.max(0, sysBudget - sysUsed);
            if (room <= 0) { dropped += 1; continue; }
            if (len <= room) {
                keptSystem.push(m);
                sysUsed += len;
            } else {
                const c = typeof m.content === 'string' ? m.content : JSON.stringify(m.content ?? '');
                keptSystem.push({ ...m, content: c.slice(0, room) });
                sysUsed += room;
                dropped += 1;
            }
        }

        // The conversation, newest first, each one TRUNCATED rather than
        // dropped when it does not fit whole — a truncated prompt still
        // answers, a dropped one answers nothing.
        const keptConv = [];
        let convUsed = 0;
        for (let i = convMsgs.length - 1; i >= 0; i--) {
            const m = convMsgs[i];
            const len = (typeof m?.content === 'string') ? m.content.length
                : JSON.stringify(m?.content ?? '').length;
            if (convUsed + len <= convBudget) {
                keptConv.unshift(m);
                convUsed += len;
                continue;
            }
            const room = Math.max(0, convBudget - convUsed);
            if (room > 0) {
                const c = typeof m?.content === 'string' ? m.content : JSON.stringify(m?.content ?? '');
                keptConv.unshift({ ...m, content: c.slice(0, room) });
                convUsed += room;
            }
            dropped += 1;
        }

        const rebuilt = [...keptSystem, ...keptConv];
        if (rebuilt.length) {
            out = rebuilt;
            trimmed = true;
        }
    }

    const finalInput = trimmed ? estimateTokens(out) : inputTokens;
    const hardMaxOutput = Math.max(MIN_OUTPUT, usable - finalInput);

    // The client's requested max_tokens wins when it is smaller (they know
    // what they want). When it is larger we honour it up to the budget.
    let maxTokens;
    if (Number.isFinite(requestedMax) && requestedMax > 0) {
        maxTokens = Math.min(requestedMax, hardMaxOutput);
    } else {
        maxTokens = Math.min(outBudget, hardMaxOutput);
    }
    // A floor of 256 so a nearly-full context still produces *something* the
    // caller can see, rather than an empty response that looks like a crash.
    maxTokens = Math.max(256, maxTokens);

    return {
        maxTokens,
        messages: out,
        contextWindow: ctx,
        inputTokens: finalInput,
        trimmed,
        dropped,
    };
}

class AIRouter {
    constructor(pool) {
        this.pool = pool;
        this.providers = new Map();
        this.keysByProvider = new Map();
        // Local endpoints from env var (comma-separated), or empty if not configured
        const envLocal = process.env.HIVV_LOCAL_ENDPOINTS || process.env.LOCAL_AI_ENDPOINTS || '';
        this.localEndpoints = envLocal.split(',')
            .filter(Boolean)
            .map((url, i) => ({ url: url.trim(), name: `local-${i + 1}`, provider: 'local' }));
        this.localProviderKeys = new Map();
        this.loadBalancingStrategy = 'priority';
        this.failoverEnabled = true;
        this.maxRetries = 3;
        this.circuitBreaker = new CircuitBreaker({
            failureThreshold: 5,
            successThreshold: 2,
            timeout: 30000
        });
        // Smart Spare-Time Donation: tracks if local nodes are actually busy.
        // Default: if a node takes > 1s to respond to a micro-request, we assume the OWNER is using it.
        // In that case, we skip donating its time to hivv. If it responds in < 300ms, it's IDLE -> Donate it.
        this.spareTimeTracker = new Map();
        this.startSpareTimeMonitor();
        // Reference to orchestrator for node-based routing
        this.orchestrator = null;
        // Measured-routing + relay libs (injected by server.js; optional)
        this.perf = null;
        this.relay = null;
        // hivv OS: capability/policy gate + pluggable route modes (§2/§3)
        this.policy = null;
        this.caps = null;
        this._rankCache = { map: new Map(), at: 0 };
        this._priceMap = new Map();
        this._routeCfg = {
            jitter: 8, unknown_tps: 8,
            quality_weights: { 'gpt-4': 10, 'o1': 10, 'o3': 10, claude: 10, gemini: 9, llama: 8, qwen: 8, gemma: 7, mistral: 7, phi: 5 },
        };
    }

    setOrchestrator(orch) {
        this.orchestrator = orch;
    }

    setPerf(perfLib) { this.perf = perfLib; }
    setRelay(relayLib) {
        this.relay = relayLib;
        // B3: apply any gateway_config.relay quotas loaded before the relay attached
        if (this._relayCfg && relayLib?.setConfig) {relayLib.setConfig(this._relayCfg);}
    }
    setPolicy(policyLib) { this.policy = policyLib; }
    setCapabilities(capsLib) { this.caps = capsLib; }

    // Map an endpoint URL hostname to a registered node (for receipts + perf)
    _nodeByHost(host) {
        if (!this.orchestrator) return null;
        for (const n of this.orchestrator.nodes.values()) {
            if (n.name === host || n.node_id === host || n.ip === host ||
                n.lan_ip === host || n.public_ip === host) return n;
        }
        return null;
    }

    static _prettyModel(m) {
        m = String(m || 'local');
        if (m.includes('/')) m = m.replace(/\/+$/, '').split('/').pop();
        return m.replace(/\.(gguf|bin|safetensors)$/i, '').slice(0, 40);
    }

    _routeLabel(local) {
        try {
            const host = new URL(local.url).hostname;
            const node = this._nodeByHost(host);
            return node ? (node.name || node.node_id) : host;
        } catch { return local.name || 'local'; }
    }

    async initialize() {
        await this.loadProviders();
        await this.loadKeys();
        await this.loadConfig();
        await this.discoverLocalEndpoints();
    }

    async loadProviders() {
        if (!this.pool) return;
        try {
            const result = await this.pool.query('SELECT * FROM ai_providers WHERE is_active = TRUE');
            this.providers.clear();
            for (const row of result.rows) {
                this.providers.set(row.name, new AIProvider(row));
                this._priceMap.set(row.name, {
                    in: parseFloat(row.cost_in_per_mtok) || 0,
                    out: parseFloat(row.cost_out_per_mtok) || 0,
                });
            }
        } catch (err) {
            console.error('Failed to load providers:', err.message);
        }
    }

    // ---- hivv OS helpers (§2 route modes / §3 gate) ---------------------
    _isLocalUrl(url) {
        try {
            const h = new URL(url).hostname;
            return h === 'localhost' || h === '127.0.0.1' || h === '[::1]' || h === '::1';
        } catch { return false; }
    }

    // §2 COST signal: does this provider bust the request's max_cost budget?
    // Rough estimate: ~256 prompt + ~512 completion tokens.
    _overCostBudget(providerName, route = {}) {
        if (route == null || route.max_cost == null || route.max_cost === '') return false;
        const max = parseFloat(route.max_cost);
        if (!isFinite(max)) return false;
        const price = this._priceMap.get(providerName) || { in: 0, out: 0 };
        const est = (price.in * 256 + price.out * 512) / 1e6;
        return est > max;
    }

    async _ensureRanks() {
        if (Date.now() - this._rankCache.at < 60000 && this._rankCache.map.size) return this._rankCache.map;
        try {
            const r = await this.pool.query('SELECT node_id, bee_rank, honey FROM hivv_node_stats');
            this._rankCache = { map: new Map(r.rows.map(x => [x.node_id, x])), at: Date.now() };
        } catch {
            this._rankCache = { map: new Map(), at: Date.now() };
        }
        return this._rankCache.map;
    }

    // Can this candidate carry OpenAI tool calls? Cloud key endpoints are assumed
    // capable; local/peer models must match a known tool-capable family (hub-side
    // truth: model name is all we get from llama.cpp/ollama /v1/models).
    _toolCapable(t) {
        if (!this._isLocalUrl(t.url) && !t.nodeId) return true; // cloud/provider-key target
        return AIRouter.TOOL_CAPABLE_RE.test(String(t.model || ''));
    }

    // Warm-cache stickiness: the same principal returning within 3 minutes keeps
    // their bee. OpenCode ships a ~5k-token cold prompt on EVERY turn — hopping
    // bees re-pays prompt processing each time (15-45s observed), while a warm
    // llama.cpp prompt cache makes turn 2+ near-instant. Keyed by principal
    // (user:<id> / visitor id) from the hivv OS envelope.
    _stickySet(options, key) {
        const p = options.principal || options.userId;
        if (!p || !key) return;
        if (!this._sticky) this._sticky = new Map();
        this._sticky.set(String(p), { key, until: Date.now() + 180000 });
        if (this._sticky.size > 500) {
            for (const [k, v] of this._sticky) if (v.until < Date.now()) this._sticky.delete(k);
        }
    }

    _stickyGet(options) {
        const p = options.principal || options.userId;
        if (!p || !this._sticky) return null;
        const e = this._sticky.get(String(p));
        if (!e) return null;
        if (Date.now() > e.until) { this._sticky.delete(String(p)); return null; }
        return e.key;
    }

    // The exact stick-key format _scoreTargets compares against — the dial
    // endpoint and /api/route/report reuse it so shim-side affinity lines up.
    stickyKeyFor(t) {
        return `${this._routeLabel(t)}::${AIRouter._prettyModel(t.model || 'local')}`;
    }

    _qualityOf(modelStr) {
        const s = String(modelStr || '').toLowerCase();
        let best = 5;
        for (const [k, v] of Object.entries(this._routeCfg.quality_weights || {})) {
            if (s.includes(k)) best = Math.max(best, v);
        }
        return best;
    }

    // Model families known to emit OpenAI tool_calls through /v1/chat/completions.
    static TOOL_CAPABLE_RE = /(gpt-|gpt-?oss|\bo[134]\b|claude|gemini|gemma|qwen|llama-?3|llama3|mistral|ministral|codestral|devstral|command-r|nemo|granite|codellama|starcoder|deepseek|phi-?[34]|glm|kimi|minicpm|aya)/i;

    // Score every candidate under the requested mode; returns sorted [{t, score, dbg}]
    async _scoreTargets(targets, mode = 'balanced', options = {}) {
        let rank = null;
        try { rank = this.perf ? this.perf.ranking() : null; } catch (_e) {}
        const tpsCache = new Map(); const latCache = new Map(); const okCache = new Map(); const ptpsCache = new Map();
        if (rank) {
            for (const r of rank) {
                if (r.via && r.via !== 'direct') continue; // relay tps includes queue+poll wait — never a direct-speed signal
                if (!(r.ok > 0)) continue;                 // error-only rows carry NO speed signal — else a dead bee (0ms/0tps) wins fast mode
                const k = `${r.node}::${r.model}`;
                if (r.tps > 0) tpsCache.set(k, Math.max(tpsCache.get(k) || 0, r.tps));
                if (r.prompt_tps > 0) ptpsCache.set(k, Math.max(ptpsCache.get(k) || 0, r.prompt_tps));
                if (r.latency_ms > 0 && (!latCache.has(k) || r.latency_ms < latCache.get(k))) latCache.set(k, r.latency_ms);
                // Reliability: how often this model actually answered. A bee whose
                // model errors on 1 request in 3 is NOT a fast bee, it is a broken
                // one — without this the tps lottery kept handing long prompts to
                // a 0.5B model that answers "I'm sorry, but I can't assist".
                const total = (r.ok || 0) + (r.err || 0);
                if (total > 0) {
                    const rate = (r.ok || 0) / total;
                    if (!okCache.has(k) || rate > okCache.get(k)) okCache.set(k, rate);
                }
            }
        }
        const rankMap = mode === 'trusted' ? await this._ensureRanks() : new Map();
        const RANK_SCORE = { 'queen-bee': 6, 'royal-jelly': 5, scout: 4, forager: 3, worker: 2, larva: 1 };
        const cfg = this._routeCfg;
        const stick = this._stickyGet(options);
        const bigPrompt = (options.promptEst || 0) > 1200; // ≈OpenCode's 5k-token system prompt
        const scored = targets.map(t => {
            const label = this._routeLabel(t);
            const knownTps = [...tpsCache.entries()].filter(([k]) => k.startsWith(`${label}::`)).map(([, v]) => v);
            const knownPtps = [...ptpsCache.entries()].filter(([k]) => k.startsWith(`${label}::`)).map(([, v]) => v);
            const knownLat = [...latCache.entries()].filter(([k]) => k.startsWith(`${label}::`)).map(([, v]) => v);
            const knownOk = [...okCache.entries()].filter(([k]) => k.startsWith(`${label}::`)).map(([, v]) => v);
            const tps = knownTps.length ? Math.max(...knownTps) : cfg.unknown_tps;
            // prompt-processing speed: unknown bees get a modest 300 t/s prior so
            // measured-fast prompt caches beat them, but unknown isn't zero
            const ptps = knownPtps.length ? Math.max(...knownPtps) : 300;
            const lat = knownLat.length ? Math.min(...knownLat) : 5000;
            // rel ∈ [0.25, 1] — a flaky model keeps a floor (so it is still
            // reachable when it is the only bee left) but never outranks a
            // healthy one. 0.25 + 0.75 * successRate²
            const rel = knownOk.length ? 0.25 + 0.75 * Math.pow(Math.max(...knownOk), 2) : 1;
            const bee = rankMap.get(label);
            const trust = RANK_SCORE[bee?.bee_rank] || 0;
            const free = this._isLocalUrl(t.url) || t.nodeId ? 100 : 0;
            let score;
            switch (mode) {
            case 'quality': score = this._qualityOf(t.model) * 12 + tps * 0.15 * rel + Math.random() * 4; break;
            case 'fast':    score = (20000 / Math.max(lat, 50)) * rel + Math.random() * 4; break;
            case 'cost':    score = free + tps * 0.1 * rel + Math.random() * 4; break;
            case 'trusted': score = trust * 20 + tps * 0.1 * rel + Math.random() * 4; break;
            case 'private':
            case 'balanced':
            default:        score = tps * rel + Math.random() * cfg.jitter;
            }
            // BIG PROMPT (≈OpenCode): a cold 5k-token prompt costs 15-45s on a
            // slow bee — prompt-processing speed becomes the dominant term;
            // hub-local gets a bonus (no LAN hop, most likely warm cache).
            // The mode's intent survives at reduced weight.
            if (bigPrompt) {
                score = ptps * 0.05 * rel + score * 0.3
                      + (this._isLocalUrl(t.url) ? 30 : 0)
                      + Math.random() * cfg.jitter;
            }
            // WARM STICKY: a principal returning within 3 min keeps their bee —
            // the warm prompt cache beats any cold candidate (skip if unhealthy).
            if (stick && rel > 0.4 &&
                `${label}::${AIRouter._prettyModel(t.model || 'local')}` === stick) {
                score += 45;
            }
            return { t, score, dbg: { node: label, model: AIRouter._prettyModel(t.model || 'local'), tps: Math.round(tps), prompt_tps: Math.round(ptps), latency_ms: Math.round(lat), quality: this._qualityOf(t.model), trust, free: free > 0, reliability: Number(rel.toFixed(3)) } };
        });
        scored.sort((a, b) => b.score - a.score);
        return scored;
    }

    async _orderByMode(targets, mode, options = {}) {
        if (options.privacy === 'localhost') targets = targets.filter(t => this._isLocalUrl(t.url));
        // OpenCode-style clients send `tools` — only route to candidates that can
        // actually emit tool_calls. If the filter would empty the list, keep the
        // original: never break a request by routing it nowhere (A3 of the plan).
        if (options.tools && options.tools.length) {
            const capable = targets.filter(t => this._toolCapable(t));
            if (capable.length) targets = capable;
            else console.log('[ROUTE] no tool-capable candidate known — routing unfiltered');
        }
        if (targets.length < 2) return targets;
        const scored = await this._scoreTargets(targets, mode, options);
        return scored.map(s => s.t);
    }

    // Dry-run: what WOULD happen — candidates, scores, transports, policy verdict.
    // Executes nothing (the anti-single-protocol test harness).
    async routePreview({ capability = 'reasoning', mode = 'balanced', privacy = 'any', principal = null } = {}) {
        if (mode === 'private' && (!privacy || privacy === 'any')) privacy = 'hive'; // same mapping as routeWithFailover
        let gate = null;
        try {
            if (this.policy) gate = await this.policy.gate({ capability, principal, mode, transport: 'preview' }, { dryRun: true });
        } catch (e) { gate = { verdict: 'error', rule: e.message }; }
        const targets = this.getAllEndpoints();
        const filtered = privacy === 'localhost' ? targets.filter(t => this._isLocalUrl(t.url)) : targets;
        const scored = await this._scoreTargets(filtered, mode);
        const relay = [];
        if (this.orchestrator && privacy !== 'localhost') {
            for (const n of this.orchestrator.nodes.values()) {
                if (n.status === 'online' && (n.volunteer === true || n.volunteer?.on === true)) {
                    relay.push({ node: n.name || n.node_id, models: (n.local_models || []).length });
                }
            }
        }
        const cloud = [];
        if (privacy === 'any') {
            for (const p of this.providers.values()) {
                if (!p.isCloud) continue;
                const price = this._priceMap.get(p.name) || { in: 0, out: 0 };
                cloud.push({ provider: p.name, cost_per_512_out_est_usd: +(price.out / 1e6 * 512).toFixed(4) });
            }
        }
        return {
            capability, mode, privacy,
            policy: gate,
            transports: {
                direct: scored.slice(0, 10).map(s => ({ ...s.dbg, score: +s.score.toFixed(1) })),
                relay,
                cloud,
            },
            order: privacy === 'localhost' ? ['local'] : privacy === 'hive' ? ['local', 'direct', 'relay'] : ['local', 'direct', 'relay', 'cloud'],
            ts: Date.now(),
        };
    }

    // LOCAL MODE: provider keys loaded from files instead of Postgres.
    // Format (one JSON file per provider in $HIVV_DATA_DIR/keys):
    //   { "api_key": "...", "api_endpoint": "https://api.openai.com/v1",
    //     "model": "gpt-4o-mini", "name": "openai" }
    async loadLocalFileKeys(keys) {
        if (!Array.isArray(keys) || !keys.length) return;
        this.keysByProvider.clear();
        this.localProviderKeys.clear();
        for (const spec of keys) {
            const name = String(spec.name || 'local-provider').toLowerCase();
            const providerDef = PROVIDER_DEFAULTS[name] || {};
            const provider = new AIProvider({
                name,
                display_name: spec.name || name,
                api_format: spec.api_format || providerDef.api_format || 'openai',
                default_endpoint: spec.api_endpoint || providerDef.defaultEndpoint || null,
                is_cloud: providerDef.isCloud !== false,
            });
            const key = new AIKey({
                id: `local:${name}`,
                provider_id: `local:${name}`,
                key_name: name,
                key_value: spec.api_key,
                api_endpoint: spec.api_endpoint || providerDef.defaultEndpoint || null,
                model: spec.model || null,
                priority: spec.priority || 50,
                weight: 1,
                is_active: true,
                cost_per_1k_input_tokens: spec.cost_in_per_1k || 0,
                cost_per_1k_output_tokens: spec.cost_out_per_1k || 0,
            });
            const isLocal = providerDef.isLocal === true;
            const bucket = isLocal ? this.localProviderKeys : this.keysByProvider;
            if (!bucket.has(key.providerId)) bucket.set(key.providerId, []);
            bucket.get(key.providerId).push({ key, provider });
        }
        console.log(`[ROUTER] loaded ${keys.length} local file key(s)`);
    }

    async loadKeys() {
        if (!this.pool) return;
        try {
            const result = await this.pool.query(`
                SELECT k.*, p.name as provider_name, p.api_format, p.default_endpoint, p.is_cloud
                FROM ai_keys k
                JOIN ai_providers p ON k.provider_id = p.id
                WHERE k.is_active = TRUE
                AND (k.expires_at IS NULL OR k.expires_at > NOW())
            `);

            this.keysByProvider.clear();
            this.localProviderKeys.clear();

            for (const row of result.rows) {
                const key = new AIKey(row);
                const provider = new AIProvider({
                    name: row.provider_name,
                    api_format: row.api_format,
                    default_endpoint: row.default_endpoint,
                    is_cloud: row.is_cloud
                });

                const providerDef = PROVIDER_DEFAULTS[provider.name];
                const isLocalProvider = providerDef?.isLocal || !provider.is_cloud;

                if (isLocalProvider) {
                    if (!this.localProviderKeys.has(key.providerId)) {
                        this.localProviderKeys.set(key.providerId, []);
                    }
                    this.localProviderKeys.get(key.providerId).push({ key, provider });
                } else {
                    if (!this.keysByProvider.has(key.providerId)) {
                        this.keysByProvider.set(key.providerId, []);
                    }
                    this.keysByProvider.get(key.providerId).push({ key, provider });
                }
            }

            const sortFn = (a, b) => b.key.priority - a.key.priority;
            for (const [providerId, keys] of this.keysByProvider) {
                keys.sort(sortFn);
            }
            for (const [providerId, keys] of this.localProviderKeys) {
                keys.sort(sortFn);
            }
        } catch (err) {
            console.error('Failed to load keys:', err.message);
        }
    }

    async startSpareTimeMonitor() {
        setInterval(async () => {
            const http = require('http');
            const https = require('https');
            
            for (const ep of this.localEndpoints) {
                if (!ep || !ep.url) continue;
                const startTime = Date.now();
                try {
                    // Send a micro-prompt to see if the LLM is currently chewing on something
                    const proto = ep.url.startsWith('https') ? https : http;
                    const payload = JSON.stringify({
                        model: ep.model || 'local',
                        messages: [{ role: 'user', content: '.' }],
                        max_tokens: 1,
                        stream: false
                    });
                    
                    const options = {
                        hostname: new URL(ep.url).hostname,
                        port: new URL(ep.url).port || 80,
                        path: new URL(ep.url).pathname,
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(payload) }
                    };

                    const req = proto.request(options, (res) => {
                        let data = '';
                        res.on('data', chunk => data += chunk);
                        res.on('end', () => {
                            if (res.statusCode === 200) {
                                const latency = Date.now() - startTime;
                                // < 400ms = IDLE (Spare time available!)
                                // > 1000ms = BUSY (Owner is using it, protect that time!)
                                //
                                // A DELEGATED endpoint is not a local model: it forwards
                                // the request to remote compute (a KwaaiNet node resolves
                                // a block chain through the libp2p hivemind DHT, so one
                                // answer crosses several machines). Its latency measures
                                // that chain, not local contention on this box, so the
                                // 400ms idle threshold would read every healthy
                                // distributed endpoint as "the owner is using it" and
                                // skip it forever. Delegated endpoints get their own
                                // threshold; a direct local model keeps the strict one.
                                const idleThreshold = ep.delegated ? 3000 : 400;
                                this.spareTimeTracker.set(ep.url, { idle: latency < idleThreshold, latency: latency, ts: Date.now() });
                            } else {
                                this.spareTimeTracker.delete(ep.url);
                            }
                        });
                    });
                    req.on('error', () => { this.spareTimeTracker.delete(ep.url); });
                    req.setTimeout(2000, () => { req.destroy(); });
                    req.write(payload);
                    req.end();
                } catch(e) {
                    this.spareTimeTracker.delete(ep.url);
                }
            }
        }, 10000); // Check every 10 seconds
    }

    async discoverLocalEndpoints() {
        const discoveryEndpoints = [
            // Standard local LLM ports — auto-discovered, no hardcoded IPs
            { url: 'http://localhost:1234/v1/models', name: 'lmstudio', provider: 'lmstudio' },
            { url: 'http://localhost:11434/api/tags', name: 'ollama', provider: 'ollama' },
            { url: 'http://localhost:8080/v1/models', name: 'llamacpp', provider: 'llamacpp' },
            { url: 'http://localhost:8000/v1/models', name: 'localai', provider: 'localai' },
            { url: 'http://localhost:5000/v1/models', name: 'textgen', provider: 'textgen' },
            { url: 'http://localhost:8000/v1/models', name: 'vllm', provider: 'vllm' },
            { url: 'http://localhost:8080/v1/models', name: 'sglang', provider: 'sglang' },
            { url: 'http://localhost:8000/v1/models', name: 'aphrodite', provider: 'aphrodite' },
            { url: 'http://localhost:5000/v1/models', name: 'koboldcpp', provider: 'koboldcpp' },
            { url: 'http://localhost:5000/v1/models', name: 'oobabooga', provider: 'oobabooga' }
        ];

        // Also discover endpoints from env var (non-local machines on the network)
        const envEndpoints = process.env.HIVV_LOCAL_ENDPOINTS || process.env.LOCAL_AI_ENDPOINTS || '';
        if (envEndpoints) {
            for (const url of envEndpoints.split(',').map(s => s.trim()).filter(Boolean)) {
                const probeUrl = url.split('?')[0].includes('/chat/completions')
                    ? url.split('?')[0].replace('/chat/completions', '/models')
                    : url.split('?')[0];
                const existing = discoveryEndpoints.find(e => e.url === url || e.probeUrl === probeUrl);
                if (!existing) {
                    // A '?delegated=1' suffix marks an endpoint that forwards to
                    // remote compute rather than serving a local model. It rides
                    // on the URL because HIVV_LOCAL_ENDPOINTS is a plain comma
                    // list of URLs and we do not want to invent a second
                    // configuration file for one flag.
                    const delegated = new URL(url).searchParams.get('delegated') === '1';
                    const clean = url.split('?')[0];
                    discoveryEndpoints.push({
                        probeUrl,
                        url: clean.includes('/models') ? clean.replace('/models', '/chat/completions') : clean,
                        name: `remote-${new URL(clean).hostname}`,
                        provider: 'ollama',
                        delegated
                    });
                }
            }
        }

        const timeout = 3000;
        // First pass: deduplicate by probe URL so we only probe each port once.
        const urlToProvider = new Map();
        for (const ep of discoveryEndpoints) {
            const probe = ep.probeUrl || ep.url;
            if (!urlToProvider.has(probe)) {
                urlToProvider.set(probe, ep);
            }
        }

        const liveUrls = new Set();
        const modelByProbe = new Map();
        for (const [probeUrl] of urlToProvider) {
            try {
                const controller = new AbortController();
                const timer = setTimeout(() => controller.abort(), timeout);
                const response = await fetch(probeUrl, { method: 'GET', signal: controller.signal });
                clearTimeout(timer);
                if (response.ok) {
                    liveUrls.add(probeUrl);
                    // Grab the real model id while we're here — routing and
                    // capability signals NEED it: a placeholder/absent model
                    // fails the tool-capable allowlist, which silently hid
                    // hub-local endpoints from every OpenCode (tools) request.
                    try {
                        const j = await response.json();
                        const id = (Array.isArray(j?.data) && j.data[0]?.id)
                                || (Array.isArray(j?.models) && (j.models[0]?.name || j.models[0]?.model))
                                || '';
                        if (id) modelByProbe.set(probeUrl, String(id));
                    } catch (_e) { /* non-JSON probe body — fine */ }
                }
            } catch (err) {}
        }

        // Second pass: register every provider that maps to a live URL.
        const discovered = discoveryEndpoints.filter(ep => liveUrls.has(ep.probeUrl || ep.url));

        // Third pass: dedup registration by stream URL.
        const seenUrls = new Set();
        for (const ep of discovered) {
            // Normalize to the CHAT endpoint first: consumers of getAllEndpoints
            // (dial, dry-run, relay) use t.url verbatim — tryLocalAI used to
            // paper over models URLs with an on-the-fly replace.
            if (ep.url.includes('/v1/models')) {
                ep.url = ep.url.replace('/v1/models', '/v1/chat/completions');
            }
            if (seenUrls.has(ep.url)) { continue; }
            seenUrls.add(ep.url);
            
            // SMART DONATION DEFAULT: Tag every new node so it enters the spare-time tracker.
            // By default, we want nodes to donate their SPARE TIME. 
            // If they are busy (>1s response to ping), they won't be donated.
            if (!ep.allowDonation && ep.url.includes('://')) {
                ep.allowDonation = true;
            }

            console.log(`Discovered local AI: ${ep.name} at ${ep.url}`);
            if (!ep.model) {
                ep.model = modelByProbe.get(ep.probeUrl || ep.url) || ep.model;
            }
            const existing = this.localEndpoints.find(e => e.url === ep.url);
            if (!existing) {
                this.localEndpoints.push(ep);
            }
        }
    }

    async loadConfig() {
        if (!this.pool) return;
        try {
            const result = await this.pool.query('SELECT key, value FROM gateway_config');
            for (const row of result.rows) {
                // pg already parses JSONB → JS. Only JSON.parse if value is a JSON-encoded string.
                // (e.g. `'"v2"'::jsonb` round-trips to JS string `"v2"` which parse() handles;
                //  `'v2'::jsonb` is rejected by postgres, so the only string we ever see is the
                //  already-decoded form like `v2`.)
                let value = row.value;
                if (typeof value === 'string') {
                    try { value = JSON.parse(value); } catch { /* already a parsed literal */ }
                }
                switch (row.key) {
                case 'load_balancing_strategy':
                    this.loadBalancingStrategy = (value && value.strategy) || 'priority';
                    break;
                case 'failover_enabled':
                    this.failoverEnabled = !value || value.enabled !== false;
                    break;
                case 'max_retries':
                    this.maxRetries = (value && value.retries) || 3;
                    break;
                case 'routing':
                    if (value && typeof value === 'object') {
                        this._routeCfg = { ...this._routeCfg, ...value };
                    }
                    break;
                case 'relay':
                    // B3: relay quotas live in gateway_config, not in code
                    if (value && typeof value === 'object') {
                        this._relayCfg = { ...(this._relayCfg || {}), ...value };
                        this.relay?.setConfig?.(this._relayCfg);
                    }
                    break;
                }
            }
        } catch (err) {
            console.error('Failed to load config:', err.message);
        }
    }

    selectKey(providerId, preferredModel = null, userId = null) {
        const keys = this.keysByProvider.get(providerId);
        if (!keys || keys.length === 0) {return null;}

        let candidates = keys.filter(k => {
            if (!k.key.isAvailable()) {return false;}
            if (k.key.userId && k.key.userId !== userId && !k.key.isShared) {return false;}
            if (preferredModel && k.key.model !== '*' && k.key.model !== preferredModel) {
                return false;
            }
            return true;
        });

        if (candidates.length === 0) {
            candidates = keys.filter(k => k.key.isAvailable());
        }

        if (candidates.length === 0) {return null;}

        switch (this.loadBalancingStrategy) {
        case 'round_robin':
            candidates.sort((a, b) => {
                const aTime = a.key.lastUsedAt ? new Date(a.key.lastUsedAt).getTime() : 0;
                const bTime = b.key.lastUsedAt ? new Date(b.key.lastUsedAt).getTime() : 0;
                return aTime - bTime;
            });
            break;

        case 'weighted':
            candidates.sort((a, b) => b.key.weight - a.key.weight);
            break;

        case 'priority':
        default:
            candidates.sort((a, b) => b.key.priority - a.key.priority);
            break;
        }

        return candidates[0];
    }

    async routeToProvider(providerName, model, messages, streamToRes, options = {}) {
        const provider = this.providers.get(providerName);
        if (!provider) {
            return { ok: false, error: `Unknown provider: ${providerName}` };
        }

        const keyPair = this.selectKey(provider.id, model, options.userId);
        if (!keyPair) {
            return { ok: false, error: `No available keys for provider: ${providerName}` };
        }

        try {
            return await this.circuitBreaker.execute(providerName, async () => {
                const client = new ProviderClient(keyPair.key, keyPair.provider);
                return await client.makeRequest(model, messages, streamToRes, options);
            });
        } catch (err) {
            return { ok: false, error: err.message, circuitBreakerOpen: this.circuitBreaker.isOpen(providerName) };
        }
    }

    // Get all available LLM endpoints (env var + registered nodes)
    getAllEndpoints() {
        const endpoints = [...this.localEndpoints];
        // This box can appear TWICE: env/discovery localhost entries AND as a
        // registered node (lan_ip:port) — the same physical llama-server split
        // across two stick-identities (warm affinity never latches). Keep the
        // localhost form; drop node duplicates that point at THIS machine.
        const localPorts = new Set();
        for (const e of this.localEndpoints) {
            try {
                const u = new URL(e.url);
                if (u.hostname === 'localhost' || u.hostname === '127.0.0.1' || u.hostname === '::1') {
                    localPorts.add(u.port || '80');
                }
            } catch (_e) { /* unparsable discovery entry */ }
        }
        let myIps = new Set();
        try {
            const ifaces = require('os').networkInterfaces();
            for (const list of Object.values(ifaces)) {
                for (const i of (list || [])) myIps.add(i.address);
            }
        } catch (_e) { /* no interfaces? keep both entries */ }
        // Add LLMs from registered nodes
        if (this.orchestrator) {
            const now = Date.now();
            for (const [nodeId, node] of this.orchestrator.nodes) {
                if (node.status !== 'online') continue;
                const hb_age = now - (node.last_heartbeat || 0);
                if (hb_age > 180000) continue; // stale (3+ min)
                for (const m of (node.local_models || [])) {
                    if (m.locked) continue; // keyed LLM (api-key) — visible in
                    // `hivv nodes` but unusable: peers can't auth either
                    const ip = node.lan_ip || node.ip;
                    if (!ip) continue;
                    const port = m.port || 8080;
                    if (myIps.has(ip) && localPorts.has(String(port))) continue; // = a localhost entry above
                    const url = `http://${ip}:${port}/v1/chat/completions`;
                    // Don't duplicate if already in localEndpoints
                    if (!endpoints.find(e => e.url === url)) {
                        endpoints.push({
                            url,
                            name: `${node.name || nodeId}:${port}`,
                            provider: m.provider || 'local',
                            nodeId,
                            model: m.model
                        });
                    }
                }
            }
        }
        return endpoints;
    }

    async tryLocalAI(messages, streamToRes, options = {}) {
        const timeout = options.timeout || 15000;
        const preferredName = options.preferredName;
        const stream = options.stream !== false;
        let targets = this.getAllEndpoints();
        if (preferredName) {
            const pLower = preferredName.toLowerCase();
            const matched = this.localEndpoints.filter(l => 
                (l.name && l.name.toLowerCase() === pLower) || 
                (l.provider && l.provider.toLowerCase() === pLower)
            );
            if (matched.length > 0) targets = matched;
        }

        // MODEL PINNING for local bees. Every bee serves exactly one model, so
        // the requested `model` used to be ignored for local endpoints — a
        // client asking for "gemma-4-E2B" could still be handed the 0.5B model
        // on the next bee, which answers long prompts with a safety refusal
        // ("I'm sorry, but I can't assist with that"). If the requested model
        // matches at least one advertised local model, restrict the candidate
        // set to those matches. No match → keep every candidate (unchanged
        // behaviour for "auto", "gpt-4o", unknown names, ...).
        const preferredModel = options.preferredModel;
        if (preferredModel && !preferredName) {
            const want = String(preferredModel).toLowerCase();
            if (want && want !== 'auto') {
                const matched = targets.filter(t => t.model && String(t.model).toLowerCase().includes(want));
                if (matched.length > 0) {
                    targets = matched;
                    console.log(`[ROUTE] model pin "${preferredModel}" → ${matched.length} candidate(s): ${matched.map(m => `${m.name}:${m.model}`).join(', ')}`);
                }
            }
        }

        // HIVV OS §2: pluggable routing modes. 'balanced' keeps the measured
        // tokens/sec epsilon-greedy behavior (HIVV_PLAN #7); quality/fast/cost/
        // trusted/private re-score the same candidates. Unknown/slow bees still
        // get exploration jitter so everyone stays warm.
        const mode = (options.route && options.route.mode) || 'balanced';
        if (!preferredName) {
            // rough prompt size (tokens ≈ chars/4) — big prompts care about
            // PROMPT speed (cold 5k-token OpenCode system prompts = 15-45s),
            // not generation tps. Feeds _scoreTargets' big-prompt formula.
            options.promptEst = Math.round((JSON.stringify(messages) || '').length / 4);
            targets = await this._orderByMode(targets, mode, options);
        }

        for (const local of targets) {
            // SPARE TIME GUARD: Check if the owner is actually using this node.
            // We measure the latency of a model lookup. If < 300ms -> IDLE (Donate it!).
            // If > 1500ms -> BUSY (Reserve time for the owner, don't donate).
            const spareHealth = this.spareTimeTracker?.get(local.url);
            if (spareHealth && !spareHealth.idle && local.allowDonation !== false) {
                console.log(`[SPARE-TIME] Skipping ${local.name}: Owner is using it (${spareHealth.latency}ms). Reserving GPU time.`);
                continue;
            }

            try {
                const controller = new AbortController();
                const timer = setTimeout(() => controller.abort(), timeout);

                const lastUserMessage = messages.filter(m => m.role === 'user').pop();
                const messageText = lastUserMessage?.content || '';

                // If no model specified, try to get available models from the endpoint
                let modelName = local.model;
                if (!modelName && local.url.includes('/v1/')) {
                    try {
                        // Construct the models URL - handle both chat/completions and models endpoints
                        let modelsUrl = local.url;
                        if (modelsUrl.includes('/chat/completions')) {
                            modelsUrl = modelsUrl.replace('/chat/completions', '/models');
                        } else if (modelsUrl.includes('/completions')) {
                            modelsUrl = modelsUrl.replace('/completions', '/models');
                        } else if (!modelsUrl.includes('/models')) {
                            // If URL doesn't have /models, append it
                            modelsUrl = modelsUrl.replace('/v1/', '/v1/models');
                        }
                        const modelsResp = await fetch(modelsUrl, { signal: controller.signal });
                        if (modelsResp.ok) {
                            const modelsData = await modelsResp.json();
                            const models = modelsData.data || modelsData.models || [];
                            if (models.length > 0) {
                                // Sanitize model name — truncate long HuggingFace GGUF paths, use clean ID/name
                                let rawName = models[0].id || models[0].name || 'local';
                                // Truncate HuggingFace cache paths (>100 chars means full file path)
                                if (rawName.length > 100) {
                                    // Extract last part after final / 
                                    const parts = rawName.split('/');
                                    const shortName = parts[parts.length - 1];
                                    modelName = shortName.replace(/\.(gguf|bin|safetensors)$/i, '');
                                } else {
                                    modelName = rawName;
                                }
                                local.model = modelName;
                            }
                        }
                    } catch (e) {
                        // Ignore model fetch errors
                    }
                }

                // SMART TOKEN BUDGET — see _budgetTokens() below for the
                // full explanation. Replaces the old hardcoded 14k/2048 caps
                // that truncated every OpenCode coding response.
                const budget = _budgetTokens(messages, {
                    modelName,
                    requestedMax: options.gen?.max_tokens,
                    contextWindow: options.contextWindow,
                });
                const safeMaxTokens = budget.maxTokens;
                const trimmedMessages = budget.messages;

                // Start the latency clock BEFORE the fetch. For non-streaming
                // responses `await fetch` only resolves once the backend has
                // finished generating, so starting the clock after the fetch
                // would record elapsed≈0 and poison fast-mode routing (a genuinely
                // fast local bee would look like it has unknown latency).
                const startTime = Date.now();
                let response;
                if (local.url.includes('/v1/')) {
                    // Convert models URL to chat completions URL
                    let chatUrl = local.url;
                    if (chatUrl.includes('/models')) {
                        chatUrl = chatUrl.replace('/models', '/chat/completions');
                    }
                    
                    const requestBody = {
                        model: modelName || 'local',
                        messages: trimmedMessages,
                        max_tokens: safeMaxTokens,
                        stream: stream,
                        chat_template_kwargs: { enable_thinking: options.gen?.thinking ? true : false }
                    };
                    if (options.gen?.temperature != null) requestBody.temperature = options.gen.temperature;
                    if (options.gen?.top_p != null) requestBody.top_p = options.gen.top_p;
                    if (options.gen?.frequency_penalty != null) requestBody.frequency_penalty = options.gen.frequency_penalty;
                    if (options.gen?.presence_penalty != null) requestBody.presence_penalty = options.gen.presence_penalty;
                    if (options.gen?.stop) requestBody.stop = options.gen.stop;
                    if (options.tools) { requestBody.tools = options.tools; }
                    if (options.toolChoice != null) { requestBody.tool_choice = options.toolChoice; }

                    response = await fetch(chatUrl, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(requestBody),
                        signal: controller.signal
                    });
                } else {
                    response = await fetch(local.url, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ message: messageText }),
                        signal: controller.signal
                    });
                }

                clearTimeout(timer);

                if (!response.ok) {
                    this.perf?.record({ node: this._routeLabel(local), model: AIRouter._prettyModel(local.model), ok: false });
                    // READ THE BODY BEFORE MOVING ON. A context overflow is
                    // the client's prompt being too big, not the bee being
                    // broken — and it is a DIFFERENT answer than "this
                    // provider is unavailable". Swallowing it (as the old
                    // `continue` did) made every bee fail identically and
                    // produced the misleading "All providers unavailable".
                    let errBody = '';
                    try { errBody = await response.text(); } catch (_e) { /* body already consumed */ }
                    if (isContextOverflow(response.status, errBody)) {
                        console.warn(`[ROUTER] context overflow at ${this._routeLabel(local)}: ${errBody.slice(0, 300)}`);
                        // Tell the truth to the client instead of trying
                        // every other bee — they all have the same limit.
                        streamToRes.statusCode = 413;
                        streamToRes.setHeader('Content-Type', 'application/json');
                        streamToRes.end(JSON.stringify({
                            error: {
                                message: 'Prompt exceeds the model context window',
                                type: 'exceed_context_size_error',
                                detail: errBody.slice(0, 500),
                                hint: 'Shorten the conversation, or raise the model\'s context window.',
                            }
                        }));
                        return { ok: false, error: 'Prompt exceeds the model context window', contextOverflow: true, status: 413 };
                    }
                    continue;
                }

                const routeNode = this._routeLabel(local);
                const routeModel = AIRouter._prettyModel(modelName || local.model || 'local');
                options.onRoute?.({ node: routeNode, model: routeModel, via: 'direct' });

                streamToRes.setHeader('X-AI-Provider', local.model || local.name);
                streamToRes.setHeader('X-Hivv-Answered-By', `${routeNode}/${routeModel} (direct)`);
                if (options.policyReceipt) { try { streamToRes.setHeader('X-Hivv-Policy', options.policyReceipt); } catch (_e) {} }
                streamToRes.setHeader('Content-Type', 'text/event-stream');
                streamToRes.setHeader('Cache-Control', 'no-cache');
                streamToRes.setHeader('Connection', 'keep-alive');
                streamToRes.setHeader('X-Accel-Buffering', 'no'); // Disable nginx buffering

                // Send provider info as first SSE event (so client knows which node is responding).
                // X-Hivv-Plain:1 suppresses custom frames for strict OpenAI clients (OpenCode).
                const providerName = local.name || local.model || 'unknown';
                if (!options.plainStream) {
                    streamToRes.write(`data: {"hivv_provider":"${providerName}"}\n\n`);
                }

                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let totalTokens = 0;
                // Bounded sample of the actual answer text — the quality gate at
                // stream end must never become a memory problem for long outputs.
                let qualityBuf = '';
                let backendTps = 0;
                let promptTps = 0;
                // SSE lines can split across TCP reads — buffer partial lines or the
                // finish_reason chunk arrives as two broken fragments and strict
                // clients (OpenCode) see a stream that "ended without finish_reason".
                let lineBuf = '';

                const emitLine = (line) => {
                    if (!line.trim()) return;
                    let outputLine = line;

                    // If this is an SSE data frame, check & sanitize model name
                    if (line.startsWith('data: ') && !line.includes('[DONE]')) {
                        try {
                            const parsed = JSON.parse(line.slice(6));
                            // Sanitize model name — remove long HuggingFace cache paths
                            if (parsed.model && (parsed.model.length > 50 || parsed.model.includes('/models--'))) {
                                const parts = parsed.model.split('/');
                                parsed.model = parts[parts.length - 1].replace(/\.(gguf|bin|safetensors)$/i, '');
                                if (parsed.model.length > 40) {
                                    parsed.model = parsed.model.slice(-35);
                                }
                            }
                            // Also sanitize model in choices
                            if (parsed.choices) {
                                for (const choice of parsed.choices) {
                                    if (choice.model && (choice.model.length > 60 || choice.model.includes('huggingface') || choice.model.includes('.cache'))) {
                                        const parts = choice.model.split('/');
                                        choice.model = parts[parts.length - 1].replace(/\.(gguf|bin|safetensors)$/i, '');
                                    }
                                }
                            }
                            outputLine = 'data: ' + JSON.stringify(parsed);

                            // Count tokens after sanitization
                            if (parsed.choices?.[0]?.delta?.content) {
                                totalTokens++;
                                if (qualityBuf.length < 4000) qualityBuf += parsed.choices[0].delta.content;
                            }
                            if (parsed.timings) {
                                backendTps = parsed.timings.predicted_per_second || 0;
                                if (parsed.timings.prompt_per_second > 0) promptTps = parsed.timings.prompt_per_second;
                            }

                            // STRIP REASONING: Gemma4 includes "reasoning_content" which explodes VRAM and delays response.
                            // We filter it out so users get fast, direct answers instead of reading thinking logs.
                            // If a chunk ONLY has reasoning (no actual content), skip it entirely.
                            const choice = parsed.choices?.[0];
                            if (choice && !choice.delta?.content && choice.delta?.reasoning_content) {
                                // Skip reasoning-only chunks — they're invisible overhead to the client
                                // UNLESS the user explicitly requested thinking mode
                                if (!(options.gen?.thinking)) {
                                    return;
                                }
                            }

                            // Also sanitize: remove reasoning_content from output even if content exists
                            // UNLESS thinking mode is on
                            if (!(options.gen?.thinking)) {
                                if (choice?.delta?.reasoning_content !== undefined && choice.delta?.reasoning_content === '') {
                                    delete choice.delta.reasoning_content;
                                }
                            }
                        } catch {}
                    } else if (line.startsWith('data: ')) {
                        // [DONE] or other marker — don't forward upstream's [DONE]; the
                        // hub writes its own after the stats frame.
                        if (line.includes('[DONE]')) return;
                    } else {
                        // Non-SSE line — could be a complete JSON response (tool_calls come in non-streaming)
                        try {
                            const parsed = JSON.parse(line);
                            if (parsed.choices?.[0]?.delta?.content) {
                                totalTokens++;
                                if (qualityBuf.length < 4000) qualityBuf += parsed.choices[0].delta.content;
                            }
                            if (parsed.timings) backendTps = parsed.timings.predicted_per_second || 0;
                            // Preserve tool_calls — wrap as SSE frame so parseSSE() can extract them
                            if (parsed.choices?.[0]?.message?.tool_calls) {
                                outputLine = 'data: ' + JSON.stringify(parsed);
                            }
                        } catch {}
                    }
                    // SSE events are separated by a BLANK line — spec-compliant
                    // decoders (OpenCode, OpenAI SDKs) treat the whole stream as one
                    // event otherwise, concatenating every JSON payload.
                    streamToRes.write(outputLine + '\n\n');
                };

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) {break;}
                    const textChunk = lineBuf + decoder.decode(value, { stream: true });
                    const lines = textChunk.split('\n');
                    lineBuf = lines.pop(); // keep the trailing partial line for the next read
                    for (const line of lines) emitLine(line);
                }
                if (lineBuf) emitLine(lineBuf); // flush the final partial line

                // Calculate tokens/sec
                const elapsed = (Date.now() - startTime) / 1000;
                const tokensPerSec = backendTps > 0 ? Math.round(backendTps) : (totalTokens > 0 ? Math.round(totalTokens / elapsed) : 0);

                // measured-routing data (HIVV_PLAN #7) — receipt header was set pre-stream.
                // ok:false on a degenerate answer: the EWMA reliability factor then
                // punishes this bee's score, so a fast garbage model gets routed
                // LESS traffic instead of winning the speed lottery (network guard).
                this.perf?.record({ node: routeNode, model: routeModel, tps: tokensPerSec, latencyMs: Math.round(elapsed * 1000), ok: !looksDegenerate(qualityBuf), promptTps });
                // warm-cache stickiness: same principal back within 3 min → same bee
                this._stickySet(options, `${routeNode}::${AIRouter._prettyModel(local.model || 'local')}`);

                // Send stats as a final SSE event before [DONE]
                if (!options.plainStream) {
                    streamToRes.write(`data: {"hivv_stats":{"tokens_per_sec":${tokensPerSec},"total_tokens":${totalTokens},"elapsed_ms":${Math.round(elapsed * 1000)}}}\n\n`);
                }
                // Standard OpenAI terminator — strict clients (OpenCode) need it
                streamToRes.write('data: [DONE]\n\n');

                streamToRes.end();

                return { ok: true, provider: local.model || local.name, tokensPerSec, totalTokens,
                         servedModel: routeModel,
                         answeredBy: `${routeNode}/${routeModel} (direct)` };
            } catch (err) {
                console.log(`Local ${local.name} (${local.url}) failed: ${err.message}`);
                this.perf?.record({ node: this._routeLabel(local), model: AIRouter._prettyModel(local.model), ok: false });
            }
        }
        return { ok: false };
    }

    async tryLocalProviderKeys(messages, streamToRes, timeout = 15000) {
        for (const [providerId, keys] of this.localProviderKeys) {
            for (const { key, provider } of keys) {
                try {
                    const endpoint = key.apiEndpoint || provider.defaultEndpoint;
                    if (!endpoint) {continue;}

                    const controller = new AbortController();
                    const timer = setTimeout(() => controller.abort(), timeout);

                    const body = JSON.stringify({
                        model: key.model || provider.supportedModels?.[0] || 'local',
                        messages: messages,
                        stream: true
                    });

                    const proto = endpoint.startsWith('https') ? require('https') : require('http');
                    const url = new URL(endpoint);

                    const options = {
                        hostname: url.hostname,
                        port: url.port || (url.protocol === 'https:' ? 443 : 80),
                        path: url.pathname + url.search,
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(body) }
                    };

                    const result = await new Promise((resolve) => {
                        const req = proto.request(options, (res) => {
                            if (res.statusCode !== 200) {
                                resolve({ ok: false, status: res.statusCode });
                                return;
                            }

                            streamToRes.setHeader('X-AI-Provider', provider.name);
                            streamToRes.setHeader('X-AI-Key-Id', String(key.id));
                            streamToRes.setHeader('Content-Type', 'text/event-stream');
                            streamToRes.setHeader('Cache-Control', 'no-cache');
                            streamToRes.setHeader('Connection', 'keep-alive');

                            let _streamTokens = 0;
                            res.on('data', chunk => {
                                streamToRes.write(chunk);
                                const m = chunk.toString().match(/"content":\s*"[^"]+"/g);
                                if (m) {_streamTokens += m.length;}
                            });
                            res.on('end', () => { streamToRes.end(); resolve({ ok: true, totalTokens: _streamTokens }); });
                            res.on('error', e => resolve({ ok: false, error: e.message }));
                        });

                        req.on('error', e => resolve({ ok: false, error: e.message }));
                        req.setTimeout(timeout, () => { req.destroy(); resolve({ ok: false, error: 'timeout' }); });

                        req.write(body);
                        req.end();
                    });

                    clearTimeout(timer);

                    if (result.ok) {
                        return { ok: true, provider: provider.name, keyId: key.id, providerId: provider.id, totalTokens: result.totalTokens || 0 };
                    }
                } catch (err) {
                    console.log(`Local provider ${provider.name} key ${key.id} failed: ${err.message}`);
                }
            }
        }
        return { ok: false };
    }

    async tryCustomEndpoint(apiUrl, messages, streamToRes, options = {}) {
        try {
            const controller = new AbortController();
            const timeoutMs = options.timeout || 45000;
            const timer = setTimeout(() => controller.abort(), timeoutMs);

            const lastUserMessage = Array.isArray(messages)
                ? messages.filter(m => m.role === 'user').pop()
                : null;
            const messageText = lastUserMessage?.content || (typeof messages === 'string' ? messages : '');

            let reqBody;
            if (apiUrl.includes('/chat/stream') || apiUrl.includes(':8888') || !apiUrl.includes('/v1/')) {
                reqBody = JSON.stringify({ message: messageText });
            } else {
                const req = {
                    model: options.preferredModel || options.model || 'local',
                    messages: Array.isArray(messages) ? messages : [{ role: 'user', content: messageText }],
                    stream: true
                };
                if (options.tools) { req.tools = options.tools; }
                if (options.toolChoice != null) { req.tool_choice = options.toolChoice; }
                reqBody = JSON.stringify(req);
            }

            const response = await fetch(apiUrl, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: reqBody,
                signal: controller.signal
            });

            clearTimeout(timer);

            if (!response.ok) {
                let errBody = '';
                try { errBody = await response.text(); } catch (_e) { /* body already consumed */ }
                if (isContextOverflow(response.status, errBody)) {
                    // Same rule as the local-bee path: a too-big prompt is
                    // the caller's problem to fix, and saying so beats a
                    // generic HTTP error.
                    return { ok: false, error: 'Prompt exceeds the model context window', contextOverflow: true, status: 413, detail: errBody.slice(0, 500) };
                }
                return { ok: false, error: `Custom endpoint returned HTTP ${response.status}` };
            }

            streamToRes.setHeader('X-AI-Provider', 'local-custom');
            streamToRes.setHeader('Content-Type', 'text/event-stream');
            streamToRes.setHeader('Cache-Control', 'no-cache');
            streamToRes.setHeader('Connection', 'keep-alive');
            streamToRes.setHeader('X-Accel-Buffering', 'no');

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let totalTokens = 0;
            let startTime = Date.now();

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                const chunk = decoder.decode(value, { stream: true });
                streamToRes.write(chunk);

                const lines = chunk.split('\n');
                for (const line of lines) {
                    if (line.startsWith('data: ') && !line.includes('[DONE]')) {
                        try {
                            const data = JSON.parse(line.slice(6));
                            if (data.choices?.[0]?.delta?.content) {
                                totalTokens++;
                            }
                        } catch {}
                    }
                }
            }

            const elapsed = (Date.now() - startTime) / 1000;
            const tokensPerSec = totalTokens > 0 ? Math.round(totalTokens / Math.max(elapsed, 0.1)) : 0;
            streamToRes.write(`data: {"choices":[{"finish_reason":"stats","index":0,"delta":{},"hivv_stats":{"tokens_per_sec":${tokensPerSec},"total_tokens":${totalTokens},"elapsed_ms":${Math.round(elapsed * 1000)}}}]}\n\n`);
            streamToRes.end();

            return { ok: true, provider: 'local-custom', tokensPerSec, totalTokens };
        } catch (err) {
            return { ok: false, error: err.message };
        }
    }

    // HIVV_PLAN #5 — relay fallback: when no direct endpoint answers, queue an
    // inference job on a volunteer node via its long-poll inbox and stream the
    // result back. Opt-in only: nodes must `hivv volunteer on` to receive work.
    async _tryRelay(messages, streamToRes, options = {}) {
        if (!this.relay || !this.orchestrator || !streamToRes) return { ok: false };
        if (options.stream === false || typeof streamToRes.setHeader !== 'function') return { ok: false };
        if (options.privacy === 'localhost') return { ok: false }; // payload must not leave this box
        const budgetMs = Math.min(options.timeout || 45000, 45000);
        const relayCfg = this.relay.getConfig ? this.relay.getConfig() : {};
        let cands = Array.from(this.orchestrator.nodes.values()).filter(n =>
            n.status === 'online' &&
            (n.volunteer === true || n.volunteer?.on === true) &&
            (n.local_models || []).some(m => !m.locked) &&   // locked (keyed) LLMs can't serve relay jobs
            this.relay.activeForNode(n.node_id) < (relayCfg.maxActivePerNode || 2));
        // Same model pin as the direct path — a relay bee that only serves
        // Qwen2.5-0.5B must not answer a request that asked for gemma-4-E2B.
        const wantModel = options.preferredModel || options.model;
        if (wantModel && String(wantModel).toLowerCase() !== 'auto') {
            const want = String(wantModel).toLowerCase();
            const pinned = cands.filter(n => (n.local_models || []).some(m =>
                m.model && String(m.model).toLowerCase().includes(want)));
            if (pinned.length) cands = pinned;
        }
        if (!cands.length) return { ok: false };
        cands.sort(() => Math.random() - 0.5);

        const start = Date.now();
        for (const node of cands.slice(0, 3)) {
            // Keep EVERY system message (the persona + the HIVE identity) and
            // the last few turns — a plain slice(-8) could drop the system
            // prompt entirely on a long chat, leaving the bee with no persona.
            const sysMsgs = messages.filter(m => m.role === 'system');
            const turnMsgs = messages.filter(m => m.role !== 'system');
            const relayMessages = [...sysMsgs, ...turnMsgs.slice(-6)];
            const created = this.relay.createJob({
                nodeId: node.node_id,
                request: {
                    model: options.model || node.local_models[0]?.name || 'local',
                    messages: relayMessages,
                    max_tokens: options.gen?.max_tokens || (relayCfg.defaultTokens || 700),
                    temperature: options.gen?.temperature,
                    top_p: options.gen?.top_p,
                    ...(options.tools ? { tools: options.tools } : {}),
                    ...(options.toolChoice != null ? { tool_choice: options.toolChoice } : {}),
                }
            });
            if (created.error) continue;
            const jobId = created.job.id;
            console.log(`[RELAY] job ${jobId} queued for ${node.node_id}`);
            while (Date.now() - start < budgetMs) {
                await new Promise(r => setTimeout(r, 800));
                const job = this.relay.getJob(jobId);
                if (!job) break;
                if (job.status === 'done') {
                    const content = String(job.result?.content || '');
                    const toolCalls = Array.isArray(job.result?.tool_calls) ? job.result.tool_calls : null;
                    if (!content && !toolCalls) break;
                    const nodeName = node.name || node.node_id;
                    const model = job.result?.model || 'local';
                    const elapsedMs = Date.now() - start;
                    const approxTokens = Math.max(1, Math.round(content.length / 4));
                    const tps = Math.round(approxTokens / Math.max(0.5, elapsedMs / 1000));
                    const route = `${nodeName}/${model} (relay)`;
                    try {
                        streamToRes.setHeader('X-AI-Provider', route);
                        streamToRes.setHeader('X-Hivv-Answered-By', route);
                        streamToRes.setHeader('Content-Type', 'text/event-stream');
                        streamToRes.setHeader('Cache-Control', 'no-cache');
                        if (content) {
                            const frame = options.plainStream
                                ? { choices: [{ index: 0, delta: { content } }] }
                                : { hivv_provider: route, choices: [{ index: 0, delta: { content } }] };
                            streamToRes.write(`data: ${JSON.stringify(frame)}\n\n`);
                        }
                        // Preserve tool_calls so OpenAI-tool clients (OpenCode) work over relay (A4)
                        if (toolCalls) {
                            streamToRes.write(`data: ${JSON.stringify({ choices: [{ index: 0, delta: { tool_calls: toolCalls }, finish_reason: job.result?.finish_reason || 'tool_calls' }] })}\n\n`);
                        }
                        if (!options.plainStream) {
                            streamToRes.write(`data: {"hivv_stats":{"tokens_per_sec":${tps},"total_tokens":${approxTokens},"elapsed_ms":${elapsedMs},"via":"relay"}}\n\n`);
                        }
                        streamToRes.write('data: [DONE]\n\n');
                        streamToRes.end();
                    } catch (e) {
                        console.log(`[RELAY] stream write failed: ${e.message}`);
                    }
                    this.perf?.record({ node: nodeName, model, tps, latencyMs: elapsedMs, ok: true, via: 'relay' });
                    options.onRoute?.({ node: nodeName, model, via: 'relay' });
                    return { ok: true, provider: route, tokensPerSec: tps, totalTokens: approxTokens, servedModel: model, answeredBy: route };
                }
                if (job.status === 'error' || job.status === 'expired') {
                    console.log(`[RELAY] job ${jobId} ${job.status}: ${job.error || ''}`);
                    this.perf?.record({ node: node.name || node.node_id, model: options.model || 'local', ok: false, via: 'relay' });
                    break;
                }
            }
        }
        return { ok: false };
    }

    async routeWithFailover(messages, options = {}) {
        const preferredProvider = options.preferredProvider || options.provider;
        const preferredModel = options.preferredModel || options.model;
        const userId = options.userId;
        const stream = options.stream !== false;
        const apiUrl = options.apiUrl || options.api_url;

        const streamToRes = options.response;

        // ── hivv OS §3: THE GATE. Every action enters YES / MAYBE / NO here. ──
        // Chat (capability 'reasoning') lands on the cached allow fast-path —
        // policy is a map lookup, not a DB write, so conversation stays free.
        const routeSpec = options.route || {};
        let privacy = routeSpec.privacy || (routeSpec.mode === 'private' ? 'hive' : 'any');
        if (options.privacy && ['localhost', 'hive', 'any'].includes(options.privacy)) privacy = options.privacy;
        options.privacy = privacy;
        options.route = { ...routeSpec, privacy };
        if (this.policy) {
            try {
                const gate = await this.policy.gate({
                    capability: options.capability || 'reasoning',
                    principal: options.principal || (userId ? `user:${userId}` : null),
                    provider: preferredProvider || null,
                    mode: routeSpec.mode || 'balanced',
                    data_class: options.data_class || null,
                    args_summary: options.capability_args || null,
                    reason: options.reason || null,
                    via: options.via || null,
                    node: options.node || null,
                });
                if (gate.verdict === 'deny') {
                    return { ok: false, denied: true, policy: gate, error: `policy refused '${gate.key}' (${gate.rule})` };
                }
                if (gate.verdict === 'ask') {
                    return { ok: false, asked: true, policy: gate, error: gate.memo || 'human approval required' };
                }
                options.policyReceipt = `ALLOW ${gate.rule}`;
                if (gate.grant_id) options.policyReceipt = `ALLOW via grant ${gate.grant_id}`;
            } catch (e) {
                console.warn(`[POLICY] gate error (fail-open for chat): ${e.message}`);
            }
        }

        // If explicit custom API URL provided, route directly to it
        if (apiUrl) {
            const customRes = await this.tryCustomEndpoint(apiUrl, messages, streamToRes, options);
            if (customRes.ok) { return customRes; }
            console.log(`Custom api_url ${apiUrl} failed: ${customRes.error}`);
        }

        const isLocalProvider = (name) => {
            if (!name) {return false;}
            const lower = name.toLowerCase();
            const localProviders = ['local', 'local-1', 'local-2', 'mcp', 'lmstudio', 'ollama', 'llamacpp', 'localai', 'textgen', 'vllm', 'sglang', 'aphrodite', 'koboldcpp', 'oobabooga'];
            return localProviders.includes(lower) || PROVIDER_DEFAULTS[lower]?.isLocal;
        };

        if (!preferredProvider || isLocalProvider(preferredProvider)) {
            const localRes = await this.tryLocalAI(messages, streamToRes, {
                timeout: 15000,
                preferredName: preferredProvider,
                preferredModel: preferredModel,
                stream: options.stream,
                tools: options.tools || null,
                toolChoice: options.toolChoice ?? null,
                plainStream: !!options.plainStream,
                principal: options.principal,
                userId: options.userId,
                onRoute: options.onRoute,
                route: options.route,
                privacy: options.privacy,
                policyReceipt: options.policyReceipt,
                gen: options.gen,
            });
            if (localRes.ok) {return localRes;}

            const localKeyRes = await this.tryLocalProviderKeys(messages, streamToRes);
            if (localKeyRes.ok) {return localKeyRes;}

            // HIVV_PLAN #5: no direct route worked — ask a volunteer bee via the hub
            const relayRes = await this._tryRelay(messages, streamToRes, options);
            if (relayRes.ok) {return relayRes;}
        }

        if (preferredProvider && !isLocalProvider(preferredProvider)) {
            if (privacy !== 'any') {
                return { ok: false, privacyBlocked: true, error: `privacy=${privacy} blocks the cloud provider '${preferredProvider}' — no local/relay bee answered` };
            }
            if (this._overCostBudget(preferredProvider, options.route)) {
                return { ok: false, costBlocked: true, error: `'${preferredProvider}' exceeds the request's max_cost` };
            }
            for (let attempt = 0; attempt < this.maxRetries; attempt++) {
                const res = await this.routeToProvider(preferredProvider, preferredModel, messages, streamToRes, {
                    userId,
                    timeout: options.timeout,
                    tools: options.tools || null,
                    toolChoice: options.toolChoice ?? null,
                    gen: options.gen,
                });
                if (res.ok) {return res;}

                console.log(`Attempt ${attempt + 1} failed for ${preferredProvider}: ${res.error}`);
                // A too-big prompt will be too big for every retry too.
                if (res.contextOverflow) {
                    return { ok: false, error: res.error || 'Prompt exceeds the model context window', contextOverflow: true, status: 413, detail: res.detail };
                }
            }
            return { ok: false, error: `All retries exhausted for ${preferredProvider}` };
        }

        const sortedProviders = Array.from(this.providers.values())
            .filter(p => !isLocalProvider(p.name))
            .filter(p => privacy === 'any')                       // §2: privacy filter is a hard gate, not a weight
            .filter(p => !this._overCostBudget(p.name, options.route)); // §2 COST signal: respect max_cost

        for (const provider of sortedProviders) {
            const keyPair = this.selectKey(provider.id, preferredModel, userId);
            if (!keyPair) {continue;}

            const client = new ProviderClient(keyPair.key, provider);
            for (let attempt = 0; attempt < this.maxRetries; attempt++) {
                // PASS THROUGH for cloud: don't modify any options
                const res = await client.makeRequest(preferredModel, messages, streamToRes, {
                    userId,
                    timeout: options.timeout,
                    passThrough: true,
                    tools: options.tools || null,
                    toolChoice: options.toolChoice ?? null,
                    gen: options.gen,
                });
                if (res.ok) {return { ...res, provider: provider.name };}

                console.log(`Provider ${provider.name} attempt ${attempt + 1} failed: ${res.error}`);
                // A context overflow is the CLIENT's prompt being too big.
                // Every other provider has the same or a smaller limit, so
                // trying them all is pointless — and the honest answer is
                // NOT "All providers unavailable" (which says the hive is
                // down). Stop and report the truth.
                if (res.contextOverflow) {
                    return { ok: false, error: res.error || 'Prompt exceeds the model context window', contextOverflow: true, status: 413, detail: res.detail };
                }
            }
        }

        return { ok: false, error: 'All providers unavailable' };
    }

    async chatCompletion(messages, options = {}) {
        const { model, stream = true, provider, userId, apiUrl, api_url, tools, toolChoice } = options;

        const capture = new StreamCapture();
        const rOpts = {
            preferredProvider: provider,
            preferredModel: model,
            userId,
            apiUrl: apiUrl || api_url,
            stream,
            tools,
            toolChoice,
            capability: options.capability,
            route: options.route,
            principal: options.principal,
            data_class: options.data_class,
            capability_args: options.capability_args,
            reason: options.reason,
            via: options.via,
            // Generation params (max_tokens / temperature / top_p …) parsed by
            // the route handler. They were dropped here, so a client's
            // max_tokens never reached a local bee — the model then generated
            // up to 2048 tokens (~60s on a 2B model) for every chat turn.
            gen: options.gen,
            response: capture,
            timeout: options.timeout || 120000
        };
        const res = await this.routeWithFailover(messages, rOpts);

        if (!res.ok) {return res;}

        // routeWithFailover stamps rOpts.policyReceipt (the gate's ALLOW rule) —
        // surface it so the non-streaming path can set X-Hivv-Policy too.
        const policyReceipt = rOpts.policyReceipt || null;

        if (!stream) {
            const parsed = capture.parseSSE();
            const hasToolCalls = parsed.toolCalls && parsed.toolCalls.length > 0;
            return {
                ok: true,
                provider: res.provider,
                answeredBy: res.answeredBy || null,
                servedModel: res.servedModel || null,
                // The gate's receipt must reach the caller even on the
                // non-streaming path — a GUI tool (or any OpenAI client) uses
                // X-Hivv-Policy to show WHICH rule allowed the request.
                policyReceipt: policyReceipt,
                tokensPerSec: res.tokensPerSec || 0,
                data: {
                    id: `chatcmpl-${  Date.now()}`,
                    object: 'chat.completion',
                    created: Math.floor(Date.now() / 1000),
                    model: model || res.provider || 'unknown',
                    choices: [{
                        index: 0,
                        message: {
                            role: 'assistant',
                            content: parsed.content,
                            ...(hasToolCalls ? { tool_calls: parsed.toolCalls } : {})
                        },
                        finish_reason: hasToolCalls ? 'tool_calls' : 'stop'
                    }],
                    usage: parsed.usage || { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
                }
            };
        }

        return { ok: true, provider: res.provider, servedModel: res.servedModel || null,
                 answeredBy: res.answeredBy || null,
                 policyReceipt: policyReceipt,
                 totalTokens: res.totalTokens || 0, tokensPerSec: res.tokensPerSec || 0 };
    }
}

module.exports = AIRouter;
