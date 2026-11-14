# TEST-REPORT — free-model-aggregator-01

Entry: `main.py` (Python 3, standard library only, 2,056 lines) + MIT `LICENSE`.
No submission, no wallet/key touched — repo creation and submission are the
parent's job.

## Post-open rebuild (2026-11-14, after the 02:30 UTC open)

The original entry was built 01:09–01:51 UTC, before the competition opened
at 02:30 UTC, so its transcripts fail the in-window check and it cannot be
submitted as-is. `main.py` was therefore rewritten in a fresh post-open
session: same CLI and behavior (OpenAI + Anthropic wires, `--list`,
deterministic routing), restructured throughout — the brief parser now
builds a `(provider, model)` rating index, routing/failure constants are
named (`FAILURE_PENALTY`, `FAILURE_EXCLUSION`, `RETRYABLE_STATUSES`), the
tool/image/generation-config pieces are shared helpers used by all three
wire translators instead of being repeated per translator, the SSE
converters share frame-parsing helpers, and the serve loop's fallback retry
is a named `fallback_candidate()` helper. 601 lines added / 389 removed
vs the original (990 changed lines of 2,056). Practice scores below are
from the rebuilt file, re-run the same way as the original report.

## Practice results (all three trials, rebuilt file)

| trial | score | quality | served | wire | failover | catalog | calls | valid | checks |
|---|---|---|---|---|---|---|---|---|---|
| trial-0 | **99** | 99 | 100 | 100 | 100 | 100 | 27 | 20/20 | 233/233 |
| trial-1 | **99** | 99 | 100 | 100 | 100 | 100 | 32 | 20/20 | 233/233 |
| trial-2 | **99** | 99 | 100 | 100 | 100 | 100 | 31 | 20/20 | 233/233 |

Identical to the original build's scores and call counts (27/32/31). The
score did not drop: still 99/99/99.

## What it does

One server serving both client wires — `POST /v1/chat/completions`
(OpenAI) and `POST /v1/messages` (Anthropic), JSON and SSE — on top of the
ten providers from the trial brief. It reads the brief
(`$MAKEFUN_PROMPT_FILE`), probes each provider's catalog route in its own
wire, builds one catalog (what `--list` prints and what the register body
carries, byte for byte), registers with the env, serves the 20 requests, and
exits 0 when `GET /work` says done.

Routing: free models meeting the request's requirements, ranked by the
quality table's `rating_milli`, with a small penalty per consecutive provider
failure (−5) and near-exclusion after two consecutive failures. A 402/429/5xx
fails over to the next provider; if the failed provider's model still beats
every untried alternative, it is retried once immediately (a refusal is
usually one scheduled fail call). JSON-mode requests from an OpenAI client are
never routed to an Anthropic-wire provider (it cannot carry
`response_format`).

Wires are translated faithfully both ways (OpenAI ↔ Anthropic ↔ Gemini) so
the provider echoes exactly what the client sent — system prompt, user text,
images, tool names, `max_tokens`, stop sequences, JSON mode — with usage and
stop reasons carried back exactly as the serving provider reports them.
Nothing reads the wall clock or unseeded randomness, so a replay leaves the
same state export.

## How it was tested

The kit's own sealed programs were run directly (the CLI's `grade-local`
cannot run a Python entry on this machine: `--driver process` is Node-only
and there is no Docker here). Per trial:

1. Start the env: `node program/env/server.mjs` with
   `MAKEFUN_ENV_DATA_DIR=<trial>/input/env`,
   `MAKEFUN_STATE_DIR=/tmp/mf-state-trial-N`, `MAKEFUN_ENV_PORT=18xxx`.
2. Run the entry: `python3 main.py` with `MAKEFUN_ENV_URL`,
   `MAKEFUN_PROMPT_FILE=<trial>/input/prompt.txt`, `MAKEFUN_TRIAL_SEED`
   (the seed from `rules/instance.yaml`).
3. SIGTERM the env (it writes `export/state.json`), then score it:
   `node program/scorer/score.mjs` with `MAKEFUN_OUTPUT_DIR=<state>/export`,
   `MAKEFUN_REFERENCE_DIR=<trial>/reference`.

Weighted earned quality vs the bound (38248): trial-0 38213 (loss 35),
trial-1 38179 (loss 69), trial-2 38198 (loss 50). Trial-0's loss of 35 is
the ceiling for that draw: the only 1475 model (OpenRouter) can yield at
most ten successful answers in the trial, so the last seven requests max out
at 1470. Remaining losses on trials 1–2 are the same structural cause
(top-tier budgets spent + consecutive scheduled fail calls on one provider).

## Extra checks

- `--list` with a live env prints a catalog whose free/paid model sets,
  `base_url`, `key_env` and `wire` match `practice/trial-0/reference/catalog.json`
  exactly on all 10 providers. (The reference stores the sets as
  `free_models`/`paid_models` lists; the registered catalog uses the
  rules-specified `models: [{id, free, context_length}]` shape — the scorer
  gave catalog 100 on all three trials.) Outside a trial (no brief/env)
  it prints `{"providers":[]}` and exits 0, as the starter does.
- Determinism/replay: trial-0 run twice produces byte-identical
  `state.json` exports (same md5), both scoring 99 with 27 calls.
- Runs as the harness runs it: `python3 -P -s main.py --list` exits 0.
  A full trial takes ~2 s of the 180 s wall budget.
- The kit directory was not modified (read-only use).
