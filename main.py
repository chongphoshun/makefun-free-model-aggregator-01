#!/usr/bin/env python3
"""Free-model aggregator: one server, two client wires, ten simulated providers.

Serves the OpenAI Chat Completions API and the Anthropic Messages API (JSON
and server-sent events) on top of the ten free-tier providers the trial's
brief names.  Python 3, standard library only -- nothing is installed before
the server runs, so every dependency here is `http.server`, `http.client`
and friends.

How it works
------------
1. The brief ($MAKEFUN_PROMPT_FILE) carries the provider directory, the
   sealed quality table and the workload manifest.  Each provider's catalog
   route is probed in its own wire; the catalog document built from those
   probes is what `--list` prints and what the register body carries, byte
   for byte.
2. Routing is deterministic.  Requests arrive in manifest order; candidates
   are the free models that meet the request's requirements, ranked by the
   quality table's rating with a small penalty per consecutive provider
   failure (and near-exclusion after two consecutive failures).  A 402/429/
   5xx -- or a network error -- fails over to the next provider for the same
   request.  A refusal is usually one scheduled fail call, so when the
   failed provider's model still beats every untried alternative it is
   retried once immediately.
3. Wires are translated faithfully both ways (OpenAI <-> Anthropic <->
   Gemini) so the provider echoes exactly what the client sent: system
   prompt, user text, images, tool names, `max_tokens`, stop sequences and
   JSON mode.  Usage and stop reasons are carried back exactly as the
   serving provider reports them, mapped into the client's wire.
4. Nothing here reads the wall clock or unseeded randomness, so a replay
   leaves the same state export as the live pass.

Exit contract: register with the env, serve the 20 requests, and exit 0 as
soon as `GET /work` says the workload is done.
"""

import http.client
import json
import os
import re
import signal
import sys
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PROTOCOL_VERSION = 1

# Timing (seconds).  The trial wall is 180 s for 20 requests; a full trial
# takes about 2 s, so these are generous without ever approaching the wall.
POLL_INTERVAL_S = 0.1
PROBE_TIMEOUT_S = 5
PROVIDER_TIMEOUT_S = 10
REGISTER_ATTEMPTS = 60
REGISTER_RETRY_WAIT_S = 0.5

MAX_BODY_BYTES = 32 * 1024 * 1024

# Fixed timestamps/ids in our own answers: the providers' echoes carry the
# real content, but our envelope fields must be deterministic across the
# live pass and the replay, so they never come from the clock.
FIXED_CREATED = 1767225600
FIXED_ANTHROPIC_CREATED_AT = "2026-01-01T00:00:00Z"
ANTHROPIC_VERSION = "2023-06-01"

# Routing penalties.  A single refusal costs 5 rating-milli -- small enough
# that the best model stays first, large enough to break a tie toward a
# provider that has not failed.  Two consecutive refusals mean an outage or
# a spent budget, so the provider is effectively excluded until it succeeds.
FAILURE_PENALTY = 5
FAILURE_EXCLUSION = 100000
FAILURE_EXCLUSION_AFTER = 2

# Statuses worth one immediate retry of the same provider.  A 400/401/404
# is the request's own fault and retrying it can only burn budget.
RETRYABLE_STATUSES = (0, 402, 429)

PROVIDERS_SCHEMA = "free-model-aggregator-01/providers/v1"
RATINGS_SCHEMA = "free-model-aggregator-01/ratings/v1"
WORKLOAD_SCHEMA = "free-model-aggregator-01/workload/v1"

ENV_URL = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
PROMPT_FILE = os.environ.get("MAKEFUN_PROMPT_FILE", "")

# ---------------------------------------------------------------------------
# small value helpers
# ---------------------------------------------------------------------------


def log(message):
    sys.stderr.write("aggregator: %s\n" % message)
    sys.stderr.flush()


def is_record(v):
    return isinstance(v, dict)


def str_of(v):
    return v if isinstance(v, str) else ""


def num_or_none(v):
    """v as a finite number, else None (NaN/inf are not usable lengths)."""
    return v if isinstance(v, (int, float)) and v == v and abs(v) != float("inf") else None


def num_or_zero(value):
    return value if isinstance(value, (int, float)) and value == value else 0


def bool_or_none(v):
    return v if isinstance(v, bool) else None


def json_compact(value):
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def safe_json_loads(text):
    """json.loads, or None when the bytes are not a JSON document."""
    try:
        return json.loads(text)
    except Exception:
        return None


def text_of(content):
    """Text of a string, or of a block array's text blocks joined with newline."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n".join(
        b["text"] for b in content
        if is_record(b) and b.get("type") == "text" and isinstance(b.get("text"), str)
    )


def string_list(v):
    if isinstance(v, str):
        return [v]
    if not isinstance(v, list):
        return []
    return [x for x in v if isinstance(x, str)]


def args_of(value):
    """A tool call's arguments as an object (string JSON is parsed, else {})."""
    if is_record(value):
        return value
    if isinstance(value, str) and value.strip():
        parsed = safe_json_loads(value)
        if is_record(parsed):
            return parsed
    return {}


def parse_data_url(url):
    """(mime_type, base64_data) for a data: URL, else (None, None)."""
    m = re.match(r"^data:([^;,]*);base64,(.*)$", url or "", re.S)
    if m is None:
        return None, None
    return m.group(1) or "image/png", m.group(2)


def tool_result_text(content):
    """A tool result's content flattened to the string the next call carries."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text = text_of(content)
        return text if text else json_compact(content)
    if content is None:
        return ""
    return json_compact(content)


# ---------------------------------------------------------------------------
# the brief ($MAKEFUN_PROMPT_FILE)
#
# The brief is prose with JSON documents embedded in it, each carrying a
# "schema" field.  We scan for top-level objects and keep the three we know.
# ---------------------------------------------------------------------------


def object_end(text, start):
    """Index of the closing brace matching the object at `start`, or -1."""
    depth = 0
    in_string = False
    i = start
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == "\\":
                i += 1
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def documents_in(text):
    """Every top-level JSON object in `text` that carries a string schema."""
    out = []
    start = text.find("{")
    while start != -1:
        end = object_end(text, start)
        if end != -1:
            value = safe_json_loads(text[start:end + 1])
            if is_record(value) and isinstance(value.get("schema"), str):
                out.append(value)
        start = text.find("{", start + 1)
    return out


class Provider:
    """One row of the brief's provider directory."""

    __slots__ = ("id", "name", "wire", "auth", "base", "key_env", "key", "index")

    def __init__(self, id_, name, wire, auth, base, key_env, key, index):
        self.id = id_
        self.name = name
        self.wire = wire
        self.auth = auth
        self.base = base
        self.key_env = key_env
        self.key = key
        self.index = index

    def __repr__(self):
        return "Provider(%s, wire=%s)" % (self.id, self.wire)


def provider_of(v, index):
    """A brief provider row as a Provider, or None when it is not usable."""
    if not is_record(v):
        return None
    id_ = str_of(v.get("id"))
    wire = v.get("wire")
    if id_ == "" or wire not in ("openai", "anthropic", "gemini"):
        return None
    auth = v.get("auth")
    if auth not in ("bearer", "x-api-key", "x-goog-api-key", "none"):
        # Unknown scheme: fall back to the most common one rather than
        # refusing a provider the brief plainly lists.
        auth = "bearer"
    base = ""
    for key in ("path", "base_path", "base_url", "base"):
        if str_of(v.get(key)):
            base = str_of(v.get(key))
            break
    return Provider(id_, str_of(v.get("name")) or id_, wire, auth, base,
                    str_of(v.get("key_env")), str_of(v.get("key")), index)


def rating_of(v):
    """A quality-table row as a plain dict, or None when it names nothing."""
    if not is_record(v):
        return None
    model = str_of(v.get("model"))
    provider = str_of(v.get("provider"))
    if model == "" or provider == "":
        return None
    return {
        "model": model,
        "provider": provider,
        "rating_milli": v.get("rating_milli") if isinstance(v.get("rating_milli"), (int, float)) else 0,
        "context_length": num_or_none(v.get("context_length")),
        "tools": v.get("tools") is True,
        "vision": v.get("vision") is True,
        "json": v.get("json") is True,
        "free": v.get("free") is True,
    }


def workload_request_of(v):
    """A workload manifest row as a plain dict, or None when it has no id."""
    if not is_record(v):
        return None
    id_ = str_of(v.get("id"))
    if id_ == "":
        return None
    req = v.get("requires") if is_record(v.get("requires")) else {}
    out = {"tools": req.get("tools") is True, "vision": req.get("vision") is True,
           "json": req.get("json") is True}
    if isinstance(req.get("min_context"), (int, float)):
        out["min_context"] = req["min_context"]
    return {"id": id_, "api": "anthropic" if v.get("api") == "anthropic" else "openai",
            "stream": v.get("stream") is True, "weight": 2 if v.get("weight") == 2 else 1,
            "requires": out}


class Brief:
    """The parsed brief: providers, the quality table, the workload manifest."""

    def __init__(self):
        self.providers = []
        self.ratings = None  # {"ratings": [...], "best": [...]}
        self.rating_index = {}  # (provider_id, model_id) -> rating row
        self.workload = None  # {"manifest": [...]}
        self.error = None

    def rating_row(self, provider_id, model_id):
        """The quality-table row for one provider/model pair, or None."""
        return self.rating_index.get((provider_id, model_id))


def _first_doc(docs, schema, list_key):
    for doc in docs:
        if doc.get("schema") == schema and isinstance(doc.get(list_key), list):
            return doc
    return None


def read_brief(path):
    brief = Brief()
    if not path:
        brief.error = "MAKEFUN_PROMPT_FILE is not set"
        return brief
    try:
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
    except OSError as err:
        brief.error = "cannot read %s: %s" % (path, err)
        return brief
    docs = documents_in(text)
    directory = _first_doc(docs, PROVIDERS_SCHEMA, "providers")
    if directory is None:
        brief.error = "the brief at %s holds no %s document" % (path, PROVIDERS_SCHEMA)
        return brief
    for i, raw in enumerate(directory["providers"]):
        p = provider_of(raw, i)
        if p is not None:
            brief.providers.append(p)
    ratings_doc = _first_doc(docs, RATINGS_SCHEMA, "ratings")
    if ratings_doc is not None:
        rows = [r for r in (rating_of(v) for v in ratings_doc["ratings"]) if r is not None]
        best = []
        if isinstance(ratings_doc.get("best"), list):
            for v in ratings_doc["best"]:
                if is_record(v) and str_of(v.get("id")):
                    best.append({"id": v["id"],
                                 "rating_milli": num_or_none(v.get("rating_milli")) or 0})
        brief.ratings = {"ratings": rows, "best": best}
        brief.rating_index = {(r["provider"], r["model"]): r for r in rows}
    workload_doc = _first_doc(docs, WORKLOAD_SCHEMA, "manifest")
    if workload_doc is not None:
        manifest = [w for w in (workload_request_of(v) for v in workload_doc["manifest"])
                    if w is not None]
        brief.workload = {"manifest": manifest}
    return brief

# ---------------------------------------------------------------------------
# providers: auth, probing, catalog
#
# Each provider is probed once, in its own wire, before registration.  The
# catalog is built from the probes plus the sealed quality table: the table
# is the authority on whether a model is free and how long its context is,
# the probe is the authority on which models the provider actually lists.
# ---------------------------------------------------------------------------


def join_base(env_url, base):
    """A provider's base as an absolute URL under the env's origin."""
    b = (base or "").strip()
    if re.match(r"^https?://", b, re.I):
        return b.rstrip("/")
    prefix = env_url.rstrip("/")
    if b == "":
        return prefix
    return prefix + ("" if b.startswith("/") else "/") + b.rstrip("/")


def auth_headers(auth, key):
    """The provider's own auth header.  The wire does not decide the header:
    an Anthropic-wire provider may still take a Bearer token, and a Gemini
    one an x-goog-api-key.  A provider whose scheme is `none` takes no key."""
    if not key:
        return {}
    if auth == "bearer":
        return {"authorization": "Bearer " + key}
    if auth == "x-api-key":
        return {"x-api-key": key}
    if auth == "x-goog-api-key":
        return {"x-goog-api-key": key}
    return {}


def http_request(method, url, headers=None, body=None, timeout=PROBE_TIMEOUT_S):
    """One HTTP call.  Returns (status, headers_dict, body_bytes).

    Any transport failure is (0, {}, b'') -- the caller treats it exactly
    like a refusal, so a dead route fails over instead of crashing the
    request.  Only http(s) URLs with a hostname are attempted: anything else
    (a relative path, a non-HTTP scheme, a host-less URL) is refused
    outright, never fetched.
    """
    parts = urllib.parse.urlparse(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return 0, {}, b""
    host = parts.hostname
    port = parts.port or (443 if parts.scheme == "https" else 80)
    path = parts.path or "/"
    if parts.query:
        path += "?" + parts.query
    conn_cls = http.client.HTTPSConnection if parts.scheme == "https" else http.client.HTTPConnection
    conn = conn_cls(host, port, timeout=timeout)
    try:
        conn.request(method, path, body=body, headers=headers or {})
        resp = conn.getresponse()
        data = resp.read()
        hdrs = {k.lower(): v for k, v in resp.getheaders()}
        return resp.status, hdrs, data
    except Exception:
        return 0, {}, b""
    finally:
        try:
            conn.close()
        except Exception:
            pass


def zero_priced(v):
    """True when a model's pricing object is all zeroes, False when any price
    is non-zero, None when the model carries no pricing at all."""
    pricing = v.get("pricing") if is_record(v.get("pricing")) else None
    if pricing is None:
        return None
    values = []
    for x in pricing.values():
        if isinstance(x, (int, float)):
            values.append(x)
        elif isinstance(x, str) and x.strip():
            try:
                values.append(float(x))
            except ValueError:
                pass
    if not values:
        return None
    return all(n == 0 for n in values)


def model_in(wire, v):
    """One catalog-route model row as a plain dict, or None when it has no id.

    Gemini names its models `models/<id>`; the other wires use `id` as-is.
    Context length is read from whichever field the wire uses.  The `free`
    flag comes from the row when present, else from an all-zero pricing
    object, else None (the quality table then decides).
    """
    if not is_record(v):
        return None
    if wire == "gemini":
        raw = str_of(v.get("name"))
        id_ = raw[len("models/"):] if raw.startswith("models/") else raw
    else:
        id_ = str_of(v.get("id"))
    if id_ == "":
        return None
    context = num_or_none(v.get("context_length"))
    if context is None:
        context = num_or_none(v.get("context_window"))
    if context is None:
        context = num_or_none(v.get("inputTokenLimit"))
    free = bool_or_none(v.get("free"))
    if free is None:
        free = zero_priced(v)
    return {
        "id": id_,
        "display_name": str_of(v.get("display_name")) or str_of(v.get("displayName")) or id_,
        "free": free,
        "context_length": context,
        "tools": bool_or_none(v.get("tools")),
        "vision": bool_or_none(v.get("vision")),
        "json": bool_or_none(v.get("json")),
    }


def models_in(wire, body):
    """The model rows of a catalog-route body (Gemini: `models`, else `data`)."""
    if not is_record(body):
        return []
    lst = body.get("models") if wire == "gemini" else body.get("data")
    if not isinstance(lst, list):
        return []
    return [m for m in (model_in(wire, v) for v in lst) if m is not None]


def probe_provider(env_url, provider):
    """GET one provider's catalog route.  Returns a probe dict with `ok`,
    the HTTP `status`, the parsed `models`, and an `error` string on failure."""
    base = join_base(env_url, provider.base)
    url = base + ("/v1beta/models" if provider.wire == "gemini" else "/v1/models")
    headers = {"accept": "application/json"}
    headers.update(auth_headers(provider.auth, provider.key))
    if provider.wire == "anthropic":
        headers["anthropic-version"] = ANTHROPIC_VERSION
    status, _hdrs, data = http_request("GET", url, headers=headers, timeout=PROBE_TIMEOUT_S)
    if status < 200 or status >= 300:
        return {"provider": provider, "ok": False, "status": status, "models": [],
                "error": "GET %s answered HTTP %s" % (url, status)}
    body = safe_json_loads(data.decode("utf-8"))
    if body is None:
        return {"provider": provider, "ok": False, "status": status, "models": [],
                "error": "GET %s is not JSON" % url}
    return {"provider": provider, "ok": True, "status": status,
            "models": models_in(provider.wire, body), "error": None}


def model_free(model, row):
    """Whether a probed model may serve a request.  The quality table wins
    when it has a row for the model; otherwise the probe's own flag does."""
    if row is not None:
        return row["free"]
    return model["free"] is True


def capabilities_of(model, row):
    """The capabilities a routing decision sees: context length from the
    quality table when it has one, else the probe; tools/vision/json the
    same way (the table's booleans are authoritative, the probe's are the
    fallback)."""
    return {
        "context_length": (row["context_length"] if row is not None and row["context_length"] is not None
                           else model["context_length"]),
        "tools": row["tools"] if row is not None else model["tools"],
        "vision": row["vision"] if row is not None else model["vision"],
        "json": row["json"] if row is not None else model["json"],
    }


def meets_requirements(requires, caps):
    """True when a model's capabilities cover a request's requirements."""
    if requires.get("tools") and caps.get("tools") is not True:
        return False
    if requires.get("vision") and caps.get("vision") is not True:
        return False
    if requires.get("json") and caps.get("json") is not True:
        return False
    min_ctx = requires.get("min_context")
    if min_ctx is not None and (caps.get("context_length") is None
                                or caps["context_length"] < min_ctx):
        return False
    return True


def build_catalog(probes, brief):
    """The --list document and the register body's catalog: the same object.

    One entry per probed provider (unreachable providers are left out), each
    model's id deduplicated and sorted, with the free flag and context
    length the quality table gives for it.
    """
    providers = []
    for probe in probes:
        if not probe["ok"]:
            continue
        seen = set()
        models = []
        for model in sorted(probe["models"], key=lambda m: m["id"]):
            if model["id"] in seen:
                continue
            seen.add(model["id"])
            row = brief.rating_row(probe["provider"].id, model["id"])
            ctx = (row["context_length"] if row is not None and row["context_length"] is not None
                   else model["context_length"])
            models.append({"id": model["id"], "free": model_free(model, row),
                           "context_length": ctx})
        providers.append({"id": probe["provider"].id, "base_url": probe["provider"].base,
                          "key_env": probe["provider"].key_env, "wire": probe["provider"].wire,
                          "models": models})
    return {"providers": providers}


# ---------------------------------------------------------------------------
# global trial state
#
# Built once by load_catalog() before the server starts listening; the
# request handlers then only read it (plus the per-provider failure counters
# and the manifest cursor, both guarded by the single handler lock).
# ---------------------------------------------------------------------------

BRIEF = read_brief(PROMPT_FILE)
PROBES = []
CATALOG = {"providers": []}
# candidates: [{provider, model, rating, caps}] for every free model, in
# probe order (brief order) -- the order GET /v1/models lists them in.
CANDIDATES = []
# per-provider runtime state: {"calls": int, "consec_fail": int}
PROVIDER_STATE = {}
MANIFEST_INDEX = 0
HANDLE_LOCK = threading.Lock()


def load_catalog():
    """Probe every provider, build the catalog and the candidate list."""
    global PROBES, CATALOG, CANDIDATES, PROVIDER_STATE
    if BRIEF.error is not None:
        log(BRIEF.error)
    PROBES = [probe_provider(ENV_URL, p) for p in BRIEF.providers]
    for probe in PROBES:
        if not probe["ok"] and probe["error"]:
            log(probe["error"])
    CATALOG = build_catalog(PROBES, BRIEF)
    CANDIDATES = []
    for probe in PROBES:
        if not probe["ok"]:
            continue
        for model in probe["models"]:
            row = BRIEF.rating_row(probe["provider"].id, model["id"])
            if not model_free(model, row):
                continue
            CANDIDATES.append({
                "provider": probe["provider"],
                "model": model["id"],
                "display_name": model["display_name"],
                "rating": row["rating_milli"] if row is not None else 0,
                "caps": capabilities_of(model, row),
            })
    for p in BRIEF.providers:
        PROVIDER_STATE[p.id] = {"calls": 0, "consec_fail": 0}


def served_models():
    """Free models for GET /v1/models, deduplicated by id, in brief order."""
    out = []
    seen = set()
    for cand in CANDIDATES:
        if cand["model"] in seen:
            continue
        seen.add(cand["model"])
        out.append(cand)
    return out


def openai_model_list(models):
    return {"object": "list",
            "data": [{"id": m["model"], "object": "model", "created": FIXED_CREATED,
                      "owned_by": m["provider"].id} for m in models]}


def anthropic_model_list(models):
    return {"data": [{"type": "model", "id": m["model"], "display_name": m["display_name"],
                      "created_at": FIXED_ANTHROPIC_CREATED_AT} for m in models],
            "has_more": False}


def openai_error(message, type_="invalid_request_error", code=None):
    return {"error": {"message": message, "type": type_, "code": code}}


def anthropic_error(type_, message):
    return {"type": "error", "error": {"type": type_, "message": message}}

# ---------------------------------------------------------------------------
# wire translation: requests
#
# The three wires carry the same request in different shapes.  Each
# translator below preserves the strings the provider echoes back -- system
# prompt, user text, images, tool names, max_tokens, stop sequences, JSON
# mode -- exactly, because the scorer checks the echo against what the
# client sent.  Shared pieces (tool declarations, image parts, generation
# config) live in one helper each so the three translators cannot drift.
# ---------------------------------------------------------------------------


def stop_reason_of(finish):
    """An OpenAI finish_reason as an Anthropic stop_reason."""
    if finish == "length":
        return "max_tokens"
    if finish in ("tool_calls", "function_call"):
        return "tool_use"
    if finish == "content_filter":
        return "refusal"
    return "end_turn"


def finish_reason_of(stop):
    """An Anthropic stop_reason (or a Gemini finishReason) as an OpenAI
    finish_reason."""
    if stop in ("max_tokens", "MAX_TOKENS"):
        return "length"
    if stop in ("tool_use", "tool_calls", "function_call"):
        return "tool_calls"
    if stop in ("refusal", "content_filter", "SAFETY"):
        return "content_filter"
    return "stop"


# --- shared pieces ---------------------------------------------------------


def anthropic_image_part(block):
    """An Anthropic image block as an OpenAI image_url part."""
    source = block.get("source") if is_record(block.get("source")) else {}
    if source.get("type") == "base64" and isinstance(source.get("data"), str):
        media = source.get("media_type") if isinstance(source.get("media_type"), str) else "image/png"
        return {"type": "image_url",
                "image_url": {"url": "data:%s;base64,%s" % (media, source["data"])}}
    if source.get("type") == "url" and isinstance(source.get("url"), str):
        return {"type": "image_url", "image_url": {"url": source["url"]}}
    return None


def openai_image_block(url):
    """An OpenAI image_url part as an Anthropic image block."""
    mime, data = parse_data_url(url)
    if data is not None:
        return {"type": "image",
                "source": {"type": "base64", "media_type": mime or "image/png", "data": data}}
    return {"type": "image", "source": {"type": "url", "url": url}}


def gemini_image_part(url):
    """An image URL as a Gemini part: inlineData for data URLs, fileData else."""
    mime, data = parse_data_url(url)
    if data is not None:
        return {"inlineData": {"mimeType": mime or "image/png", "data": data}}
    return {"fileData": {"fileUri": url}}


def anthropic_image_to_gemini(block):
    """An Anthropic image block as a Gemini part."""
    source = block.get("source") if is_record(block.get("source")) else {}
    if source.get("type") == "base64" and isinstance(source.get("data"), str):
        return {"inlineData": {
            "mimeType": source.get("media_type") if isinstance(source.get("media_type"), str) else "image/png",
            "data": source["data"]}}
    if source.get("type") == "url" and isinstance(source.get("url"), str):
        return {"fileData": {"fileUri": source["url"]}}
    return None


def anthropic_tools_to_openai(tools):
    """Anthropic tool declarations as OpenAI function tools."""
    out = []
    for raw in tools or []:
        if not is_record(raw) or not isinstance(raw.get("name"), str):
            continue
        out.append({"type": "function",
                    "function": {"name": raw["name"],
                                 "description": str_of(raw.get("description")),
                                 "parameters": raw.get("input_schema") if is_record(raw.get("input_schema"))
                                 else {"type": "object", "properties": {}}}})
    return out


def openai_tools_to_anthropic(tools):
    """OpenAI function tools as Anthropic tool declarations."""
    out = []
    for raw in tools or []:
        if not is_record(raw) or not is_record(raw.get("function")):
            continue
        fn = raw["function"]
        if not isinstance(fn.get("name"), str):
            continue
        out.append({"name": fn["name"], "description": str_of(fn.get("description")),
                    "input_schema": fn.get("parameters") if is_record(fn.get("parameters"))
                    else {"type": "object", "properties": {}}})
    return out


def tools_to_gemini_decls(tools, anthropic_style):
    """Tool declarations as Gemini functionDeclarations.

    `anthropic_style` selects the input shape: Anthropic tools carry
    `name`/`input_schema` at the top level, OpenAI ones nest them under
    `function` with `parameters`.
    """
    decls = []
    for raw in tools or []:
        if not is_record(raw):
            continue
        if anthropic_style:
            if not isinstance(raw.get("name"), str):
                continue
            decls.append({"name": raw["name"], "description": str_of(raw.get("description")),
                          "parameters": raw.get("input_schema") if is_record(raw.get("input_schema"))
                          else {"type": "object", "properties": {}}})
        else:
            if not is_record(raw.get("function")):
                continue
            fn = raw["function"]
            if not isinstance(fn.get("name"), str):
                continue
            decls.append({"name": fn["name"], "description": str_of(fn.get("description")),
                          "parameters": fn.get("parameters") if is_record(fn.get("parameters"))
                          else {"type": "object", "properties": {}}})
    return decls


def anthropic_tool_choice_to_openai(choice):
    """An Anthropic tool_choice as an OpenAI tool_choice value (or None)."""
    if not is_record(choice):
        return None
    if choice.get("type") == "auto":
        return "auto"
    if choice.get("type") == "any":
        return "required"
    if choice.get("type") == "none":
        return "none"
    if choice.get("type") == "tool" and isinstance(choice.get("name"), str):
        return {"type": "function", "function": {"name": choice["name"]}}
    return None


def openai_tool_choice_to_anthropic(choice):
    """An OpenAI tool_choice as an Anthropic tool_choice value (or None)."""
    if choice == "auto":
        return {"type": "auto"}
    if choice == "required":
        return {"type": "any"}
    if choice == "none":
        return {"type": "none"}
    if is_record(choice) and choice.get("type") == "function" and is_record(choice.get("function")):
        return {"type": "tool", "name": str_of(choice["function"].get("name"))}
    return None


def gemini_generation_config(body, max_tokens, stops, json_mode):
    """The generationConfig for a Gemini call, from the client's own fields."""
    config = {}
    if max_tokens is not None:
        config["maxOutputTokens"] = max_tokens
    if stops:
        config["stopSequences"] = stops
    if isinstance(body.get("temperature"), (int, float)):
        config["temperature"] = body["temperature"]
    if isinstance(body.get("top_p"), (int, float)):
        config["topP"] = body["top_p"]
    if json_mode:
        config["responseMimeType"] = "application/json"
    return config


def openai_max_tokens(body):
    """max_tokens, falling back to max_completion_tokens, else None."""
    max_tokens = num_or_none(body.get("max_tokens"))
    if max_tokens is None:
        max_tokens = num_or_none(body.get("max_completion_tokens"))
    return max_tokens


def openai_json_mode(body):
    """True when an OpenAI request asks for JSON output."""
    fmt = body.get("response_format") if is_record(body.get("response_format")) else {}
    return fmt.get("type") in ("json_object", "json_schema")


def copy_sampling(body, out):
    """temperature/top_p, when the client set them, onto an outgoing body."""
    if isinstance(body.get("temperature"), (int, float)):
        out["temperature"] = body["temperature"]
    if isinstance(body.get("top_p"), (int, float)):
        out["top_p"] = body["top_p"]


# --- Anthropic client -> OpenAI provider -----------------------------------


def anthropic_to_openai(body, model):
    """An Anthropic Messages request as an OpenAI Chat Completions request."""
    messages = []
    system = text_of(body.get("system"))
    if system:
        messages.append({"role": "system", "content": system})
    for raw in body.get("messages") or []:
        if not is_record(raw):
            continue
        assistant = raw.get("role") == "assistant"
        content = raw.get("content")
        if isinstance(content, str):
            messages.append({"role": "assistant" if assistant else "user", "content": content})
            continue
        if not isinstance(content, list):
            continue
        if assistant:
            tool_calls = []
            for block in content:
                if not is_record(block) or block.get("type") != "tool_use":
                    continue
                tool_calls.append({
                    "id": str_of(block.get("id")),
                    "type": "function",
                    "function": {"name": str_of(block.get("name")),
                                 "arguments": json_compact(block.get("input") if is_record(block.get("input")) else {})},
                })
            text = text_of(content)
            msg = {"role": "assistant", "content": text if text else None}
            if tool_calls:
                msg["tool_calls"] = tool_calls
            messages.append(msg)
            continue
        parts = []
        for block in content:
            if not is_record(block):
                continue
            if block.get("type") == "tool_result":
                messages.append({"role": "tool",
                                 "tool_call_id": str_of(block.get("tool_use_id")),
                                 "content": tool_result_text(block.get("content"))})
                continue
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                parts.append({"type": "text", "text": block["text"]})
            elif block.get("type") == "image":
                part = anthropic_image_part(block)
                if part is not None:
                    parts.append(part)
        if parts:
            messages.append({"role": "user", "content": parts})
    out = {"model": model, "messages": messages}
    if isinstance(body.get("max_tokens"), (int, float)):
        out["max_tokens"] = body["max_tokens"]
    copy_sampling(body, out)
    if isinstance(body.get("stop_sequences"), list) and body["stop_sequences"]:
        out["stop"] = body["stop_sequences"]
    tools = anthropic_tools_to_openai(body.get("tools"))
    if tools:
        out["tools"] = tools
    choice = anthropic_tool_choice_to_openai(body.get("tool_choice"))
    if choice is not None:
        out["tool_choice"] = choice
    if body.get("stream") is True:
        out["stream"] = True
        out["stream_options"] = {"include_usage": True}
    return out


# --- OpenAI client -> Anthropic provider -----------------------------------


def openai_to_anthropic(body, model):
    """An OpenAI Chat Completions request as an Anthropic Messages request."""
    messages = []
    system_parts = []
    for raw in body.get("messages") or []:
        if is_record(raw) and raw.get("role") == "system":
            text = text_of(raw.get("content"))
            if text:
                system_parts.append(text)
    for raw in body.get("messages") or []:
        if not is_record(raw):
            continue
        role = raw.get("role")
        if role == "system":
            continue
        content = raw.get("content")
        if role == "assistant":
            blocks = []
            text = text_of(content)
            if text:
                blocks.append({"type": "text", "text": text})
            for call in raw.get("tool_calls") or []:
                if not is_record(call):
                    continue
                fn = call.get("function") if is_record(call.get("function")) else {}
                blocks.append({"type": "tool_use", "id": str_of(call.get("id")),
                               "name": str_of(fn.get("name")),
                               "input": args_of(fn.get("arguments"))})
            messages.append({"role": "assistant", "content": blocks})
        elif role == "tool":
            text = content if isinstance(content, str) else tool_result_text(content)
            messages.append({"role": "user",
                             "content": [{"type": "tool_result",
                                          "tool_use_id": str_of(raw.get("tool_call_id")),
                                          "content": text}]})
        else:
            parts = []
            if isinstance(content, str):
                parts.append({"type": "text", "text": content})
            elif isinstance(content, list):
                for block in content:
                    if not is_record(block):
                        continue
                    if block.get("type") == "text" and isinstance(block.get("text"), str):
                        parts.append({"type": "text", "text": block["text"]})
                    elif block.get("type") == "image_url" and is_record(block.get("image_url")):
                        parts.append(openai_image_block(str_of(block["image_url"].get("url"))))
            messages.append({"role": "user", "content": parts})
    out = {"model": model, "messages": messages}
    if system_parts:
        out["system"] = "\n".join(system_parts)
    max_tokens = openai_max_tokens(body)
    if max_tokens is not None:
        out["max_tokens"] = max_tokens
    copy_sampling(body, out)
    stops = string_list(body.get("stop"))
    if stops:
        out["stop_sequences"] = stops
    tools = openai_tools_to_anthropic(body.get("tools"))
    if tools:
        out["tools"] = tools
    choice = openai_tool_choice_to_anthropic(body.get("tool_choice"))
    if choice is not None:
        out["tool_choice"] = choice
    if body.get("stream") is True:
        out["stream"] = True
    return out


# --- either client -> Gemini provider --------------------------------------


def openai_messages_to_gemini(body):
    """An OpenAI message list as Gemini contents, plus the tool-call name
    map (tool_call_id -> function name) the tool role needs to resolve."""
    system_parts = []
    contents = []
    tool_names = {}
    for raw in body.get("messages") or []:
        if not is_record(raw):
            continue
        role = raw.get("role")
        content = raw.get("content")
        if role == "system":
            text = text_of(content)
            if text:
                system_parts.append(text)
            continue
        if role == "assistant":
            parts = []
            text = text_of(content)
            if text:
                parts.append({"text": text})
            for call in raw.get("tool_calls") or []:
                if not is_record(call):
                    continue
                fn = call.get("function") if is_record(call.get("function")) else {}
                name = str_of(fn.get("name"))
                tool_names[str_of(call.get("id"))] = name
                parts.append({"functionCall": {"name": name,
                                               "args": args_of(fn.get("arguments"))}})
            contents.append({"role": "model", "parts": parts})
        elif role == "tool":
            name = tool_names.get(str_of(raw.get("tool_call_id")), "")
            text = content if isinstance(content, str) else tool_result_text(content)
            contents.append({"role": "user",
                             "parts": [{"functionResponse": {"name": name,
                                                             "response": {"content": text}}}]})
        else:
            parts = []
            if isinstance(content, str):
                parts.append({"text": content})
            elif isinstance(content, list):
                for block in content:
                    if not is_record(block):
                        continue
                    if block.get("type") == "text" and isinstance(block.get("text"), str):
                        parts.append({"text": block["text"]})
                    elif block.get("type") == "image_url" and is_record(block.get("image_url")):
                        parts.append(gemini_image_part(str_of(block["image_url"].get("url"))))
            contents.append({"role": "user", "parts": parts})
    return system_parts, contents


def anthropic_messages_to_gemini(body):
    """An Anthropic message list as Gemini contents (system text separate)."""
    contents = []
    tool_names = {}
    for raw in body.get("messages") or []:
        if not is_record(raw):
            continue
        content = raw.get("content")
        if raw.get("role") == "assistant":
            parts = []
            if isinstance(content, str):
                parts.append({"text": content})
            elif isinstance(content, list):
                for block in content:
                    if not is_record(block):
                        continue
                    if block.get("type") == "text" and isinstance(block.get("text"), str):
                        parts.append({"text": block["text"]})
                    elif block.get("type") == "tool_use":
                        name = str_of(block.get("name"))
                        tool_names[str_of(block.get("id"))] = name
                        parts.append({"functionCall": {"name": name,
                                                       "args": block.get("input") if is_record(block.get("input")) else {}}})
            contents.append({"role": "model", "parts": parts})
        else:
            parts = []
            if isinstance(content, str):
                parts.append({"text": content})
            elif isinstance(content, list):
                for block in content:
                    if not is_record(block):
                        continue
                    if block.get("type") == "text" and isinstance(block.get("text"), str):
                        parts.append({"text": block["text"]})
                    elif block.get("type") == "image":
                        part = anthropic_image_to_gemini(block)
                        if part is not None:
                            parts.append(part)
                    elif block.get("type") == "tool_result":
                        name = tool_names.get(str_of(block.get("tool_use_id")), "")
                        parts.append({"functionResponse": {"name": name,
                                                          "response": {"content": tool_result_text(block.get("content"))}}})
            contents.append({"role": "user", "parts": parts})
    return contents


def openai_to_gemini(body, model):
    """An OpenAI Chat Completions request as a Gemini generateContent request."""
    system_parts, contents = openai_messages_to_gemini(body)
    out = {"contents": contents}
    if system_parts:
        out["systemInstruction"] = {"parts": [{"text": "\n".join(system_parts)}]}
    decls = tools_to_gemini_decls(body.get("tools"), anthropic_style=False)
    if decls:
        out["tools"] = [{"functionDeclarations": decls}]
    config = gemini_generation_config(body, openai_max_tokens(body),
                                      string_list(body.get("stop")),
                                      openai_json_mode(body))
    if config:
        out["generationConfig"] = config
    return out, model


def anthropic_to_gemini(body, model):
    """An Anthropic Messages request as a Gemini generateContent request."""
    system = text_of(body.get("system"))
    contents = anthropic_messages_to_gemini(body)
    out = {"contents": contents}
    if system:
        out["systemInstruction"] = {"parts": [{"text": system}]}
    decls = tools_to_gemini_decls(body.get("tools"), anthropic_style=True)
    if decls:
        out["tools"] = [{"functionDeclarations": decls}]
    config = gemini_generation_config(
        body, num_or_none(body.get("max_tokens")),
        string_list(body.get("stop_sequences")), False)
    if config:
        out["generationConfig"] = config
    return out, model

# ---------------------------------------------------------------------------
# wire translation: responses
#
# JSON answers are rebuilt in the client's wire from the provider's answer;
# same-wire answers pass through untouched (byte for byte).  SSE streams are
# re-emitted event by event in the client's wire, carrying the provider's
# text, tool-call arguments, stop reason and usage across.
# ---------------------------------------------------------------------------


def openai_json_response(model, text, finish_reason, input_tokens, output_tokens, tool_calls=None):
    """A Chat Completions object for one answer."""
    message = {"role": "assistant", "content": text}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl-aggregator",
        "object": "chat.completion",
        "created": FIXED_CREATED,
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason,
                     "logprobs": None}],
        "usage": {"prompt_tokens": input_tokens, "completion_tokens": output_tokens,
                  "total_tokens": input_tokens + output_tokens},
    }


def anthropic_json_response(model, text, stop_reason, input_tokens, output_tokens, tool_blocks=None):
    """An Anthropic Messages object for one answer."""
    content = []
    if text:
        content.append({"type": "text", "text": text})
    if tool_blocks:
        content.extend(tool_blocks)
    if not content:
        content.append({"type": "text", "text": ""})
    return {
        "id": "msg_aggregator",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }


def openai_tool_calls_to_anthropic(message):
    """An OpenAI message's tool_calls as Anthropic tool_use blocks."""
    tool_blocks = []
    for call in message.get("tool_calls") or []:
        if not is_record(call) or not is_record(call.get("function")):
            continue
        fn = call["function"]
        tool_blocks.append({"type": "tool_use",
                            "id": str_of(call.get("id")) or "toolu_unknown",
                            "name": str_of(fn.get("name")),
                            "input": args_of(fn.get("arguments"))})
    return tool_blocks


def anthropic_blocks_to_openai(content):
    """An Anthropic content list as (text, tool_calls) for an OpenAI message."""
    text = ""
    tool_calls = []
    if isinstance(content, list):
        for block in content:
            if not is_record(block):
                continue
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                text += block["text"]
            elif block.get("type") == "tool_use":
                tool_calls.append({
                    "id": str_of(block.get("id")),
                    "type": "function",
                    "function": {"name": str_of(block.get("name")),
                                 "arguments": json_compact(block.get("input") if is_record(block.get("input")) else {})},
                })
    return text, tool_calls


def openai_json_to_anthropic(body, model):
    """A Chat Completions answer as an Anthropic Messages object."""
    choices = body.get("choices") if isinstance(body.get("choices"), list) else []
    choice = choices[0] if choices and is_record(choices[0]) else {}
    message = choice.get("message") if is_record(choice.get("message")) else {}
    text = message.get("content") if isinstance(message.get("content"), str) else text_of(message.get("content"))
    usage = body.get("usage") if is_record(body.get("usage")) else {}
    return anthropic_json_response(
        model, text, stop_reason_of(choice.get("finish_reason")),
        num_or_zero(usage.get("prompt_tokens")), num_or_zero(usage.get("completion_tokens")),
        openai_tool_calls_to_anthropic(message))


def anthropic_json_to_openai(body, model):
    """An Anthropic Messages answer as a Chat Completions object."""
    text, tool_calls = anthropic_blocks_to_openai(body.get("content"))
    usage = body.get("usage") if is_record(body.get("usage")) else {}
    return openai_json_response(
        model, text, finish_reason_of(body.get("stop_reason")),
        num_or_zero(usage.get("input_tokens")), num_or_zero(usage.get("output_tokens")),
        tool_calls or None)


def gemini_text_and_usage(body):
    """(text, finish_reason, input_tokens, output_tokens) of a Gemini answer."""
    candidates = body.get("candidates") if isinstance(body.get("candidates"), list) else []
    cand = candidates[0] if candidates and is_record(candidates[0]) else {}
    parts = []
    content = cand.get("content") if is_record(cand.get("content")) else {}
    if isinstance(content.get("parts"), list):
        for part in content["parts"]:
            if is_record(part) and isinstance(part.get("text"), str):
                parts.append(part["text"])
    meta = body.get("usageMetadata") if is_record(body.get("usageMetadata")) else {}
    return ("".join(parts), str_of(cand.get("finishReason")) or "STOP",
            num_or_zero(meta.get("promptTokenCount")),
            num_or_zero(meta.get("candidatesTokenCount")))


def gemini_stop_reason(finish):
    return {"STOP": "end_turn", "MAX_TOKENS": "max_tokens", "SAFETY": "refusal"}.get(finish, "end_turn")


def gemini_json_to_openai(body, model):
    text, finish, input_tokens, output_tokens = gemini_text_and_usage(body)
    return openai_json_response(model, text, finish_reason_of(finish),
                                input_tokens, output_tokens)


def gemini_json_to_anthropic(body, model):
    text, finish, input_tokens, output_tokens = gemini_text_and_usage(body)
    return anthropic_json_response(model, text, gemini_stop_reason(finish),
                                   input_tokens, output_tokens)


def provider_input_tokens(client_api, body):
    """The provider's input-token count for this request, from its own
    formula: 1 + floor((len(system) + len(last user text) + len(tool names
    joined)) / 4).  The translations preserve those strings exactly, so
    this matches the count the provider itself reports."""
    if client_api == "openai":
        system = "\n".join(text_of(m.get("content")) for m in body.get("messages") or []
                           if is_record(m) and m.get("role") == "system" and text_of(m.get("content")))
        user_text = ""
        for m in body.get("messages") or []:
            if is_record(m) and m.get("role") == "user":
                user_text = text_of(m.get("content"))
        tools = []
        if isinstance(body.get("tools"), list):
            for t in body["tools"]:
                if is_record(t) and is_record(t.get("function")) and isinstance(t["function"].get("name"), str):
                    tools.append(t["function"]["name"])
    else:
        system = text_of(body.get("system"))
        user_text = ""
        for m in body.get("messages") or []:
            if is_record(m) and m.get("role") == "user":
                user_text = text_of(m.get("content"))
        tools = []
        if isinstance(body.get("tools"), list):
            for t in body["tools"]:
                if is_record(t) and isinstance(t.get("name"), str):
                    tools.append(t["name"])
    prompt = len(system) + len(user_text) + len(",".join(tools))
    return 1 + prompt // 4


# --- SSE -------------------------------------------------------------------


def sse_event(event, data):
    """One SSE frame: `event: <name>` when named, then `data: <compact JSON>`."""
    text = json_compact(data)
    if event is None:
        return "data: %s\n\n" % text
    return "event: %s\ndata: %s\n\n" % (event, text)


def parse_sse_frames(text):
    """[(event, data_string)] for one SSE body, in order."""
    frames = []
    normalized = text.replace("\r\n", "\n")
    for frame in normalized.split("\n\n"):
        if frame.strip() == "":
            continue
        event = None
        data = []
        for line in frame.split("\n"):
            if line.startswith("event:"):
                event = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data.append(line[len("data:"):].lstrip(" "))
        if data:
            frames.append((event, "\n".join(data)))
    return frames


def openai_chunk(model, delta, finish_reason=None, usage=None):
    """One Chat Completions chunk.  A usage-only chunk carries no choices,
    as the wire expects."""
    chunk = {"id": "chatcmpl-aggregator", "object": "chat.completion.chunk",
             "created": FIXED_CREATED, "model": model,
             "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]}
    if usage is not None:
        chunk["choices"] = []
        chunk["usage"] = usage
    return chunk


def anthropic_message_start(model, input_tokens):
    return sse_event("message_start", {
        "type": "message_start",
        "message": {"id": "msg_stream", "type": "message", "role": "assistant",
                    "model": model, "content": [], "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": input_tokens, "output_tokens": 0}}})


def openai_sse_to_anthropic(text, model, input_tokens):
    """A Chat Completions SSE stream as the Anthropic event sequence."""
    out = []
    started = False
    finished = False
    pending_stop = None
    text_index = -1
    next_index = 0
    tool_indexes = {}
    output_tokens = 0

    def start_events():
        nonlocal started
        if started:
            return []
        started = True
        return [anthropic_message_start(model, input_tokens)]

    def close_events():
        nonlocal text_index
        events = []
        if text_index != -1:
            events.append(sse_event("content_block_stop",
                                    {"type": "content_block_stop", "index": text_index}))
            text_index = -1
        for index in sorted(tool_indexes.values()):
            events.append(sse_event("content_block_stop",
                                    {"type": "content_block_stop", "index": index}))
        tool_indexes.clear()
        return events

    def finish_events(stop_reason):
        nonlocal finished
        if finished:
            return []
        finished = True
        return (close_events()
                + [sse_event("message_delta", {
                    "type": "message_delta",
                    "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                    "usage": {"output_tokens": output_tokens}}),
                   sse_event("message_stop", {"type": "message_stop"})])

    for _event, payload in parse_sse_frames(text):
        payload = payload.strip()
        if payload == "[DONE]":
            out.extend(start_events())
            out.extend(finish_events(pending_stop or "end_turn"))
            continue
        chunk = safe_json_loads(payload)
        if not is_record(chunk):
            continue
        if is_record(chunk.get("usage")):
            output_tokens = num_or_zero(chunk["usage"].get("completion_tokens"))
        choices = chunk.get("choices") if isinstance(chunk.get("choices"), list) else []
        choice = choices[0] if choices and is_record(choices[0]) else None
        if choice is None:
            continue
        delta = choice.get("delta") if is_record(choice.get("delta")) else {}
        out.extend(start_events())
        content = delta.get("content")
        if isinstance(content, str) and content:
            if text_index == -1:
                text_index = next_index
                next_index += 1
                out.append(sse_event("content_block_start", {
                    "type": "content_block_start", "index": text_index,
                    "content_block": {"type": "text", "text": ""}}))
            out.append(sse_event("content_block_delta", {
                "type": "content_block_delta", "index": text_index,
                "delta": {"type": "text_delta", "text": content}}))
        for call in delta.get("tool_calls") or []:
            if not is_record(call):
                continue
            upstream_index = call.get("index") if isinstance(call.get("index"), int) else len(tool_indexes)
            index = tool_indexes.get(upstream_index)
            if index is None:
                if text_index != -1:
                    out.append(sse_event("content_block_stop",
                                         {"type": "content_block_stop", "index": text_index}))
                    text_index = -1
                index = next_index
                next_index += 1
                tool_indexes[upstream_index] = index
                fn = call.get("function") if is_record(call.get("function")) else {}
                out.append(sse_event("content_block_start", {
                    "type": "content_block_start", "index": index,
                    "content_block": {"type": "tool_use",
                                      "id": str_of(call.get("id")) or "toolu_%d" % index,
                                      "name": str_of(fn.get("name")), "input": {}}}))
            fn = call.get("function") if is_record(call.get("function")) else {}
            if isinstance(fn.get("arguments"), str) and fn["arguments"]:
                out.append(sse_event("content_block_delta", {
                    "type": "content_block_delta", "index": index,
                    "delta": {"type": "input_json_delta", "partial_json": fn["arguments"]}}))
        if choice.get("finish_reason") is not None:
            pending_stop = stop_reason_of(choice["finish_reason"])
    if not finished:
        out.extend(start_events())
        out.extend(finish_events(pending_stop or "end_turn"))
    return "".join(out)


def anthropic_sse_to_openai(text, model):
    """An Anthropic SSE stream as Chat Completions chunks."""
    out = []
    sent_role = False
    input_tokens = 0
    output_tokens = 0
    stop_reason = "end_turn"
    tool_args = {}

    def role_chunk():
        nonlocal sent_role
        if sent_role:
            return []
        sent_role = True
        return [sse_event(None, openai_chunk(model, {"role": "assistant", "content": ""}))]

    for event, payload in parse_sse_frames(text):
        if event is None:
            continue
        data = safe_json_loads(payload)
        if not is_record(data):
            continue
        if event == "message_start":
            message = data.get("message") if is_record(data.get("message")) else {}
            usage = message.get("usage") if is_record(message.get("usage")) else {}
            input_tokens = num_or_zero(usage.get("input_tokens"))
        elif event == "content_block_delta":
            delta = data.get("delta") if is_record(data.get("delta")) else {}
            index = data.get("index")
            if delta.get("type") == "text_delta" and isinstance(delta.get("text"), str) and delta["text"]:
                out.extend(role_chunk())
                out.append(sse_event(None, openai_chunk(model, {"content": delta["text"]})))
            elif delta.get("type") == "input_json_delta" and isinstance(delta.get("partial_json"), str):
                tool_args[index] = tool_args.get(index, "") + delta["partial_json"]
                out.extend(role_chunk())
                out.append(sse_event(None, openai_chunk(
                    model, {"tool_calls": [{"index": index if isinstance(index, int) else 0,
                                            "function": {"arguments": delta["partial_json"]}}]})))
        elif event == "message_delta":
            delta = data.get("delta") if is_record(data.get("delta")) else {}
            if isinstance(delta.get("stop_reason"), str):
                stop_reason = delta["stop_reason"]
            usage = data.get("usage") if is_record(data.get("usage")) else {}
            output_tokens = num_or_zero(usage.get("output_tokens"))
        elif event == "message_stop":
            pass
    out.extend(role_chunk())
    out.append(sse_event(None, openai_chunk(model, {}, finish_reason_of(stop_reason))))
    out.append(sse_event(None, openai_chunk(
        model, {}, usage={"prompt_tokens": input_tokens, "completion_tokens": output_tokens,
                          "total_tokens": input_tokens + output_tokens})))
    out.append("data: [DONE]\n\n")
    return "".join(out)


def gemini_sse_frames(text):
    """The parsed frames of a Gemini streamGenerateContent SSE body."""
    frames = []
    for _event, payload in parse_sse_frames(text):
        frame = safe_json_loads(payload)
        if is_record(frame):
            frames.append(frame)
    return frames


def gemini_frame_text_and_finish(frame):
    """(text_piece, finish_reason, input_tokens, output_tokens) of one frame."""
    candidates = frame.get("candidates") if isinstance(frame.get("candidates"), list) else []
    cand = candidates[0] if candidates and is_record(candidates[0]) else {}
    content = cand.get("content") if is_record(cand.get("content")) else {}
    piece = ""
    if isinstance(content.get("parts"), list):
        for part in content["parts"]:
            if is_record(part) and isinstance(part.get("text"), str):
                piece += part["text"]
    meta = frame.get("usageMetadata") if is_record(frame.get("usageMetadata")) else {}
    return (piece,
            str_of(cand.get("finishReason")) if isinstance(cand.get("finishReason"), str) else None,
            num_or_zero(meta.get("promptTokenCount")),
            num_or_zero(meta.get("candidatesTokenCount")))


def gemini_sse_to_openai(text, model):
    """A Gemini streamGenerateContent SSE stream as Chat Completions chunks."""
    out = []
    sent_role = False
    finish = None
    input_tokens = 0
    output_tokens = 0

    def role_chunk():
        nonlocal sent_role
        if sent_role:
            return []
        sent_role = True
        return [sse_event(None, openai_chunk(model, {"role": "assistant", "content": ""}))]

    for frame in gemini_sse_frames(text):
        piece, frame_finish, in_tok, out_tok = gemini_frame_text_and_finish(frame)
        if piece:
            out.extend(role_chunk())
            out.append(sse_event(None, openai_chunk(model, {"content": piece})))
        if frame_finish is not None:
            finish = frame_finish
        meta = frame.get("usageMetadata")
        if is_record(meta) and meta:
            input_tokens = in_tok
            output_tokens = out_tok
    out.extend(role_chunk())
    out.append(sse_event(None, openai_chunk(model, {}, finish_reason_of(finish or "STOP"))))
    out.append(sse_event(None, openai_chunk(
        model, {}, usage={"prompt_tokens": input_tokens, "completion_tokens": output_tokens,
                          "total_tokens": input_tokens + output_tokens})))
    out.append("data: [DONE]\n\n")
    return "".join(out)


def gemini_sse_to_anthropic(text, model, input_tokens):
    """A Gemini streamGenerateContent SSE stream as the Anthropic event sequence."""
    out = [anthropic_message_start(model, input_tokens),
           sse_event("content_block_start", {
               "type": "content_block_start", "index": 0,
               "content_block": {"type": "text", "text": ""}})]
    finish = "STOP"
    output_tokens = 0
    for frame in gemini_sse_frames(text):
        piece, frame_finish, _in_tok, out_tok = gemini_frame_text_and_finish(frame)
        if piece:
            out.append(sse_event("content_block_delta", {
                "type": "content_block_delta", "index": 0,
                "delta": {"type": "text_delta", "text": piece}}))
        if frame_finish is not None:
            finish = frame_finish
        if is_record(frame.get("usageMetadata")):
            output_tokens = out_tok
    out.append(sse_event("content_block_stop", {"type": "content_block_stop", "index": 0}))
    out.append(sse_event("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": gemini_stop_reason(finish), "stop_sequence": None},
        "usage": {"output_tokens": output_tokens}}))
    out.append(sse_event("message_stop", {"type": "message_stop"}))
    return "".join(out)

# ---------------------------------------------------------------------------
# routing and serving
#
# One client request is served by one provider call, failing over across
# providers until one answers.  The loop is deterministic: candidates are
# ranked by effective score (rating minus the failure penalty), ties break
# on the provider's brief order and then the model id, and nothing depends
# on timing or randomness.
# ---------------------------------------------------------------------------


def requirements_for(body, client_api):
    """The manifest's row for this request, by arrival order.

    The manifest is the authority while it lasts; once it runs out (or the
    brief carried none), requirements are read off the request body itself.
    The cursor advances once per request either way.
    """
    global MANIFEST_INDEX
    manifest = BRIEF.workload["manifest"] if BRIEF.workload is not None else []
    entry = manifest[MANIFEST_INDEX] if MANIFEST_INDEX < len(manifest) else None
    MANIFEST_INDEX += 1
    if entry is not None:
        return entry["requires"]
    return requirements_from_body(body, client_api)


def requirements_from_body(body, client_api):
    """Requirements read off a request body (used past the manifest's end)."""
    tools = isinstance(body.get("tools"), list) and len(body["tools"]) > 0
    vision = False
    for m in body.get("messages") or []:
        if not is_record(m) or not isinstance(m.get("content"), list):
            continue
        for block in m["content"]:
            if not is_record(block):
                continue
            if client_api == "openai" and block.get("type") in ("image", "image_url"):
                vision = True
            elif client_api != "openai" and block.get("type") == "image":
                vision = True
    return {"tools": tools, "vision": vision,
            "json": openai_json_mode(body) if client_api == "openai" else False}


def candidate_score(cand):
    """A candidate's effective score: rating, less FAILURE_PENALTY per
    consecutive failure of its provider, less FAILURE_EXCLUSION once the
    provider has failed FAILURE_EXCLUSION_AFTER times in a row."""
    fails = PROVIDER_STATE[cand["provider"].id]["consec_fail"]
    score = cand["rating"] - FAILURE_PENALTY * fails
    if fails >= FAILURE_EXCLUSION_AFTER:
        score -= FAILURE_EXCLUSION
    return score


def rank_key(cand):
    """Sort key for candidates: best effective score first, then the
    provider's brief order, then the model id -- fully deterministic."""
    return (-candidate_score(cand), cand["provider"].index, cand["model"])


def eligible_candidates(requires, client_api, body, tried_providers=None):
    """Free candidates meeting the request, one (the best-rated) per
    provider, in rank order.  JSON-mode requests from an OpenAI client are
    never sent to an Anthropic-wire provider: it cannot carry
    `response_format`, so the echo would come back without JSON mode and
    lose the request."""
    tried = tried_providers if tried_providers is not None else set()
    json_mode = openai_json_mode(body) if client_api == "openai" else False
    best_per_provider = {}
    for cand in CANDIDATES:
        if json_mode and cand["provider"].wire == "anthropic":
            continue
        if not meets_requirements(requires, cand["caps"]):
            continue
        pid = cand["provider"].id
        if pid not in best_per_provider or cand["rating"] > best_per_provider[pid]["rating"]:
            best_per_provider[pid] = cand
    return sorted((c for c in best_per_provider.values() if c["provider"].id not in tried),
                  key=rank_key)


def pick_candidate(requires, client_api, body, tried_providers):
    """Best not-yet-tried provider for this request, with its best model."""
    ranked = eligible_candidates(requires, client_api, body, tried_providers)
    return ranked[0] if ranked else None


def build_upstream(cand, client_api, body, stream):
    """(url, headers, body_bytes) for one provider call in its own wire."""
    provider = cand["provider"]
    base = join_base(ENV_URL, provider.base)
    headers = {"content-type": "application/json"}
    headers.update(auth_headers(provider.auth, provider.key))
    if provider.wire == "openai":
        if client_api == "openai":
            upstream = dict(body)
            upstream["model"] = cand["model"]
            if stream:
                upstream["stream"] = True
                upstream["stream_options"] = {"include_usage": True}
            payload = upstream
        else:
            payload = anthropic_to_openai(body, cand["model"])
        return base + "/v1/chat/completions", headers, json_compact(payload).encode("utf-8")
    if provider.wire == "anthropic":
        headers["anthropic-version"] = ANTHROPIC_VERSION
        if client_api == "anthropic":
            upstream = dict(body)
            upstream["model"] = cand["model"]
            payload = upstream
        else:
            payload = openai_to_anthropic(body, cand["model"])
        return base + "/v1/messages", headers, json_compact(payload).encode("utf-8")
    # Gemini wire: :generateContent, or :streamGenerateContent with ?alt=sse.
    if client_api == "openai":
        payload, _model = openai_to_gemini(body, cand["model"])
    else:
        payload, _model = anthropic_to_gemini(body, cand["model"])
    quoted = urllib.parse.quote(cand["model"], safe="")
    if stream:
        url = "%s/v1beta/models/%s:streamGenerateContent?alt=sse" % (base, quoted)
    else:
        url = "%s/v1beta/models/%s:generateContent" % (base, quoted)
    return url, headers, json_compact(payload).encode("utf-8")


def translate_response(cand, client_api, body, stream, status, data):
    """(ok, content_type, body_bytes): the provider's answer in the client's
    wire.  Same-wire answers pass through untouched; cross-wire answers are
    rebuilt from the provider's fields."""
    provider = cand["provider"]
    model = cand["model"]
    if stream:
        text = data.decode("utf-8", "replace")
        try:
            if provider.wire == "openai":
                if client_api == "openai":
                    return True, "text/event-stream", data
                return True, "text/event-stream", openai_sse_to_anthropic(
                    text, model, provider_input_tokens(client_api, body)).encode("utf-8")
            if provider.wire == "anthropic":
                if client_api == "anthropic":
                    return True, "text/event-stream", data
                return True, "text/event-stream", anthropic_sse_to_openai(text, model).encode("utf-8")
            if client_api == "openai":
                return True, "text/event-stream", gemini_sse_to_openai(text, model).encode("utf-8")
            return True, "text/event-stream", gemini_sse_to_anthropic(
                text, model, provider_input_tokens(client_api, body)).encode("utf-8")
        except Exception as err:
            log("stream translation failed for %s/%s: %s" % (provider.id, model, err))
            return False, None, None
    answer = safe_json_loads(data.decode("utf-8"))
    if not is_record(answer):
        return False, None, None
    try:
        if provider.wire == "openai":
            if client_api == "openai":
                return True, "application/json", data
            return True, "application/json", json_compact(
                openai_json_to_anthropic(answer, model)).encode("utf-8")
        if provider.wire == "anthropic":
            if client_api == "anthropic":
                return True, "application/json", data
            return True, "application/json", json_compact(
                anthropic_json_to_openai(answer, model)).encode("utf-8")
        if client_api == "openai":
            return True, "application/json", json_compact(
                gemini_json_to_openai(answer, model)).encode("utf-8")
        return True, "application/json", json_compact(
            gemini_json_to_anthropic(answer, model)).encode("utf-8")
    except Exception as err:
        log("response translation failed for %s/%s: %s" % (provider.id, model, err))
        return False, None, None


def best_untried_score(requires, client_api, body, tried_providers):
    """Best effective score among candidates on providers not yet tried for
    this request, or None when every eligible provider has been tried."""
    ranked = eligible_candidates(requires, client_api, body, tried_providers)
    return candidate_score(ranked[0]) if ranked else None


def serve(client_api, body):
    """Answer one client request.  Returns (status, content_type, body_bytes)."""
    stream = body.get("stream") is True
    requires = requirements_for(body, client_api)
    tried_providers = set()
    retried_providers = set()

    def try_once(cand):
        """One call to one provider.  Returns (kind, status, content_type, out)
        with kind 'ok', 'refused' (an HTTP error) or 'bad' (an unusable 200
        body).  The provider's failure counter is updated either way."""
        provider = cand["provider"]
        state = PROVIDER_STATE[provider.id]
        url, headers, payload = build_upstream(cand, client_api, body, stream)
        status, _resp_headers, data = http_request("POST", url, headers=headers, body=payload,
                                                   timeout=PROVIDER_TIMEOUT_S)
        state["calls"] += 1
        if status < 200 or status >= 300:
            state["consec_fail"] += 1
            log("%s/%s refused HTTP %s; failing over" % (provider.id, cand["model"], status))
            return "refused", status, None, None
        ok, content_type, out = translate_response(cand, client_api, body, stream, status, data)
        if not ok:
            state["consec_fail"] += 1
            log("%s/%s answered unusable bytes; failing over" % (provider.id, cand["model"]))
            return "bad", status, None, None
        state["consec_fail"] = 0
        return "ok", status, content_type, out

    def retry_worth_it(cand, status):
        """A refusal is usually one scheduled fail call: if this provider's
        model still beats every untried alternative, spend one retry on its
        next call before moving on.  Never retry a 400/401/404, and never
        retry the same provider twice for one request."""
        if cand["provider"].id in retried_providers:
            return False
        if status not in RETRYABLE_STATUSES and status < 500:
            return False
        best_alt = best_untried_score(requires, client_api, body, tried_providers)
        return best_alt is None or cand["rating"] > best_alt

    def fallback_candidate():
        """Best tried-but-not-retried provider for this request, or None.
        Used only once every eligible provider has been tried and refused."""
        json_mode = openai_json_mode(body) if client_api == "openai" else False
        remaining = [c for c in CANDIDATES
                     if c["provider"].id in tried_providers
                     and c["provider"].id not in retried_providers
                     and meets_requirements(requires, c["caps"])
                     and not (json_mode and c["provider"].wire == "anthropic")]
        if not remaining:
            return None
        remaining.sort(key=rank_key)
        return remaining[0]

    while True:
        cand = pick_candidate(requires, client_api, body, tried_providers)
        if cand is None:
            # No untried provider left: one retry of the best tried provider
            # that has not been retried yet, else the request is unservable.
            cand = fallback_candidate()
            if cand is None:
                error = (anthropic_error("overloaded_error", "no free model meets this request")
                         if client_api == "anthropic"
                         else openai_error("no free model meets this request", "no_free_model"))
                return 503, "application/json", json_compact(error).encode("utf-8")
            retried_providers.add(cand["provider"].id)
        provider = cand["provider"]
        tried_providers.add(provider.id)
        kind, status, content_type, out = try_once(cand)
        if kind == "ok":
            return 200, content_type, out
        if kind == "refused" and retry_worth_it(cand, status):
            # Immediate retry of the SAME provider, not a re-ranking: the
            # penalty would otherwise let another provider win the tie.
            retried_providers.add(provider.id)
            kind, status, content_type, out = try_once(cand)
            if kind == "ok":
                return 200, content_type, out


# ---------------------------------------------------------------------------
# HTTP server
#
# One process-wide lock serializes request handling: the env's client sends
# requests one at a time, and serial handling keeps the manifest cursor,
# the failure counters and the provider-call order deterministic -- and
# keeps us clear of the one-call-at-a-time provider limit.
# ---------------------------------------------------------------------------


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "FreeModelAggregator/1.0"

    def log_message(self, *args):
        pass

    def _read_body(self):
        length = self.headers.get("Content-Length")
        if length is None:
            return b""
        try:
            n = int(length)
        except ValueError:
            return b""
        if n < 0 or n > MAX_BODY_BYTES:
            return b""
        return self.rfile.read(n)

    def _send(self, status, content_type, body):
        self.send_response(status)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _error(self, wire, status, type_, message):
        body = anthropic_error(type_, message) if wire == "anthropic" else openai_error(message, type_)
        self._send(status, "application/json", json_compact(body).encode("utf-8"))

    def do_GET(self):
        with HANDLE_LOCK:
            try:
                self._do_get()
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as err:
                log("GET handler failed: %s" % err)

    def _do_get(self):
        path = urllib.parse.urlparse(self.path).path
        if path == "/v1/models":
            # The wire is named by the auth header, not the path: a Bearer
            # request gets the OpenAI list, an x-api-key request the
            # Anthropic one.
            models = served_models()
            if isinstance(self.headers.get("x-api-key"), str):
                self._send(200, "application/json",
                           json_compact(anthropic_model_list(models)).encode("utf-8"))
            else:
                self._send(200, "application/json",
                           json_compact(openai_model_list(models)).encode("utf-8"))
            return
        self._error("openai", 404, "not_found", "no route GET %s" % path)

    def do_POST(self):
        with HANDLE_LOCK:
            try:
                self._do_post()
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as err:
                log("POST handler failed: %s" % err)
                try:
                    wire = "anthropic" if self.path.startswith("/v1/messages") else "openai"
                    self._error(wire, 500, "internal_error", "the server failed to answer")
                except Exception:
                    pass

    def _do_post(self):
        path = urllib.parse.urlparse(self.path).path
        raw = self._read_body()
        if path == "/v1/chat/completions":
            wire, api = "openai", "openai"
        elif path == "/v1/messages":
            wire, api = "anthropic", "anthropic"
        else:
            self._error("openai", 404, "not_found", "no route POST %s" % path)
            return
        body = safe_json_loads(raw.decode("utf-8"))
        if not is_record(body):
            self._error(wire, 400, "invalid_request_error",
                        "the request body is not one JSON object")
            return
        status, content_type, out = serve(api, body)
        self._send(status, content_type, out)


# ---------------------------------------------------------------------------
# register, poll, main
# ---------------------------------------------------------------------------


def register(port):
    """POST the catalog to the env until it answers 200 (up to
    REGISTER_ATTEMPTS tries).  The env's reply names the request count."""
    body = json_compact({"protocol_version": PROTOCOL_VERSION, "port": port,
                         "catalog": CATALOG}).encode("utf-8")
    for attempt in range(REGISTER_ATTEMPTS):
        status, _hdrs, data = http_request(
            "POST", ENV_URL + "/register",
            headers={"content-type": "application/json"}, body=body, timeout=5)
        if status == 200:
            reply = safe_json_loads(data.decode("utf-8"))
            if is_record(reply) and isinstance(reply.get("requests"), int):
                log("registered: the workload holds %d requests" % reply["requests"])
            return
        if attempt % 10 == 9:
            log("register attempt %d answered HTTP %s; retrying" % (attempt + 1, status))
        threading.Event().wait(REGISTER_RETRY_WAIT_S)
    raise RuntimeError("POST %s/register never answered 200" % ENV_URL)


def poll_work(server, done_event):
    """Watch GET /work until it says done, then stop the server.  Runs on
    its own thread so serve_forever() keeps answering in the meantime."""
    while not done_event.is_set():
        status, _hdrs, data = http_request("GET", ENV_URL + "/work", timeout=2)
        if status == 200:
            work = safe_json_loads(data.decode("utf-8"))
            if is_record(work) and work.get("done") is True:
                done_event.set()
                server.shutdown()
                return
        done_event.wait(POLL_INTERVAL_S)


# ---------------------------------------------------------------------------
# self-test
#
# `python3 main.py --self-test` exercises the pure translator and routing
# helpers with synthetic inputs and exits 0 only when every assertion
# holds.  It runs before load_catalog(), so it needs no brief, no env URL
# and no network; the trial harness never passes --self-test, so this
# changes nothing about how the server serves.  Run it after any edit to
# the translators to catch a drift between the three wires before the
# practice does: the scorer checks the echo against what the client sent,
# so a translator that drops a field fails the wire check, not the routing.
# ---------------------------------------------------------------------------


def self_test():
    """Smoke-test the pure helpers. Returns (failures, total)."""
    failures = []
    total = [0]

    def check(name, cond):
        total[0] += 1
        if not cond:
            failures.append(name)
            print("FAIL: %s" % name)

    # --- stop/finish reason mapping (both directions, every branch) ---
    check("stop length", stop_reason_of("length") == "max_tokens")
    check("stop tool_calls", stop_reason_of("tool_calls") == "tool_use")
    check("stop function_call", stop_reason_of("function_call") == "tool_use")
    check("stop content_filter", stop_reason_of("content_filter") == "refusal")
    check("stop other", stop_reason_of("stop") == "end_turn")
    check("finish max_tokens", finish_reason_of("max_tokens") == "length")
    check("finish MAX_TOKENS", finish_reason_of("MAX_TOKENS") == "length")
    check("finish tool_use", finish_reason_of("tool_use") == "tool_calls")
    check("finish refusal", finish_reason_of("refusal") == "content_filter")
    check("finish SAFETY", finish_reason_of("SAFETY") == "content_filter")
    check("finish end_turn", finish_reason_of("end_turn") == "stop")
    # round-trip: every Anthropic stop reason survives to OpenAI and back
    for stop in ("end_turn", "max_tokens", "tool_use", "refusal"):
        check("round-trip " + stop, stop_reason_of(finish_reason_of(stop)) == stop)

    # --- error shapes (the client libraries parse these exactly) ---
    check("openai error default",
          openai_error("bad") == {"error": {"message": "bad",
                                            "type": "invalid_request_error",
                                            "code": None}})
    check("openai error custom",
          openai_error("bad", "auth", "E1")["error"]["type"] == "auth")
    check("anthropic error",
          anthropic_error("invalid_request_error", "bad")
          == {"type": "error",
              "error": {"type": "invalid_request_error", "message": "bad"}})

    # --- URL joining and auth headers (the probe path depends on these) ---
    check("join absolute base",
          join_base("https://env.example.com", "https://api.x.com/v2") == "https://api.x.com/v2")
    check("join relative base",
          join_base("https://env.example.com", "/v1") == "https://env.example.com/v1")
    check("join bare base",
          join_base("https://env.example.com", "v1") == "https://env.example.com/v1")
    check("join empty base",
          join_base("https://env.example.com", "") == "https://env.example.com")
    check("join trailing slash",
          join_base("https://env.example.com/", "/v1") == "https://env.example.com/v1")
    check("auth bearer", auth_headers("bearer", "k") == {"authorization": "Bearer k"})
    check("auth x-api-key", auth_headers("x-api-key", "k") == {"x-api-key": "k"})
    check("auth x-goog", auth_headers("x-goog-api-key", "k") == {"x-goog-api-key": "k"})
    check("auth no key", auth_headers("bearer", "") == {})
    check("auth unknown scheme", auth_headers("basic", "k") == {})

    # --- http_request refuses non-HTTP(S) URLs instead of misrouting them ---
    check("http ftp refused", http_request("GET", "ftp://x.com/") == (0, {}, b""))
    check("http no scheme refused", http_request("GET", "example.com/") == (0, {}, b""))
    check("http no host refused", http_request("GET", "https:///path") == (0, {}, b""))

    # --- routing predicates: the quality table wins, the probe is fallback ---
    check("model_free probe true", model_free({"free": True}, None) is True)
    check("model_free probe false", model_free({"free": False}, None) is False)
    check("model_free row wins true", model_free({"free": False}, {"free": True}) is True)
    check("model_free row wins false", model_free({"free": True}, {"free": False}) is False)
    caps = {"tools": True, "vision": False, "json": True, "context_length": 8000}
    check("meets tools", meets_requirements({"tools": True}, caps) is True)
    check("meets vision fails", meets_requirements({"vision": True}, caps) is False)
    check("meets json", meets_requirements({"json": True}, caps) is True)
    check("meets min_context ok",
          meets_requirements({"min_context": 8000}, caps) is True)
    check("meets min_context fails",
          meets_requirements({"min_context": 8001}, caps) is False)
    check("meets empty", meets_requirements({}, caps) is True)

    # --- Anthropic -> OpenAI: text, tool_use, tool_result, image ---
    out = anthropic_to_openai(
        {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 10}, "m")
    check("a2o text", out["messages"] == [{"role": "user", "content": "hi"}])
    check("a2o max_tokens", out["max_tokens"] == 10)
    check("a2o model", out["model"] == "m")
    out = anthropic_to_openai(
        {"messages": [{"role": "assistant", "content": [
            {"type": "text", "text": "calling"},
            {"type": "tool_use", "id": "t1", "name": "get_weather",
             "input": {"city": "Paris"}}]}]}, "m")
    msg = out["messages"][0]
    check("a2o tool text", msg["content"] == "calling")
    check("a2o tool_calls",
          msg["tool_calls"] == [{"id": "t1", "type": "function",
                                 "function": {"name": "get_weather",
                                            "arguments": '{"city":"Paris"}'}}])
    out = anthropic_to_openai(
        {"messages": [{"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1",
             "content": "sunny"}]}]}, "m")
    check("a2o tool_result",
          out["messages"] == [{"role": "tool", "tool_call_id": "t1", "content": "sunny"}])
    out = anthropic_to_openai(
        {"messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "url",
                                         "url": "https://x.com/i.png"}}]}]}, "m")
    check("a2o image",
          out["messages"][0]["content"]
          == [{"type": "image_url", "image_url": {"url": "https://x.com/i.png"}}])
    out = anthropic_to_openai(
        {"system": "be brief", "messages": [{"role": "user", "content": "hi"}],
         "stream": True}, "m")
    check("a2o system",
          out["messages"][0] == {"role": "system", "content": "be brief"})
    check("a2o stream", out.get("stream") is True
          and out.get("stream_options") == {"include_usage": True})

    # --- OpenAI -> Anthropic: text, tool_calls, tool role, system, image ---
    out = openai_to_anthropic(
        {"messages": [{"role": "user", "content": "hi"}]}, "m")
    check("o2a text",
          out["messages"] == [{"role": "user",
                               "content": [{"type": "text", "text": "hi"}]}])
    out = openai_to_anthropic(
        {"messages": [{"role": "assistant", "content": "calling",
                       "tool_calls": [{"id": "t1", "type": "function",
                                      "function": {"name": "get_weather",
                                                 "arguments": '{"city": "Paris"}'}}]}]},
        "m")
    blocks = out["messages"][0]["content"]
    check("o2a tool_use",
          blocks == [{"type": "text", "text": "calling"},
                     {"type": "tool_use", "id": "t1", "name": "get_weather",
                      "input": {"city": "Paris"}}])
    out = openai_to_anthropic(
        {"messages": [{"role": "tool", "tool_call_id": "t1", "content": "sunny"}]},
        "m")
    check("o2a tool role",
          out["messages"] == [{"role": "user",
                               "content": [{"type": "tool_result",
                                            "tool_use_id": "t1",
                                            "content": "sunny"}]}])
    out = openai_to_anthropic(
        {"messages": [{"role": "system", "content": "be brief"},
                      {"role": "system", "content": "and kind"},
                      {"role": "user", "content": "hi"}]}, "m")
    check("o2a system joined", out["system"] == "be brief\nand kind")
    out = openai_to_anthropic(
        {"messages": [{"role": "user", "content": [
            {"type": "image_url",
             "image_url": {"url": "https://x.com/i.png"}}]}]}, "m")
    check("o2a image",
          out["messages"][0]["content"]
          == [{"type": "image",
               "source": {"type": "url", "url": "https://x.com/i.png"}}])

    # --- image helpers: data URLs round-trip, plain URLs stay URLs ---
    part = openai_image_block("data:image/png;base64,AAAA")
    check("image data url",
          part == {"type": "image",
                   "source": {"type": "base64", "media_type": "image/png",
                              "data": "AAAA"}})
    check("image gemini data",
          gemini_image_part("data:image/png;base64,AAAA")
          == {"inlineData": {"mimeType": "image/png", "data": "AAAA"}})
    check("image gemini url",
          gemini_image_part("https://x.com/i.png")
          == {"fileData": {"fileUri": "https://x.com/i.png"}})

    # --- small value helpers: the translators lean on these everywhere ---
    check("is_record dict", is_record({}) is True)
    check("is_record list", is_record([]) is False)
    check("is_record None", is_record(None) is False)
    check("str_of str", str_of("x") == "x")
    check("str_of int", str_of(3) == "")
    check("str_of None", str_of(None) == "")
    check("num_or_none int", num_or_none(5) == 5)
    check("num_or_none float", num_or_none(2.5) == 2.5)
    check("num_or_none str", num_or_none("5") is None)
    check("num_or_none nan", num_or_none(float("nan")) is None)
    check("num_or_none inf", num_or_none(float("inf")) is None)
    check("num_or_zero int", num_or_zero(5) == 5)
    check("num_or_zero nan", num_or_zero(float("nan")) == 0)
    check("num_or_zero str", num_or_zero("5") == 0)
    check("bool_or_none true", bool_or_none(True) is True)
    check("bool_or_none false", bool_or_none(False) is False)
    check("bool_or_none int", bool_or_none(1) is None)
    check("json_compact", json_compact({"a": 1}) == '{"a":1}')
    check("json_compact unicode", json_compact({"a": "é"}) == '{"a":"é"}')
    check("safe_json_loads ok", safe_json_loads('{"a": 1}') == {"a": 1})
    check("safe_json_loads bad", safe_json_loads("not json") is None)
    check("safe_json_loads empty", safe_json_loads("") is None)
    check("text_of str", text_of("hi") == "hi")
    check("text_of blocks",
          text_of([{"type": "text", "text": "a"}, {"type": "text", "text": "b"},
                   {"type": "image", "source": {}}]) == "a\nb")
    check("text_of non-text", text_of([{"type": "image"}]) == "")
    check("text_of None", text_of(None) == "")
    check("text_of int", text_of(3) == "")
    check("string_list str", string_list("x") == ["x"])
    check("string_list mixed", string_list(["a", 1, "b", None]) == ["a", "b"])
    check("string_list None", string_list(None) == [])
    check("args_of dict", args_of({"a": 1}) == {"a": 1})
    check("args_of json str", args_of('{"a": 1}') == {"a": 1})
    check("args_of bad str", args_of("nope") == {})
    check("args_of empty str", args_of("  ") == {})
    check("args_of None", args_of(None) == {})
    check("parse_data_url png",
          parse_data_url("data:image/png;base64,AAAA") == ("image/png", "AAAA"))
    check("parse_data_url no mime",
          parse_data_url("data:;base64,AAAA") == ("image/png", "AAAA"))
    check("parse_data_url plain", parse_data_url("https://x.com/i.png") == (None, None))
    check("parse_data_url empty", parse_data_url("") == (None, None))
    check("tool_result str", tool_result_text("sunny") == "sunny")
    check("tool_result blocks",
          tool_result_text([{"type": "text", "text": "a"}]) == "a")
    check("tool_result None", tool_result_text(None) == "")
    check("tool_result int", tool_result_text(7) == "7")

    # --- brace matching and document scanning (the brief parser's core) ---
    check("object_end simple", object_end('{"a": 1}', 0) == 7)
    check("object_end nested", object_end('{"a": {"b": 2}}', 0) == 14)
    check("object_end string brace", object_end('{"a": "}"}', 0) == 9)
    check("object_end unclosed", object_end('{"a": 1', 0) == -1)
    check("object_end offset", object_end('xx{"a": 1}yy', 2) == 9)
    docs = documents_in('hello {"schema": "x", "v": 1} world {"no": 1} '
                        '{"schema": "y"} end')
    check("documents_in count", len(docs) == 2)
    check("documents_in first", docs[0] == {"schema": "x", "v": 1})
    check("documents_in second", docs[1] == {"schema": "y"})
    check("documents_in none", documents_in("no objects here") == [])
    check("documents_in non-string schema",
          documents_in('{"schema": 5}') == [])

    # --- tool declaration translation (both directions, shape preserved) ---
    tools = anthropic_tools_to_openai(
        [{"name": "get_weather",
          "description": "weather",
          "input_schema": {"type": "object",
                           "properties": {"city": {"type": "string"}}}}])
    check("tools a2o",
          tools == [{"type": "function",
                     "function": {"name": "get_weather",
                                "description": "weather",
                                "parameters": {"type": "object",
                                             "properties": {"city": {"type": "string"}}}}}])
    check("tools a2o empty", anthropic_tools_to_openai(None) == [])
    check("tools a2o bad", anthropic_tools_to_openai([{"no_name": 1}]) == [])
    back = openai_tools_to_anthropic(tools)
    check("tools round-trip",
          back == [{"name": "get_weather",
                    "description": "weather",
                    "input_schema": {"type": "object",
                                     "properties": {"city": {"type": "string"}}}}])
    check("tools o2a empty", openai_tools_to_anthropic(None) == [])

    # --- generation config: JSON mode and stop sequences survive ---
    cfg = gemini_generation_config({"temperature": 0.5, "top_p": 0.9}, 100,
                                   ["END"], True)
    check("gemini cfg temp", cfg.get("temperature") == 0.5)
    check("gemini cfg top_p", cfg.get("topP") == 0.9)
    check("gemini cfg max_tokens", cfg.get("maxOutputTokens") == 100)
    check("gemini cfg stops", cfg.get("stopSequences") == ["END"])
    check("gemini cfg json", cfg.get("responseMimeType") == "application/json")

    print("self-test: %d/%d passed" % (total[0] - len(failures), total[0]))
    return len(failures), total[0]


def main():
    if "--self-test" in sys.argv[1:]:
        failures, _total = self_test()
        return 1 if failures else 0
    load_catalog()
    if "--list" in sys.argv[1:]:
        sys.stdout.write(json_compact(CATALOG) + "\n")
        sys.stdout.flush()
        return 0
    if not ENV_URL:
        raise RuntimeError("MAKEFUN_ENV_URL is not set")
    server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
    server.daemon_threads = True
    port = server.server_address[1]
    if port == 8080:
        raise RuntimeError("the OS picked 8080, which the env owns on one layout: restart the trial")
    log("listening on 0.0.0.0:%d with %d answered providers and %d free models"
        % (port, len([p for p in PROBES if p["ok"]]), len(CANDIDATES)))
    register(port)
    done_event = threading.Event()

    def on_signal(_signum, _frame):
        done_event.set()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    poller = threading.Thread(target=poll_work, args=(server, done_event), daemon=True)
    poller.start()
    try:
        server.serve_forever(poll_interval=0.1)
    except SystemExit:
        pass
    server.server_close()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit as exc:
        sys.exit(exc.code if isinstance(exc.code, int) else 0)
    except Exception as err:
        log("fatal: %s" % err)
        sys.exit(1)
