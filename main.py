#!/usr/bin/env python3
"""Free-model aggregator: one server, two client wires, ten simulated providers.

Serves the OpenAI Chat Completions API and the Anthropic Messages API (JSON and
server-sent events) on top of the ten free-tier providers the trial's brief
names.  Standard library only.

Design
------
* The brief ($MAKEFUN_PROMPT_FILE) carries the provider directory, the sealed
  quality table and the workload manifest.  Each provider's catalog route is
  probed in its own wire; the catalog document built from those probes is what
  `--list` prints and what the register body carries, byte for byte.
* Routing is deterministic: requests arrive in manifest order, candidates are
  free models meeting the request's requirements, ranked by the quality
  table's rating (with a small penalty per consecutive provider failure, and
  near-exclusion after two consecutive failures).  A 402/429/5xx (or a
  network error) fails over to the next provider for the same request.
* Wires are translated faithfully both ways (OpenAI <-> Anthropic <-> Gemini)
  so the provider echoes exactly what the client sent: system prompt, user
  text, images, tool names, max_tokens, stop sequences and JSON mode.  Usage
  and stop reasons are carried back exactly as the serving provider reports
  them, mapped into the client's wire.
* Nothing here reads the wall clock or unseeded randomness, so a replay sees
  the same choices as the live pass.
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
POLL_INTERVAL_S = 0.1
PROBE_TIMEOUT_S = 5
PROVIDER_TIMEOUT_S = 10
MAX_BODY_BYTES = 32 * 1024 * 1024
FIXED_CREATED = 1767225600
FIXED_ANTHROPIC_CREATED_AT = "2026-01-01T00:00:00Z"
ANTHROPIC_VERSION = "2023-06-01"

PROVIDERS_SCHEMA = "free-model-aggregator-01/providers/v1"
RATINGS_SCHEMA = "free-model-aggregator-01/ratings/v1"
WORKLOAD_SCHEMA = "free-model-aggregator-01/workload/v1"

ENV_URL = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
PROMPT_FILE = os.environ.get("MAKEFUN_PROMPT_FILE", "")

# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def log(message):
    sys.stderr.write("aggregator: %s\n" % message)
    sys.stderr.flush()


def is_record(v):
    return isinstance(v, dict)


def str_of(v):
    return v if isinstance(v, str) else ""


def num_or_none(v):
    return v if isinstance(v, (int, float)) and v == v and abs(v) != float("inf") else None


def bool_or_none(v):
    return v if isinstance(v, bool) else None


def json_compact(value):
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def text_of(content):
    """Text of a string or of a block array's text blocks, joined with newline."""
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
    """A tool call's arguments as an object."""
    if is_record(value):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except Exception:
            return {}
        if is_record(parsed):
            return parsed
    return {}


def parse_data_url(url):
    """(mime_type, base64_data) for a data: URL, else (None, None)."""
    m = re.match(r"^data:([^;,]*);base64,(.*)$", url or "", re.S)
    if m is None:
        return None, None
    return m.group(1) or "image/png", m.group(2)


# ---------------------------------------------------------------------------
# the brief
# ---------------------------------------------------------------------------


def object_end(text, start):
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
    out = []
    start = text.find("{")
    while start != -1:
        end = object_end(text, start)
        if end != -1:
            try:
                value = json.loads(text[start:end + 1])
            except Exception:
                value = None
            if is_record(value) and isinstance(value.get("schema"), str):
                out.append(value)
        start = text.find("{", start + 1)
    return out


class Provider:
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


def provider_of(v, index):
    if not is_record(v):
        return None
    id_ = str_of(v.get("id"))
    wire = v.get("wire")
    if id_ == "" or wire not in ("openai", "anthropic", "gemini"):
        return None
    auth = v.get("auth")
    if auth not in ("bearer", "x-api-key", "x-goog-api-key", "none"):
        auth = "bearer"
    base = ""
    for key in ("path", "base_path", "base_url", "base"):
        if str_of(v.get(key)):
            base = str_of(v.get(key))
            break
    return Provider(id_, str_of(v.get("name")) or id_, wire, auth, base,
                    str_of(v.get("key_env")), str_of(v.get("key")), index)


def rating_of(v):
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
    def __init__(self):
        self.providers = []
        self.ratings = None  # {"ratings": [...], "best": [...]}
        self.workload = None  # {"manifest": [...]}
        self.error = None


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
    directory = next((d for d in docs if d.get("schema") == PROVIDERS_SCHEMA
                      and isinstance(d.get("providers"), list)), None)
    if directory is None:
        brief.error = "the brief at %s holds no %s document" % (path, PROVIDERS_SCHEMA)
        return brief
    for i, raw in enumerate(directory["providers"]):
        p = provider_of(raw, i)
        if p is not None:
            brief.providers.append(p)
    ratings_doc = next((d for d in docs if d.get("schema") == RATINGS_SCHEMA
                        and isinstance(d.get("ratings"), list)), None)
    if ratings_doc is not None:
        rows = [r for r in (rating_of(v) for v in ratings_doc["ratings"]) if r is not None]
        best = []
        if isinstance(ratings_doc.get("best"), list):
            for v in ratings_doc["best"]:
                if is_record(v) and str_of(v.get("id")):
                    best.append({"id": v["id"],
                                 "rating_milli": num_or_none(v.get("rating_milli")) or 0})
        brief.ratings = {"ratings": rows, "best": best}
    workload_doc = next((d for d in docs if d.get("schema") == WORKLOAD_SCHEMA
                         and isinstance(d.get("manifest"), list)), None)
    if workload_doc is not None:
        manifest = [w for w in (workload_request_of(v) for v in workload_doc["manifest"])
                    if w is not None]
        brief.workload = {"manifest": manifest}
    return brief


# ---------------------------------------------------------------------------
# providers: auth, probing, catalog
# ---------------------------------------------------------------------------


def join_base(env_url, base):
    b = (base or "").strip()
    if re.match(r"^https?://", b, re.I):
        return b.rstrip("/")
    prefix = env_url.rstrip("/")
    if b == "":
        return prefix
    return prefix + ("" if b.startswith("/") else "/") + b.rstrip("/")


def auth_headers(auth, key):
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
    """One HTTP call. Returns (status, headers_dict, body_bytes) or (0, {}, b'')."""
    parts = urllib.parse.urlparse(url)
    host = parts.hostname or ""
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
    if not is_record(v):
        return None
    if wire == "gemini":
        id_ = str_of(v.get("name")).replace("models/", "", 1) if str_of(v.get("name")).startswith("models/") else str_of(v.get("name"))
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
    if not is_record(body):
        return []
    lst = body.get("models") if wire == "gemini" else body.get("data")
    if not isinstance(lst, list):
        return []
    return [m for m in (model_in(wire, v) for v in lst) if m is not None]


def probe_provider(env_url, provider):
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
    try:
        body = json.loads(data.decode("utf-8"))
    except Exception as err:
        return {"provider": provider, "ok": False, "status": status, "models": [],
                "error": "GET %s is not JSON: %s" % (url, err)}
    return {"provider": provider, "ok": True, "status": status,
            "models": models_in(provider.wire, body), "error": None}


def rating_row(brief, provider_id, model_id):
    if brief.ratings is None:
        return None
    for row in brief.ratings["ratings"]:
        if row["provider"] == provider_id and row["model"] == model_id:
            return row
    return None


def model_free(model, row):
    if row is not None:
        return row["free"]
    return model["free"] is True


def capabilities_of(model, row):
    return {
        "context_length": (row["context_length"] if row is not None and row["context_length"] is not None
                           else model["context_length"]),
        "tools": row["tools"] if row is not None else model["tools"],
        "vision": row["vision"] if row is not None else model["vision"],
        "json": row["json"] if row is not None else model["json"],
    }


def meets_requirements(requires, caps):
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
    """The --list document and the register body's catalog: the same object."""
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
            row = rating_row(brief, probe["provider"].id, model["id"])
            ctx = row["context_length"] if row is not None and row["context_length"] is not None else model["context_length"]
            models.append({"id": model["id"], "free": model_free(model, row),
                           "context_length": ctx})
        providers.append({"id": probe["provider"].id, "base_url": probe["provider"].base,
                          "key_env": probe["provider"].key_env, "wire": probe["provider"].wire,
                          "models": models})
    return {"providers": providers}


# ---------------------------------------------------------------------------
# global trial state
# ---------------------------------------------------------------------------

BRIEF = read_brief(PROMPT_FILE)
PROBES = []
CATALOG = {"providers": []}
# candidates: [{provider, model, rating, caps, wire}] for every free model
CANDIDATES = []
# per-provider runtime state: {"calls": int, "consec_fail": int}
PROVIDER_STATE = {}
MANIFEST_INDEX = 0
HANDLE_LOCK = threading.Lock()


def load_catalog():
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
            row = rating_row(BRIEF, probe["provider"].id, model["id"])
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
    """Free models for GET /v1/models, deduplicated by id, brief order."""
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
# ---------------------------------------------------------------------------


def stop_reason_of(finish):
    if finish == "length":
        return "max_tokens"
    if finish in ("tool_calls", "function_call"):
        return "tool_use"
    if finish == "content_filter":
        return "refusal"
    return "end_turn"


def finish_reason_of(stop):
    if stop in ("max_tokens", "MAX_TOKENS"):
        return "length"
    if stop in ("tool_use", "tool_calls", "function_call"):
        return "tool_calls"
    if stop in ("refusal", "content_filter", "SAFETY"):
        return "content_filter"
    return "stop"


def tool_result_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text = text_of(content)
        return text if text else json_compact(content)
    if content is None:
        return ""
    return json_compact(content)


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
    if isinstance(body.get("temperature"), (int, float)):
        out["temperature"] = body["temperature"]
    if isinstance(body.get("top_p"), (int, float)):
        out["top_p"] = body["top_p"]
    if isinstance(body.get("stop_sequences"), list) and body["stop_sequences"]:
        out["stop"] = body["stop_sequences"]
    if isinstance(body.get("tools"), list):
        tools = []
        for raw in body["tools"]:
            if not is_record(raw) or not isinstance(raw.get("name"), str):
                continue
            tools.append({"type": "function",
                          "function": {"name": raw["name"],
                                       "description": str_of(raw.get("description")),
                                       "parameters": raw.get("input_schema") if is_record(raw.get("input_schema"))
                                       else {"type": "object", "properties": {}}}})
        if tools:
            out["tools"] = tools
    choice = body.get("tool_choice")
    if is_record(choice):
        if choice.get("type") == "auto":
            out["tool_choice"] = "auto"
        elif choice.get("type") == "any":
            out["tool_choice"] = "required"
        elif choice.get("type") == "none":
            out["tool_choice"] = "none"
        elif choice.get("type") == "tool" and isinstance(choice.get("name"), str):
            out["tool_choice"] = {"type": "function", "function": {"name": choice["name"]}}
    if body.get("stream") is True:
        out["stream"] = True
        out["stream_options"] = {"include_usage": True}
    return out


def openai_image_block(url):
    """An OpenAI image_url part as an Anthropic image block."""
    mime, data = parse_data_url(url)
    if data is not None:
        return {"type": "image",
                "source": {"type": "base64", "media_type": mime or "image/png", "data": data}}
    return {"type": "image", "source": {"type": "url", "url": url}}


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
    max_tokens = num_or_none(body.get("max_tokens"))
    if max_tokens is None:
        max_tokens = num_or_none(body.get("max_completion_tokens"))
    if max_tokens is not None:
        out["max_tokens"] = max_tokens
    if isinstance(body.get("temperature"), (int, float)):
        out["temperature"] = body["temperature"]
    if isinstance(body.get("top_p"), (int, float)):
        out["top_p"] = body["top_p"]
    stops = string_list(body.get("stop"))
    if stops:
        out["stop_sequences"] = stops
    if isinstance(body.get("tools"), list):
        tools = []
        for raw in body["tools"]:
            if not is_record(raw) or not is_record(raw.get("function")):
                continue
            fn = raw["function"]
            if not isinstance(fn.get("name"), str):
                continue
            tools.append({"name": fn["name"], "description": str_of(fn.get("description")),
                          "input_schema": fn.get("parameters") if is_record(fn.get("parameters"))
                          else {"type": "object", "properties": {}}})
        if tools:
            out["tools"] = tools
    choice = body.get("tool_choice")
    if choice == "auto":
        out["tool_choice"] = {"type": "auto"}
    elif choice == "required":
        out["tool_choice"] = {"type": "any"}
    elif choice == "none":
        out["tool_choice"] = {"type": "none"}
    elif is_record(choice) and choice.get("type") == "function" and is_record(choice.get("function")):
        out["tool_choice"] = {"type": "tool", "name": str_of(choice["function"].get("name"))}
    if body.get("stream") is True:
        out["stream"] = True
    return out


def gemini_image_part(url):
    """An image URL as a Gemini part: inlineData for data URLs, fileData else."""
    mime, data = parse_data_url(url)
    if data is not None:
        return {"inlineData": {"mimeType": mime or "image/png", "data": data}}
    return {"fileData": {"fileUri": url}}


def _gemini_generation_config(body, max_tokens, stops, json_mode):
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


def openai_to_gemini(body, model):
    """An OpenAI Chat Completions request as a Gemini generateContent request."""
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
    out = {"contents": contents}
    if system_parts:
        out["systemInstruction"] = {"parts": [{"text": "\n".join(system_parts)}]}
    if isinstance(body.get("tools"), list):
        decls = []
        for raw in body["tools"]:
            if not is_record(raw) or not is_record(raw.get("function")):
                continue
            fn = raw["function"]
            if not isinstance(fn.get("name"), str):
                continue
            decls.append({"name": fn["name"], "description": str_of(fn.get("description")),
                          "parameters": fn.get("parameters") if is_record(fn.get("parameters"))
                          else {"type": "object", "properties": {}}})
        if decls:
            out["tools"] = [{"functionDeclarations": decls}]
    max_tokens = num_or_none(body.get("max_tokens"))
    if max_tokens is None:
        max_tokens = num_or_none(body.get("max_completion_tokens"))
    fmt = body.get("response_format") if is_record(body.get("response_format")) else {}
    json_mode = fmt.get("type") in ("json_object", "json_schema")
    config = _gemini_generation_config(body, max_tokens, string_list(body.get("stop")), json_mode)
    if config:
        out["generationConfig"] = config
    return out, model


def anthropic_to_gemini(body, model):
    """An Anthropic Messages request as a Gemini generateContent request."""
    system = text_of(body.get("system"))
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
                        source = block.get("source") if is_record(block.get("source")) else {}
                        if source.get("type") == "base64" and isinstance(source.get("data"), str):
                            parts.append({"inlineData": {
                                "mimeType": source.get("media_type") if isinstance(source.get("media_type"), str) else "image/png",
                                "data": source["data"]}})
                        elif source.get("type") == "url" and isinstance(source.get("url"), str):
                            parts.append({"fileData": {"fileUri": source["url"]}})
                    elif block.get("type") == "tool_result":
                        name = tool_names.get(str_of(block.get("tool_use_id")), "")
                        parts.append({"functionResponse": {"name": name,
                                                          "response": {"content": tool_result_text(block.get("content"))}}})
            contents.append({"role": "user", "parts": parts})
    out = {"contents": contents}
    if system:
        out["systemInstruction"] = {"parts": [{"text": system}]}
    if isinstance(body.get("tools"), list):
        decls = []
        for raw in body["tools"]:
            if not is_record(raw) or not isinstance(raw.get("name"), str):
                continue
            decls.append({"name": raw["name"], "description": str_of(raw.get("description")),
                          "parameters": raw.get("input_schema") if is_record(raw.get("input_schema"))
                          else {"type": "object", "properties": {}}})
        if decls:
            out["tools"] = [{"functionDeclarations": decls}]
    config = _gemini_generation_config(
        body, num_or_none(body.get("max_tokens")),
        string_list(body.get("stop_sequences")), False)
    if config:
        out["generationConfig"] = config
    return out, model

# ---------------------------------------------------------------------------
# wire translation: responses
# ---------------------------------------------------------------------------


def num_or_zero(value):
    return value if isinstance(value, (int, float)) and value == value else 0


def openai_json_response(model, text, finish_reason, input_tokens, output_tokens, tool_calls=None):
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


def openai_json_to_anthropic(body, model):
    """A Chat Completions answer as an Anthropic Messages object."""
    choices = body.get("choices") if isinstance(body.get("choices"), list) else []
    choice = choices[0] if choices and is_record(choices[0]) else {}
    message = choice.get("message") if is_record(choice.get("message")) else {}
    text = message.get("content") if isinstance(message.get("content"), str) else text_of(message.get("content"))
    tool_blocks = []
    for call in message.get("tool_calls") or []:
        if not is_record(call) or not is_record(call.get("function")):
            continue
        fn = call["function"]
        tool_blocks.append({"type": "tool_use",
                            "id": str_of(call.get("id")) or "toolu_unknown",
                            "name": str_of(fn.get("name")),
                            "input": args_of(fn.get("arguments"))})
    usage = body.get("usage") if is_record(body.get("usage")) else {}
    return anthropic_json_response(
        model, text, stop_reason_of(choice.get("finish_reason")),
        num_or_zero(usage.get("prompt_tokens")), num_or_zero(usage.get("completion_tokens")),
        tool_blocks)


def anthropic_json_to_openai(body, model):
    """An Anthropic Messages answer as a Chat Completions object."""
    content = body.get("content")
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


def gemini_json_to_openai(body, model):
    text, finish, input_tokens, output_tokens = gemini_text_and_usage(body)
    return openai_json_response(model, text, finish_reason_of(finish),
                                input_tokens, output_tokens)


def gemini_json_to_anthropic(body, model):
    text, finish, input_tokens, output_tokens = gemini_text_and_usage(body)
    stop = {"STOP": "end_turn", "MAX_TOKENS": "max_tokens", "SAFETY": "refusal"}.get(finish, "end_turn")
    return anthropic_json_response(model, text, stop, input_tokens, output_tokens)


def provider_input_tokens(client_api, body):
    """The provider's input-token count for this request, from its own formula:
    1 + floor((len(system) + len(last user text) + len(tool names joined)) / 4).
    The translations preserve those strings exactly, so this matches."""
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
    text = json_compact(data)
    if event is None:
        return "data: %s\n\n" % text
    return "event: %s\ndata: %s\n\n" % (event, text)


def parse_sse_frames(text):
    """[(event, data_string)] for one SSE body."""
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
    chunk = {"id": "chatcmpl-aggregator", "object": "chat.completion.chunk",
             "created": FIXED_CREATED, "model": model,
             "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]}
    if usage is not None:
        chunk["choices"] = []
        chunk["usage"] = usage
    return chunk


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
        return [sse_event("message_start", {
            "type": "message_start",
            "message": {"id": "msg_stream", "type": "message", "role": "assistant",
                        "model": model, "content": [], "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": input_tokens, "output_tokens": 0}}})]

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
        try:
            chunk = json.loads(payload)
        except Exception:
            continue
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
        try:
            data = json.loads(payload)
        except Exception:
            continue
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

    for _event, payload in parse_sse_frames(text):
        try:
            frame = json.loads(payload)
        except Exception:
            continue
        if not is_record(frame):
            continue
        candidates = frame.get("candidates") if isinstance(frame.get("candidates"), list) else []
        cand = candidates[0] if candidates and is_record(candidates[0]) else {}
        content = cand.get("content") if is_record(cand.get("content")) else {}
        piece = ""
        if isinstance(content.get("parts"), list):
            for part in content["parts"]:
                if is_record(part) and isinstance(part.get("text"), str):
                    piece += part["text"]
        if piece:
            out.extend(role_chunk())
            out.append(sse_event(None, openai_chunk(model, {"content": piece})))
        if isinstance(cand.get("finishReason"), str):
            finish = cand["finishReason"]
        meta = frame.get("usageMetadata") if is_record(frame.get("usageMetadata")) else {}
        if meta:
            input_tokens = num_or_zero(meta.get("promptTokenCount"))
            output_tokens = num_or_zero(meta.get("candidatesTokenCount"))
    out.extend(role_chunk())
    out.append(sse_event(None, openai_chunk(model, {}, finish_reason_of(finish or "STOP"))))
    out.append(sse_event(None, openai_chunk(
        model, {}, usage={"prompt_tokens": input_tokens, "completion_tokens": output_tokens,
                          "total_tokens": input_tokens + output_tokens})))
    out.append("data: [DONE]\n\n")
    return "".join(out)


def gemini_sse_to_anthropic(text, model, input_tokens):
    """A Gemini streamGenerateContent SSE stream as the Anthropic event sequence."""
    out = [sse_event("message_start", {
        "type": "message_start",
        "message": {"id": "msg_stream", "type": "message", "role": "assistant",
                    "model": model, "content": [], "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": input_tokens, "output_tokens": 0}}}),
           sse_event("content_block_start", {
               "type": "content_block_start", "index": 0,
               "content_block": {"type": "text", "text": ""}})]
    finish = "STOP"
    output_tokens = 0
    for _event, payload in parse_sse_frames(text):
        try:
            frame = json.loads(payload)
        except Exception:
            continue
        if not is_record(frame):
            continue
        candidates = frame.get("candidates") if isinstance(frame.get("candidates"), list) else []
        cand = candidates[0] if candidates and is_record(candidates[0]) else {}
        content = cand.get("content") if is_record(cand.get("content")) else {}
        piece = ""
        if isinstance(content.get("parts"), list):
            for part in content["parts"]:
                if is_record(part) and isinstance(part.get("text"), str):
                    piece += part["text"]
        if piece:
            out.append(sse_event("content_block_delta", {
                "type": "content_block_delta", "index": 0,
                "delta": {"type": "text_delta", "text": piece}}))
        if isinstance(cand.get("finishReason"), str):
            finish = cand["finishReason"]
        meta = frame.get("usageMetadata") if is_record(frame.get("usageMetadata")) else {}
        if meta:
            output_tokens = num_or_zero(meta.get("candidatesTokenCount"))
    stop = {"STOP": "end_turn", "MAX_TOKENS": "max_tokens", "SAFETY": "refusal"}.get(finish, "end_turn")
    out.append(sse_event("content_block_stop", {"type": "content_block_stop", "index": 0}))
    out.append(sse_event("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": stop, "stop_sequence": None},
        "usage": {"output_tokens": output_tokens}}))
    out.append(sse_event("message_stop", {"type": "message_stop"}))
    return "".join(out)

# ---------------------------------------------------------------------------
# routing and serving
# ---------------------------------------------------------------------------


def requirements_for(body, client_api):
    """The manifest's row by arrival order, else requirements read off the body."""
    global MANIFEST_INDEX
    manifest = BRIEF.workload["manifest"] if BRIEF.workload is not None else []
    entry = manifest[MANIFEST_INDEX] if MANIFEST_INDEX < len(manifest) else None
    MANIFEST_INDEX += 1
    if entry is not None:
        return entry["requires"]
    return requirements_from_body(body, client_api)


def requirements_from_body(body, client_api):
    if client_api == "openai":
        vision = False
        for m in body.get("messages") or []:
            if not is_record(m) or not isinstance(m.get("content"), list):
                continue
            for block in m["content"]:
                if is_record(block) and block.get("type") in ("image", "image_url"):
                    vision = True
        fmt = body.get("response_format") if is_record(body.get("response_format")) else {}
        return {"tools": isinstance(body.get("tools"), list) and len(body["tools"]) > 0,
                "vision": vision,
                "json": fmt.get("type") in ("json_object", "json_schema")}
    vision = False
    for m in body.get("messages") or []:
        if not is_record(m) or not isinstance(m.get("content"), list):
            continue
        for block in m["content"]:
            if is_record(block) and block.get("type") == "image":
                vision = True
    return {"tools": isinstance(body.get("tools"), list) and len(body["tools"]) > 0,
            "vision": vision, "json": False}


def is_json_mode(client_api, body):
    if client_api != "openai":
        return False
    fmt = body.get("response_format") if is_record(body.get("response_format")) else {}
    return fmt.get("type") in ("json_object", "json_schema")


def candidate_score(cand):
    state = PROVIDER_STATE[cand["provider"].id]
    fails = state["consec_fail"]
    # A single refusal is usually one scheduled fail call, not a dead provider:
    # a small penalty keeps the best model first, while two consecutive
    # refusals near-exclude the provider (outage or spent budget).
    score = cand["rating"] - 5 * fails
    if fails >= 2:
        score -= 100000
    return score


def pick_candidate(requires, client_api, body, tried_providers):
    """Best not-yet-tried provider for this request, with its best model."""
    json_mode = is_json_mode(client_api, body)
    best_per_provider = {}
    for cand in CANDIDATES:
        if json_mode and cand["provider"].wire == "anthropic":
            # An Anthropic-wire provider cannot carry response_format: it would
            # echo json_mode absent and lose the request.
            continue
        if not meets_requirements(requires, cand["caps"]):
            continue
        pid = cand["provider"].id
        if pid not in best_per_provider or cand["rating"] > best_per_provider[pid]["rating"]:
            best_per_provider[pid] = cand
    ranked = sorted(best_per_provider.values(),
                    key=lambda c: (-candidate_score(c), c["provider"].index, c["model"]))
    for cand in ranked:
        if cand["provider"].id not in tried_providers:
            return cand
    return None


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
        if provider.wire == "anthropic":
            headers["anthropic-version"] = ANTHROPIC_VERSION
        if client_api == "anthropic":
            upstream = dict(body)
            upstream["model"] = cand["model"]
            payload = upstream
        else:
            payload = openai_to_anthropic(body, cand["model"])
        return base + "/v1/messages", headers, json_compact(payload).encode("utf-8")
    # gemini wire
    if client_api == "openai":
        payload, _model = openai_to_gemini(body, cand["model"])
    else:
        payload, _model = anthropic_to_gemini(body, cand["model"])
    if stream:
        url = "%s/v1beta/models/%s:streamGenerateContent?alt=sse" % (
            base, urllib.parse.quote(cand["model"], safe=""))
    else:
        url = "%s/v1beta/models/%s:generateContent" % (
            base, urllib.parse.quote(cand["model"], safe=""))
    return url, headers, json_compact(payload).encode("utf-8")


def translate_response(cand, client_api, body, stream, status, data):
    """(ok, content_type, body_bytes): the provider's answer in the client's wire."""
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
            # gemini wire
            if client_api == "openai":
                return True, "text/event-stream", gemini_sse_to_openai(text, model).encode("utf-8")
            return True, "text/event-stream", gemini_sse_to_anthropic(
                text, model, provider_input_tokens(client_api, body)).encode("utf-8")
        except Exception as err:
            log("stream translation failed for %s/%s: %s" % (provider.id, model, err))
            return False, None, None
    try:
        answer = json.loads(data.decode("utf-8"))
    except Exception:
        return False, None, None
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
    """Best effective score among candidates on providers not yet tried, or None."""
    json_mode = is_json_mode(client_api, body)
    best = None
    for cand in CANDIDATES:
        if cand["provider"].id in tried_providers:
            continue
        if json_mode and cand["provider"].wire == "anthropic":
            continue
        if not meets_requirements(requires, cand["caps"]):
            continue
        score = candidate_score(cand)
        if best is None or score > best:
            best = score
    return best


def serve(client_api, body):
    """Answer one client request. Returns (status, content_type, body_bytes)."""
    stream = body.get("stream") is True
    requires = requirements_for(body, client_api)
    tried_providers = set()
    retried_providers = set()

    def try_once(cand):
        """One call to one provider. Returns (kind, status, content_type, out):
        kind 'ok', 'refused' (HTTP error) or 'bad' (unusable 200 bytes)."""
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
        next call before moving on. Never retry a 400/401/404."""
        if cand["provider"].id in retried_providers:
            return False
        if not (status == 0 or status == 402 or status == 429 or status >= 500):
            return False
        best_alt = best_untried_score(requires, client_api, body, tried_providers)
        return best_alt is None or cand["rating"] > best_alt

    while True:
        cand = pick_candidate(requires, client_api, body, tried_providers)
        if cand is None:
            # No untried provider left: one retry of the best tried provider
            # that has not been retried yet, else the request is unservable.
            json_mode = is_json_mode(client_api, body)
            remaining = [c for c in CANDIDATES
                         if c["provider"].id in tried_providers
                         and c["provider"].id not in retried_providers
                         and meets_requirements(requires, c["caps"])
                         and not (json_mode and c["provider"].wire == "anthropic")]
            if not remaining:
                error = (anthropic_error("overloaded_error", "no free model meets this request")
                         if client_api == "anthropic"
                         else openai_error("no free model meets this request", "no_free_model"))
                return 503, "application/json", json_compact(error).encode("utf-8")
            remaining.sort(key=lambda c: (-candidate_score(c), c["provider"].index, c["model"]))
            cand = remaining[0]
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
        try:
            body = json.loads(raw.decode("utf-8"))
        except Exception:
            self._error(wire, 400, "invalid_request_error",
                        "the request body is not one JSON object")
            return
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
    body = json_compact({"protocol_version": PROTOCOL_VERSION, "port": port,
                         "catalog": CATALOG}).encode("utf-8")
    for attempt in range(60):
        status, _hdrs, data = http_request(
            "POST", ENV_URL + "/register",
            headers={"content-type": "application/json"}, body=body, timeout=5)
        if status == 200:
            try:
                reply = json.loads(data.decode("utf-8"))
            except Exception:
                reply = {}
            if is_record(reply) and isinstance(reply.get("requests"), int):
                log("registered: the workload holds %d requests" % reply["requests"])
            return
        if attempt % 10 == 9:
            log("register attempt %d answered HTTP %s; retrying" % (attempt + 1, status))
        threading.Event().wait(0.5)
    raise RuntimeError("POST %s/register never answered 200" % ENV_URL)


def poll_work(server, done_event):
    while not done_event.is_set():
        status, _hdrs, data = http_request("GET", ENV_URL + "/work", timeout=2)
        if status == 200:
            try:
                work = json.loads(data.decode("utf-8"))
            except Exception:
                work = {}
            if is_record(work) and work.get("done") is True:
                done_event.set()
                server.shutdown()
                return
        done_event.wait(POLL_INTERVAL_S)


def main():
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
