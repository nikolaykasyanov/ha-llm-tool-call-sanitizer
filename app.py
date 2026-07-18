"""
ha-tool-sanitizer — a tiny reverse proxy between Home Assistant and an
OpenAI-compatible LLM server (oMLX, Ollama, llama.cpp, vLLM, ...).

It intercepts /v1/chat/completions responses and post-processes tool calls:
if a configured argument (by default: name, area, floor) contains a
comma-separated alias list, only the first alias is kept. This guarantees
Home Assistant's intent matcher receives a single valid name, regardless
of how well the model follows instructions.

Streaming is handled by a selective SSE transformer: content deltas are
forwarded in real time, while tool-call deltas (whose argument JSON arrives
fragmented across chunks) are buffered per tool-call index, assembled,
sanitized, and emitted whole. Buffering tool calls costs no usable latency:
a partial tool call cannot be executed anyway.

Environment variables:
  UPSTREAM_BASE_URL  where the real LLM server lives
                     (default: http://host.docker.internal:8000)
  SANITIZE_KEYS      comma-separated argument names to sanitize, matched
                     case-insensitively and recursively
                     (default: name,area,floor)
  STRIP_MODE         "first" -> keep text before the first comma (default)
                     "off"   -> disable sanitizing (pure passthrough)
  STREAM_MODE        "transform" -> live SSE transformation (default)
                     "buffer"    -> fetch whole completion, re-emit as SSE
  PORT               listen port (default: 8100)
  LOG_LEVEL          DEBUG / INFO / WARNING (default: INFO)
"""

import json
import logging
import os
import time
import uuid

import httpx
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

# ---------------------------------------------------------------- config

UPSTREAM = os.environ.get(
    "UPSTREAM_BASE_URL", "http://host.docker.internal:8000"
).rstrip("/")
SANITIZE_KEYS = {
    k.strip().lower()
    for k in os.environ.get("SANITIZE_KEYS", "name,area,floor").split(",")
    if k.strip()
}
STRIP_MODE = os.environ.get("STRIP_MODE", "first").lower()
STREAM_MODE = os.environ.get("STREAM_MODE", "transform").lower()
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("sanitizer")

_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
    "content-encoding",
}

_TIMEOUT = httpx.Timeout(connect=10.0, read=600.0, write=60.0, pool=10.0)

client: httpx.AsyncClient | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global client
    client = httpx.AsyncClient(base_url=UPSTREAM, timeout=_TIMEOUT)
    log.info("proxying to %s, sanitizing keys %s (strip=%s, stream=%s)",
             UPSTREAM, sorted(SANITIZE_KEYS), STRIP_MODE, STREAM_MODE)
    yield
    if client is not None:
        await client.aclose()

app = FastAPI(lifespan=lifespan)

# ---------------------------------------------------------- sanitizing

def _fix_value(value: str) -> str:
    """Keep only the first comma-separated component of an alias list."""
    first = value.split(",")[0].strip()
    return first or value


def _sanitize_obj(obj, hits: list) -> object:
    """Recursively sanitize configured keys inside parsed tool arguments."""
    if isinstance(obj, dict):
        out = {}
        for key, val in obj.items():
            if (
                key.lower() in SANITIZE_KEYS
                and isinstance(val, str)
                and "," in val
            ):
                fixed = _fix_value(val)
                if fixed != val:
                    hits.append((key, val, fixed))
                    val = fixed
            else:
                val = _sanitize_obj(val, hits)
            out[key] = val
        return out
    if isinstance(obj, list):
        return [_sanitize_obj(item, hits) for item in obj]
    return obj


def sanitize_arguments(tool_name, raw_args):
    """Sanitize a tool call's JSON argument string. Returns the new string."""
    if STRIP_MODE == "off":
        return raw_args
    if not isinstance(raw_args, str) or "," not in raw_args:
        return raw_args
    try:
        parsed = json.loads(raw_args)
    except (json.JSONDecodeError, ValueError):
        log.warning("tool call %s: arguments are not valid JSON, "
                    "passing through untouched", tool_name)
        return raw_args
    hits: list = []
    parsed = _sanitize_obj(parsed, hits)
    if not hits:
        return raw_args
    for key, old, new in hits:
        log.info("sanitized %s.%s: %r -> %r", tool_name, key, old, new)
    return json.dumps(parsed, ensure_ascii=False)


def sanitize_completion(payload: dict) -> dict:
    """Sanitize tool-call arguments in a full chat.completion payload."""
    for choice in payload.get("choices") or []:
        message = choice.get("message") or {}
        for tool_call in message.get("tool_calls") or []:
            function = tool_call.get("function") or {}
            function["arguments"] = sanitize_arguments(
                function.get("name"), function.get("arguments")
            )
    return payload


# ------------------------------------------------ streaming transformer

def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def _merge_tool_call_delta(buffered: dict, tc_delta: dict) -> None:
    """Merge one streamed tool_calls delta into the per-index buffer.

    OpenAI streaming semantics: `id`, `type` and `function.name` appear on
    the first delta for a given index; `function.arguments` arrives as
    string fragments to be concatenated across deltas.
    """
    idx = tc_delta.get("index", 0)
    slot = buffered.setdefault(idx, {
        "id": None,
        "type": "function",
        "function": {"name": None, "arguments": ""},
    })
    if tc_delta.get("id"):
        slot["id"] = tc_delta["id"]
    if tc_delta.get("type"):
        slot["type"] = tc_delta["type"]
    fn = tc_delta.get("function") or {}
    if fn.get("name"):
        slot["function"]["name"] = fn["name"]
    fragment = fn.get("arguments")
    if fragment:
        slot["function"]["arguments"] += fragment


def _flush_tool_calls(buffered: dict, meta: dict) -> str | None:
    """Assemble, sanitize and emit buffered tool calls as one delta chunk."""
    if not buffered:
        return None
    tool_calls = []
    for idx in sorted(buffered):
        slot = buffered[idx]
        slot["function"]["arguments"] = sanitize_arguments(
            slot["function"].get("name"), slot["function"]["arguments"]
        )
        if slot["id"] is None:
            slot["id"] = f"call_{uuid.uuid4().hex[:12]}"
        tool_calls.append({"index": idx, **slot})
    buffered.clear()
    return _sse({
        "id": meta.get("id") or f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion.chunk",
        "created": meta.get("created") or int(time.time()),
        "model": meta.get("model", ""),
        "choices": [{
            "index": meta.get("choice_index", 0),
            "delta": {"role": "assistant", "tool_calls": tool_calls},
            "finish_reason": None,
        }],
    })


async def transform_sse(line_iter):
    """Transform an upstream SSE line stream.

    Content deltas pass through immediately; tool-call deltas are buffered
    per index and emitted whole (sanitized) right before the finish chunk
    or [DONE]. SSE comments (keep-alives) are forwarded. `line_iter` yields
    decoded lines without trailing newlines.
    """
    buffered: dict = {}
    meta: dict = {}
    done_sent = False

    async for line in line_iter:
        line = line.rstrip("\r")
        if not line:
            continue
        if line.startswith(":"):
            # SSE comment / keep-alive — forward so clients don't time out.
            yield f"{line}\n\n"
            continue
        if not line.startswith("data:"):
            yield f"{line}\n\n"
            continue

        data = line[len("data:"):].strip()
        if data == "[DONE]":
            flush = _flush_tool_calls(buffered, meta)
            if flush:
                yield flush
            yield "data: [DONE]\n\n"
            done_sent = True
            continue

        try:
            chunk = json.loads(data)
        except (json.JSONDecodeError, ValueError):
            yield f"data: {data}\n\n"
            continue

        meta.setdefault("id", chunk.get("id"))
        meta.setdefault("created", chunk.get("created"))
        meta.setdefault("model", chunk.get("model"))

        choices = chunk.get("choices")
        if not choices:
            # e.g. final usage-only chunk — pass through untouched.
            yield _sse(chunk)
            continue

        passthrough_choices = []
        finish_choices = []
        for choice in choices:
            meta["choice_index"] = choice.get("index", 0)
            delta = choice.get("delta") or {}
            tool_deltas = delta.pop("tool_calls", None)
            if tool_deltas:
                for tc_delta in tool_deltas:
                    _merge_tool_call_delta(buffered, tc_delta)
            finish = choice.get("finish_reason")
            has_payload = any(
                delta.get(k) is not None
                for k in ("content", "role", "refusal", "reasoning_content")
            )
            if finish:
                finish_choices.append(choice)
                if has_payload:
                    stripped = dict(choice)
                    stripped["finish_reason"] = None
                    passthrough_choices.append(stripped)
            elif has_payload:
                passthrough_choices.append(choice)

        if passthrough_choices:
            for choice in passthrough_choices:
                choice.get("delta", {}).pop("tool_calls", None)
            yield _sse({**chunk, "choices": passthrough_choices})

        if finish_choices:
            flush = _flush_tool_calls(buffered, meta)
            if flush:
                yield flush
            yield _sse({
                **chunk,
                "choices": [
                    {"index": c.get("index", 0), "delta": {},
                     "finish_reason": c["finish_reason"]}
                    for c in finish_choices
                ],
            })

    # Stream ended without [DONE]: flush anything left and terminate.
    if not done_sent:
        flush = _flush_tool_calls(buffered, meta)
        if flush:
            yield flush
        yield "data: [DONE]\n\n"


# ------------------------------------------------- buffered fallback

def _as_sse(payload: dict):
    """Re-emit a full (non-streaming) completion as SSE (STREAM_MODE=buffer)."""
    completion_id = payload.get("id") or f"chatcmpl-{uuid.uuid4().hex}"
    created = payload.get("created") or int(time.time())
    model = payload.get("model", "")

    def chunk(delta: dict, finish_reason=None, index: int = 0) -> str:
        return _sse({
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {"index": index, "delta": delta, "finish_reason": finish_reason}
            ],
        })

    for choice in payload.get("choices") or [{}]:
        index = choice.get("index", 0)
        message = choice.get("message") or {}
        delta: dict = {"role": message.get("role", "assistant")}
        if message.get("content") is not None:
            delta["content"] = message["content"]
        tool_calls = message.get("tool_calls")
        if tool_calls:
            delta["tool_calls"] = [
                {
                    "index": i,
                    "id": tc.get("id"),
                    "type": tc.get("type", "function"),
                    "function": tc.get("function", {}),
                }
                for i, tc in enumerate(tool_calls)
            ]
        yield chunk(delta, None, index)
        yield chunk({}, choice.get("finish_reason", "stop"), index)

    yield "data: [DONE]\n\n"


# ----------------------------------------------------------- endpoints

@app.get("/healthz")
async def healthz():
    return {"ok": True, "upstream": UPSTREAM, "stream_mode": STREAM_MODE}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    assert client is not None  # initialized in lifespan
    body = await request.json()
    wants_stream = bool(body.get("stream"))
    headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in _HOP_HEADERS
    }

    # ---- live streaming path
    if wants_stream and STREAM_MODE == "transform":
        try:
            upstream_request = client.build_request(
                "POST", "/v1/chat/completions", json=body, headers=headers
            )
            upstream_response = await client.send(upstream_request, stream=True)
        except httpx.HTTPError as exc:
            log.error("upstream error: %s", exc)
            return JSONResponse(
                status_code=502,
                content={"error": {"message": f"upstream error: {exc}"}},
            )

        if upstream_response.status_code != 200:
            content = await upstream_response.aread()
            await upstream_response.aclose()
            return Response(
                content=content,
                status_code=upstream_response.status_code,
                media_type=upstream_response.headers.get(
                    "content-type", "application/json"
                ),
            )

        async def event_stream():
            try:
                async for event in transform_sse(
                    upstream_response.aiter_lines()
                ):
                    yield event
            finally:
                await upstream_response.aclose()

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    # ---- non-streaming path (also STREAM_MODE=buffer fallback)
    body["stream"] = False
    body.pop("stream_options", None)
    try:
        upstream_response = await client.post(
            "/v1/chat/completions", json=body, headers=headers
        )
    except httpx.HTTPError as exc:
        log.error("upstream error: %s", exc)
        return JSONResponse(
            status_code=502,
            content={"error": {"message": f"upstream error: {exc}"}},
        )

    if upstream_response.status_code != 200:
        return Response(
            content=upstream_response.content,
            status_code=upstream_response.status_code,
            media_type=upstream_response.headers.get(
                "content-type", "application/json"
            ),
        )

    try:
        payload = upstream_response.json()
    except ValueError:
        log.warning("upstream returned non-JSON body, passing through")
        return Response(
            content=upstream_response.content,
            status_code=200,
            media_type=upstream_response.headers.get("content-type", ""),
        )

    payload = sanitize_completion(payload)

    if wants_stream:
        return StreamingResponse(_as_sse(payload), media_type="text/event-stream")
    return JSONResponse(content=payload)


@app.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
)
async def passthrough(request: Request, path: str):
    """Transparent passthrough for every other endpoint (/v1/models, ...)."""
    assert client is not None  # initialized in lifespan
    headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in _HOP_HEADERS
    }
    content = await request.body()
    try:
        upstream_request = client.build_request(
            request.method,
            f"/{path}",
            params=request.query_params,
            headers=headers,
            content=content or None,
        )
        upstream_response = await client.send(upstream_request, stream=True)
    except httpx.HTTPError as exc:
        log.error("upstream error on /%s: %s", path, exc)
        return JSONResponse(
            status_code=502,
            content={"error": {"message": f"upstream error: {exc}"}},
        )

    response_headers = {
        k: v for k, v in upstream_response.headers.items()
        if k.lower() not in _HOP_HEADERS
    }

    async def body_iter():
        try:
            async for part in upstream_response.aiter_bytes():
                yield part
        finally:
            await upstream_response.aclose()

    return StreamingResponse(
        body_iter(),
        status_code=upstream_response.status_code,
        headers=response_headers,
    )
