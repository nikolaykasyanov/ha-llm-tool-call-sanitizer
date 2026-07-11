"""Unit tests for ha-tool-sanitizer (run with: python -m unittest -v).

fastapi/httpx are stubbed at import time so the sanitizing and SSE
transformation logic can be tested without the web dependencies installed.
"""
import json
import sys
import types
import unittest

# --------------------------------------------------------------- stubs
# Must be installed in sys.modules before `import app`.

def _stub(name, **attrs):
    mod = types.ModuleType(name)
    for key, val in attrs.items():
        setattr(mod, key, val)
    sys.modules[name] = mod
    return mod


class _Dummy:
    def __init__(self, *args, **kwargs): pass
    def __call__(self, *args, **kwargs): return self
    def __getattr__(self, _name): return _Dummy()


class _FakeApp:
    def __init__(self, *args, **kwargs): pass
    def _decorator(self, *args, **kwargs):
        def wrap(fn): return fn
        return wrap
    on_event = get = post = api_route = _decorator


_stub("fastapi", FastAPI=_FakeApp, Request=_Dummy)
_stub("fastapi.responses", JSONResponse=_Dummy, Response=_Dummy,
      StreamingResponse=_Dummy)
_stub("httpx", AsyncClient=_Dummy, Timeout=_Dummy, HTTPError=Exception)

import app  # noqa: E402  (import after stubbing is intentional)

FULL = "Кухня, kitchen worktop light, подсветка кухни"


# ------------------------------------------------------------- helpers

def make_completion(args: dict) -> dict:
    """A minimal non-streaming chat.completion with one tool call."""
    return {
        "id": "chatcmpl-1", "object": "chat.completion", "created": 1,
        "model": "gemma-4-26b",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant", "content": None,
                "tool_calls": [{
                    "id": "call_1", "type": "function",
                    "function": {
                        "name": "HassTurnOn",
                        "arguments": json.dumps(args, ensure_ascii=False),
                    },
                }],
            },
            "finish_reason": "tool_calls",
        }],
    }


def args_of(payload: dict) -> dict:
    return json.loads(
        payload["choices"][0]["message"]["tool_calls"][0]
        ["function"]["arguments"]
    )


def sse_line(obj: dict) -> str:
    return "data: " + json.dumps(obj, ensure_ascii=False)


def make_chunk(delta: dict, finish=None, **extra) -> str:
    body = {
        "id": "chatcmpl-9", "object": "chat.completion.chunk",
        "created": 5, "model": "gemma-4-26b",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    body.update(extra)
    return sse_line(body)


def tool_call_delta(index=0, call_id=None, name=None, arguments=None) -> dict:
    tc: dict = {"index": index}
    if call_id is not None:
        tc["id"] = call_id
        tc["type"] = "function"
    fn = {}
    if name is not None:
        fn["name"] = name
    if arguments is not None:
        fn["arguments"] = arguments
    if fn:
        tc["function"] = fn
    return tc


def parse_events(events: list) -> list:
    """Parse emitted SSE strings into (kind, payload) tuples."""
    out = []
    for event in events:
        event = event.strip()
        if not event.startswith("data:"):
            out.append(("raw", event))
            continue
        data = event[len("data:"):].strip()
        out.append(("done", None) if data == "[DONE]"
                   else ("chunk", json.loads(data)))
    return out


# ----------------------------------------------- non-streaming sanitizer

class SanitizeCompletionTests(unittest.TestCase):

    def test_alias_list_reduced_to_first_alias(self):
        payload = app.sanitize_completion(
            make_completion({"name": FULL, "domain": "light"}))
        self.assertEqual(args_of(payload)["name"], "Кухня")

    def test_other_arguments_untouched(self):
        payload = app.sanitize_completion(
            make_completion({"name": FULL, "domain": "light"}))
        self.assertEqual(args_of(payload)["domain"], "light")

    def test_area_sanitized(self):
        payload = app.sanitize_completion(
            make_completion({"name": "Люстра", "area": "Детская, kids' room"}))
        result = args_of(payload)
        self.assertEqual(result["area"], "Детская")
        self.assertEqual(result["name"], "Люстра")

    def test_single_alias_passes_through(self):
        payload = app.sanitize_completion(
            make_completion({"name": "подсветка кухни"}))
        self.assertEqual(args_of(payload)["name"], "подсветка кухни")

    def test_unlisted_keys_keep_commas(self):
        payload = app.sanitize_completion(
            make_completion({"name": "Кухня", "message": "a, b, c"}))
        self.assertEqual(args_of(payload)["message"], "a, b, c")

    def test_nested_structures_sanitized(self):
        payload = app.sanitize_completion(
            make_completion({"targets": [{"name": FULL}], "name": "ok"}))
        self.assertEqual(args_of(payload)["targets"][0]["name"], "Кухня")

    def test_malformed_argument_json_untouched(self):
        payload = make_completion({})
        broken = '{"name": "Кухня, broken'
        payload["choices"][0]["message"]["tool_calls"][0]["function"][
            "arguments"] = broken
        app.sanitize_completion(payload)
        self.assertEqual(
            payload["choices"][0]["message"]["tool_calls"][0]["function"][
                "arguments"], broken)

    def test_reply_without_tool_calls_untouched(self):
        payload = {"choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Готово"},
            "finish_reason": "stop",
        }]}
        app.sanitize_completion(payload)
        self.assertEqual(payload["choices"][0]["message"]["content"], "Готово")

    def test_strip_mode_off_disables_sanitizing(self):
        original = app.STRIP_MODE
        app.STRIP_MODE = "off"
        self.addCleanup(setattr, app, "STRIP_MODE", original)
        payload = app.sanitize_completion(make_completion({"name": FULL}))
        self.assertEqual(args_of(payload)["name"], FULL)


# --------------------------------------------------- streaming transformer

class TransformSSETests(unittest.IsolatedAsyncioTestCase):

    async def transform(self, lines: list) -> list:
        async def line_iter():
            for line in lines:
                yield line
        return [event async for event in app.transform_sse(line_iter())]

    async def test_content_streams_through_live_and_in_order(self):
        events = await self.transform([
            make_chunk({"role": "assistant", "content": "Вклю"}),
            make_chunk({"content": "чаю."}),
            make_chunk({}, finish="stop"),
            "data: [DONE]",
        ])
        parsed = parse_events(events)
        self.assertEqual(len(parsed), 4)
        contents = [c["choices"][0]["delta"].get("content")
                    for kind, c in parsed if kind == "chunk"]
        self.assertEqual([c for c in contents if c], ["Вклю", "чаю."])
        self.assertEqual(parsed[2][1]["choices"][0]["finish_reason"], "stop")
        self.assertEqual(parsed[-1][0], "done")

    async def test_fragmented_tool_call_assembled_and_sanitized(self):
        args_json = json.dumps({"name": FULL, "domain": "light"},
                               ensure_ascii=False)
        frag1, frag2, frag3 = args_json[:10], args_json[10:25], args_json[25:]
        events = await self.transform([
            make_chunk({"role": "assistant", "tool_calls": [
                tool_call_delta(0, "call_7", "HassTurnOn", frag1)]}),
            make_chunk({"tool_calls": [tool_call_delta(0, arguments=frag2)]}),
            make_chunk({"tool_calls": [tool_call_delta(0, arguments=frag3)]}),
            make_chunk({}, finish="tool_calls"),
            "data: [DONE]",
        ])
        parsed = parse_events(events)
        # role chunk, assembled tool call, finish, [DONE]
        self.assertEqual(len(parsed), 4)
        self.assertEqual(parsed[0][1]["choices"][0]["delta"].get("role"),
                         "assistant")
        tc = parsed[1][1]["choices"][0]["delta"]["tool_calls"][0]
        arguments = json.loads(tc["function"]["arguments"])
        self.assertEqual(arguments["name"], "Кухня")
        self.assertEqual(arguments["domain"], "light")
        self.assertEqual(tc["id"], "call_7")
        self.assertEqual(parsed[2][1]["choices"][0]["finish_reason"],
                         "tool_calls")

    async def test_keepalive_comment_forwarded(self):
        events = await self.transform([
            ": keep-alive",
            make_chunk({"role": "assistant", "content": "hi"}),
            make_chunk({}, finish="stop"),
            "data: [DONE]",
        ])
        self.assertEqual(events[0].strip(), ": keep-alive")

    async def test_stream_without_finish_or_done_is_flushed(self):
        events = await self.transform([
            make_chunk({"role": "assistant", "tool_calls": [
                tool_call_delta(0, "call_8", "HassTurnOn",
                                json.dumps({"name": FULL},
                                           ensure_ascii=False))]}),
        ])
        parsed = parse_events(events)
        tc = parsed[1][1]["choices"][0]["delta"]["tool_calls"][0]
        self.assertEqual(json.loads(tc["function"]["arguments"])["name"],
                         "Кухня")
        self.assertEqual(parsed[-1][0], "done")

    async def test_content_then_tool_call_in_same_message(self):
        events = await self.transform([
            make_chunk({"role": "assistant", "content": "Хорошо, "}),
            make_chunk({"tool_calls": [
                tool_call_delta(0, "call_9", "HassTurnOn",
                                json.dumps({"area": "Детская, kids' room"},
                                           ensure_ascii=False))]}),
            make_chunk({}, finish="tool_calls"),
            "data: [DONE]",
        ])
        parsed = parse_events(events)
        self.assertEqual(parsed[0][1]["choices"][0]["delta"]["content"],
                         "Хорошо, ")
        tc = parsed[1][1]["choices"][0]["delta"]["tool_calls"][0]
        self.assertEqual(json.loads(tc["function"]["arguments"])["area"],
                         "Детская")

    async def test_usage_only_chunk_forwarded(self):
        events = await self.transform([
            make_chunk({"role": "assistant", "content": "ok"}),
            make_chunk({}, finish="stop"),
            sse_line({"id": "chatcmpl-9", "object": "chat.completion.chunk",
                      "created": 5, "model": "m", "choices": [],
                      "usage": {"total_tokens": 42}}),
            "data: [DONE]",
        ])
        parsed = parse_events(events)
        usage_chunks = [c for kind, c in parsed
                        if kind == "chunk" and c.get("usage")]
        self.assertEqual(usage_chunks[0]["usage"]["total_tokens"], 42)

    async def test_parallel_tool_calls_keep_separate_indexes(self):
        events = await self.transform([
            make_chunk({"role": "assistant", "tool_calls": [
                tool_call_delta(0, "a", "HassTurnOn",
                                '{"name": "Люстра"}')]}),
            make_chunk({"tool_calls": [
                tool_call_delta(1, "b", "HassTurnOff",
                                json.dumps({"name": FULL},
                                           ensure_ascii=False))]}),
            make_chunk({}, finish="tool_calls"),
            "data: [DONE]",
        ])
        parsed = parse_events(events)
        tool_calls = parsed[1][1]["choices"][0]["delta"]["tool_calls"]
        self.assertEqual(len(tool_calls), 2)
        self.assertEqual(
            json.loads(tool_calls[0]["function"]["arguments"])["name"],
            "Люстра")
        self.assertEqual(
            json.loads(tool_calls[1]["function"]["arguments"])["name"],
            "Кухня")

    async def test_non_sse_line_forwarded_raw(self):
        events = await self.transform([
            "event: ping",
            make_chunk({"role": "assistant", "content": "ok"}),
            make_chunk({}, finish="stop"),
            "data: [DONE]",
        ])
        self.assertEqual(events[0].strip(), "event: ping")


if __name__ == "__main__":
    unittest.main(verbosity=2)
