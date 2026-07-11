# ha-tool-sanitizer

A tiny reverse proxy that sits between Home Assistant and an
OpenAI-compatible LLM server (oMLX, Ollama, llama.cpp, vLLM, ...) and
deterministically fixes a common local-LLM failure mode: tool calls whose
`name` / `area` arguments contain the full comma-separated alias list from
the prompt context, e.g.

```
name = "Кухня, kitchen worktop light, подсветка кухни"   ->   name = "Кухня"
```

Home Assistant's intent matcher can't resolve the former; the proxy rewrites
it to the first alias before HA ever sees it. Everything else — other
endpoints, headers, errors — is passed through untouched.

## Quick start

```bash
docker compose up -d --build
```

Then point Home Assistant at the proxy instead of oMLX: in your
OpenAI-compatible conversation integration, change the base URL from
`http://<omlx-host>:8000/v1` to `http://<proxy-host>:8100/v1`. Nothing else
changes — model name, API key, prompt all stay the same.

```
Home Assistant  ->  ha-tool-sanitizer:8100  ->  oMLX:8000
```

## Configuration

| Variable            | Default                            | Meaning                                                        |
|---------------------|------------------------------------|----------------------------------------------------------------|
| `UPSTREAM_BASE_URL` | `http://host.docker.internal:8000` | The real LLM server                                            |
| `SANITIZE_KEYS`     | `name,area,floor`                  | Argument names to sanitize (case-insensitive, nested included) |
| `STRIP_MODE`        | `first`                            | `first` = keep text before the first comma; `off` = disable    |
| `STREAM_MODE`       | `transform`                        | `transform` = live SSE transformation; `buffer` = fetch whole completion, re-emit as SSE |
| `PORT`              | `8100`                             | Listen port                                                    |
| `LOG_LEVEL`         | `INFO`                             | `DEBUG` logs full request/response flow                        |

Every rewrite is logged at INFO level:

```
sanitized HassTurnOn.name: 'Кухня, kitchen worktop light, подсветка кухни' -> 'Кухня'
```

so you can watch `docker logs -f ha-tool-sanitizer` and see exactly how
often the model still misbehaves.

## Verifying

```bash
# unit tests (no dependencies needed; fastapi/httpx are stubbed)
python3 -m unittest -v
```

```bash
# health check
curl http://localhost:8100/healthz

# passthrough check
curl http://localhost:8100/v1/models
```

Then ask Assist to turn on a multi-alias light ("включи подсветку кухни")
and watch the proxy logs.

## Caveats

- **Streaming is transformed selectively, not buffered.** Content deltas
  and keep-alive comments are forwarded in real time; only tool-call deltas
  (whose argument JSON arrives fragmented across chunks) are held back,
  assembled per tool-call index, sanitized, and emitted whole right before
  the finish chunk. This costs no usable latency: a partial tool call
  cannot be executed anyway. If the live transformer ever misbehaves with
  a particular server, set `STREAM_MODE=buffer` to fall back to fetching
  the whole completion and re-emitting it as SSE.
- **Commas inside legitimate names.** If you ever create an HA entity whose
  *actual name* contains a comma, the proxy would truncate it. Rename the
  entity, or remove the relevant key from `SANITIZE_KEYS`.
- **Scope.** Only `/v1/chat/completions` responses are modified. The
  Anthropic-style `/v1/messages` endpoint is passed through untouched; if
  you switch HA to that API, the sanitizer won't apply.
