# Changes: demo reliability (branch `fix/demo-reliability`)

## Why
- OpenRouter credits were exhausted (paid `meta-llama/llama-3.3-70b-instruct`).
- `retriever.py` created the Pinecone client and index handle at import time,
  which makes a network round-trip before uvicorn can even bind the port.

## What changed
- **LLM provider chain** in `main.py`: `groq` (free tier) first, then
  `openrouter` on a `:free` model. Same OpenAI chat format incl. tool calls, so
  both `/chat` (web UI, booking tool) and `/chat/completions` (Vapi, forwarded
  tools) work unchanged. Old paid setup is one env change away
  (`LLM_PROVIDER=openrouter`, `OPENROUTER_MODEL=meta-llama/llama-3.3-70b-instruct`).
- **Lazy Pinecone**: client + index are created on the first query, not at import.
  Gemini embedding calls retry on 429/5xx.
- **`GET /health`** now returns `{"ok": true}` and touches nothing. `GET /stats`
  shows configured providers and today's quota use.
- **Daily cap** (`DAILY_REQUEST_CAP`, default 200) on both chat endpoints,
  checked *before* the embedding call. Over cap: `/chat` returns 200 with a
  friendly "email pranayreddy672@gmail.com" reply; the Vapi endpoint speaks it.
- **`index.html`**: pings `/health` on load and shows a "Warming up the server…"
  banner with elapsed seconds until it answers; send button disabled meanwhile.
- STT/TTS: not swapped, because this repo has none. Voice calls run through
  Vapi (its own STT/TTS); the web page is text-only.

## Measured (laptop, repo venv)
| Metric | Before | After |
|---|---|---|
| `import main` | 6.7 s (3.3 s of it Pinecone at import) | 1.8 s |
| Process start → first `/health` | not measured separately | 2.4 s |

## Env vars you now need
| Var | Required | Where |
|---|---|---|
| `GROQ_API_KEY` | yes | https://console.groq.com/keys |
| `OPENROUTER_API_KEY` | optional fallback | existing key works for `:free` models (verified live with `minimax/minimax-m2.7:free`: grounded answer + correct booking tool call; Nemotron free leaks reasoning, Gemma/GLM free were rate-limited) |
| `GEMINI_API_KEY` | yes (embeddings) | https://aistudio.google.com/apikey — must start with `AIza` |
| `PINECONE_API_KEY`, `PINECONE_INDEX` | unchanged | |
| `CAL_API_KEY`, `CAL_USERNAME` | unchanged | |
| `DAILY_REQUEST_CAP`, `CONTACT_EMAIL` | optional | |

See `.env.example`.
