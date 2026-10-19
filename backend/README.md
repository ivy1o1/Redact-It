# RedactIt backend

RedactIt is a local-first, instruction-driven document redaction tool. You upload PDF/DOCX
files and write a plain-English instruction such as *"Remove all student names, phone numbers
and home addresses, but keep grades and course names unchanged."* A local Gemma model reads the
instruction and proposes a structured redaction plan; a deterministic matching engine finds the
concrete candidate spans; you review and approve them; the backend then produces genuinely
sanitised copies (PDF text is deleted from the content stream via PyMuPDF redaction annotations,
not painted over with black rectangles; DOCX text is rewritten at the run level). Originals are
never modified. Everything — model, files, logs — stays on this machine.

**Division of responsibility (model proposes, deterministic code validates).** The deterministic
parser (`app/llm/instruction_parser.py`) owns what the user asked to **keep** and the
keyword→detector mapping, because a false "redact this" destroys data and must never come from a
1B model. The model contributes literal phrases and a natural-language clarification, and can only
select a detector *by name* from a closed registry (`app/matching/patterns.py`) — it can never
inject a regular expression. Every model proposal is validated before it can affect a detection:
the detector name must exist in the registry; the detector must be consistent with the model's own
label for it (this is what stops "grades" from mapping onto `credit_card`); an exclusion is
accepted only if the term literally occurs in the document; and any target you asked to keep is
always dropped. **Nothing is dropped silently** — every rejection is reported in
`interpretation.warnings` and `interpretation.rejected_model_proposals`.

## 1. Requirements and Windows PowerShell setup

Runtime is Windows-only. There is no Ollama, no OpenAI, and no cloud inference of any kind.

Verified on this machine:

- **llama.cpp**: `llama-server.exe` is on PATH at
  `C:\Users\swast\AppData\Local\Microsoft\WinGet\Packages\ggml.llamacpp_Microsoft.Winget.Source_8wekyb3d8bbwe\llama-server.exe`
  (build b11382 / 0.5.0-dev, installed via winget `ggml.llamacpp`).
- **Model file**: `C:/Users/swast/.cache/huggingface/hub/models--ggml-org--gemma-3-1b-it-GGUF/snapshots/f9c28bcd85737ffc5aef028638d3341d49869c27/gemma-3-1b-it-Q4_K_M.gguf`,
  served with alias `gemma-3-1b-it`.
- **Target hardware**: a Windows laptop with 8 GB RAM. The launch flags below fit that budget; a
  Q4_K_M 1B model plus the FastAPI process coexist within 8 GB.
- **Python**: 3.13.6 in `backend/.venv`. **pytest**: 9.

### Step 1 — create the venv and install dependencies

```powershell
cd C:\Users\swast\OneDrive\Documents\RedactIt\backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r requirements-dev.txt
```

### Step 2 — copy the environment template

```powershell
Copy-Item .env.example .env
```

`.env` is git-ignored and must never be committed. It is read by `app/config.py` on load.

### Step 3 — start llama-server (loopback only)

```powershell
$GGUF = "C:/Users/swast/.cache/huggingface/hub/models--ggml-org--gemma-3-1b-it-GGUF/snapshots/f9c28bcd85737ffc5aef028638d3341d49869c27/gemma-3-1b-it-Q4_K_M.gguf"
llama-server -m $GGUF --alias gemma-3-1b-it --host 127.0.0.1 --port 8080 --ctx-size 8192 --parallel 2 --threads 6 --flash-attn on --jinja --no-webui
```

Keep `--host 127.0.0.1`. If the model or port differs, update `LLM_MODEL` / `LLM_BASE_URL` in
`.env` to match.

### Step 4 — verify the model is actually loaded

```powershell
Invoke-RestMethod "http://127.0.0.1:8080/health"
(Invoke-RestMethod "http://127.0.0.1:8080/v1/models").data | Select-Object id
```

`/health` must return `{"status":"ok"}` and `/v1/models` must list `gemma-3-1b-it`.
**A responding port is not proof the model loaded** — a port can answer while the model is still
loading. The `/v1/models` alias check is what confirms the served model name matches `LLM_MODEL`.

### Step 5 — start the API

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Then confirm it is wired to the model:

```powershell
Invoke-RestMethod "http://127.0.0.1:8000/api/v1/health"
Invoke-RestMethod "http://127.0.0.1:8000/api/v1/health/model"   # reports ready and the model name
```

If you would rather not run a model at all, set `INTERPRETATION_MODE=deterministic` (or pass
`interpretation_mode=deterministic` per request) and skip steps 3–4; see §7.

Everything above is the API on its own. A static web interface for it lives in the repository root
and is documented in §11.

## 2. Configuration

Configuration is environment-driven; `.env.example` is the reference. The settings that matter:

| Variable | Default | Meaning |
| --- | --- | --- |
| `HOST` / `PORT` | `127.0.0.1` / `8000` | Where the API listens (keep loopback). |
| `API_PREFIX` | `/api/v1` | Prefix for every non-liveness route. |
| `CORS_ALLOW_ORIGINS` | localhost:3000 / :5173 variants | Frontend origins. |
| `LLM_BASE_URL` | `http://127.0.0.1:8080/v1` | Must include `/v1`; must be loopback. |
| `LLM_MODEL` | `gemma-3-1b-it` | Must match llama-server `--alias`. |
| `LLM_TIMEOUT_SECONDS` / `LLM_CONNECT_TIMEOUT_SECONDS` | `120` / `5` | Read / connect timeouts. |
| `LLM_MAX_PROMPT_CHARS` | `6000` | Max document text ever shown to the model. |
| `LLM_USE_JSON_SCHEMA` | `true` | Grammar-constrained JSON; set `false` if your build rejects it. |
| `INTERPRETATION_MODE` | `model` | `deterministic` skips the model entirely. |
| `LLM_ALLOW_CLOUD_FALLBACK` | `false` | **Must stay `false`** — `true` aborts startup. |
| `LLM_ALLOW_NON_LOOPBACK` | `false` | Only raise this to run llama-server on another local interface. |
| `SESSION_TTL_MINUTES` / `DOWNLOAD_TTL_MINUTES` | `30` / `15` | Session and download-link lifetimes. |
| `MAX_UPLOAD_FILES` / `MAX_FILE_SIZE_BYTES` / `MAX_TOTAL_UPLOAD_BYTES` | `5` / `25 MB` / `60 MB` | Upload limits. |
| `MAX_EXTRACTED_CHARS` | `200000` | Per-document extraction budget before truncation. |
| `REDACTION_REPLACEMENT_TEXT` | `[REDACTED]` | Marker written where DOCX text is removed. |

`app/config.py` fails closed at load if `LLM_ALLOW_CLOUD_FALLBACK=true` or if `LLM_BASE_URL` is a
non-loopback host (unless `LLM_ALLOW_NON_LOOPBACK=true`). There is no cloud option in
`INTERPRETATION_MODE` — the only values are `model` and `deterministic`.

## 3. API workflow

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | Liveness probe (no prefix). |
| GET | `/api/v1/health` | Same liveness under the prefix. |
| GET | `/api/v1/health/model` | Probe the local model; **sends no document content**. |
| POST | `/api/v1/analyze` | Multipart: `files` (PDF/DOCX) + `instruction`, optional `interpretation_mode`. Returns a reviewable plan. |
| GET | `/api/v1/sessions/{id}` | Session status (counts, outputs, expiry). |
| DELETE | `/api/v1/sessions/{id}` | Purge the session and all its files. |
| POST | `/api/v1/sessions/{id}/review` | JSON `approved_detection_ids` / `rejected_detection_ids`. |
| POST | `/api/v1/sessions/{id}/redact` | Apply approved detections; produce sanitised copies. |
| GET | `/api/v1/downloads/{token}` | Download one output by its opaque token. |

Analyze returns `200` with `session_id`, per-document summaries, `interpretation` (targets,
exclusions, warnings, rejected proposals), a `summary` tally, and the full `detections` list with
`matched_text`, page/paragraph location, confidence and surrounding context. Nothing is changed
until you POST the review, and only detections you explicitly approve are applied. Anything you
neither approve nor reject is kept.

### Worked synthetic example (analyze → review → redact → download)

Uses fabricated report-card data only. From `backend/`:

```powershell
# 1. Analyse (multipart). Save the JSON so we can read ids out of it.
curl.exe -s -X POST "http://127.0.0.1:8000/api/v1/analyze" `
  -F "files=@.\report.pdf" `
  -F "instruction=Remove all student names, phone numbers and home addresses, but keep grades and course names unchanged." `
  -o analyze.json

$analysis = Get-Content analyze.json -Raw | ConvertFrom-Json
$session  = $analysis.session_id
$approve  = $analysis.detections | ForEach-Object { $_.detection_id }

# 2. Review: approve the candidate ids you want removed.
$body = @{ approved_detection_ids = $approve; rejected_detection_ids = @() } | ConvertTo-Json
Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:8000/api/v1/sessions/$session/review" `
  -ContentType "application/json" -Body $body

# 3. Redact: build sanitised copies (originals untouched) and verify them.
$redacted = Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:8000/api/v1/sessions/$session/redact"

if ($redacted.residual_text_found) { Write-Warning "Residual text detected - do not distribute." }

# 4. Download each output before its token expires.
foreach ($o in $redacted.outputs) {
  curl.exe -s -o ".\$($o.output_name)" ("http://127.0.0.1:8000" + $o.download_url)
}
```

## 4. Supported detectors and keep clauses

Detectors come from the closed `PATTERN_REGISTRY`; the model may only name one of these.

| Detector | Matches | Notes |
| --- | --- | --- |
| `phone` | NA landline/mobile/fax, optional country code + extension | |
| `email` | RFC-ish email addresses | |
| `ssn` | US dashed `NNN-NN-NNNN`, invalid groups rejected | Spaced/unseparated SSNs are missed. |
| `credit_card` | 13–19 digit numbers that pass Luhn | |
| `date` | Common Western numeric and month-name dates | Broad — course codes can look like dates. |
| `ip_address` | Dotted-quad IPv4 with valid octets | Version numbers like `1.2.3.4` also match. |
| `url` | `http(s)`/`www.` hosts and bare common-TLD domains | |
| `zip_code` | State-prefixed ZIP / ZIP+4 | Bare 5-digit numbers are not treated as ZIPs. |
| `address` | Street number + name + suffix, optional unit/city/state/ZIP | Needs a numeric street number; `2 Dr` can false-positive. |
| `person_name` | Label-introduced names (higher confidence) + a loose two-to-three capitalised-word heuristic | The least reliable detector — review these carefully. |

**Keep/exclusion behaviour.** Keep clauses are decided by the deterministic parser, never the
model. An exclusion suppresses any candidate that overlaps its protected text, and the number
suppressed is reported in `summary.suppressed_by_exclusion`. Exclusion protection is **bounded**:
an exclusion extends past the matched term only up to `MAX_CLAUSE_EXTENSION` (120) characters and
stops at the next `Label:` boundary and at the end of the extracted line, so a common word in a
keep clause cannot protect an arbitrary amount of surrounding text. When a term you asked to keep
would otherwise be redacted, the proposal is refused and reported rather than applied silently.

## 5. Local-model privacy guarantees

- **Loopback enforced twice**: at config load (`config.py` rejects non-loopback `LLM_BASE_URL`)
  and again immediately before every request (`app/llm/_loopback.py::assert_local_target`, called
  in both `health()` and `chat_json()`). Hostnames are resolved and must be loopback.
- **Bounded prompt input**: at most `LLM_MAX_PROMPT_CHARS` (default 6000) characters of document
  text are ever shown to the model; matching always runs over the full text locally, and any
  truncation is reported in `warnings`.
- **No cloud fallback**: the only backends are local (`model`) and no-model (`deterministic`).
  Enabling cloud fallback aborts startup.
- **No telemetry**: nothing in `app/` sends analytics or traces; only llama-server's OpenAI-compatible
  `/health`, `/v1/models` and `/v1/chat/completions` on loopback are contacted.
- **Content never leaves the machine**: document bytes, extracted text and instructions go only to
  the configured local server.
- **Log scrubbing**: `SensitiveContentFilter` replaces known-sensitive keys (instruction, prompt,
  completion, matched_text, api_key, …) and any run of 160+ free-text characters with
  `[redacted-from-log]`. Validation errors are sanitised so the submitted instruction never echoes
  back, and 500 tracebacks are logged but never returned.
- **Opaque ids**: session ids (`sess_`), download tokens (`dl_`) and detection ids (`det_`) use
  `secrets.token_urlsafe` / SHA-256, so they are not guessable or enumerable.
- **TTLs and cleanup**: sessions expire after `SESSION_TTL_MINUTES` (30) and downloads after
  `DOWNLOAD_TTL_MINUTES` (15); a background loop purges expired sessions, orphaned working
  directories are removed at startup, and everything is purged on shutdown.
- **Originals never modified**: uploads live under `uploads/`, generated files under `outputs/`;
  redaction always writes a new `*.redacted.*` copy.

## 6. Limitations — read this before trusting an output

RedactIt is **not a compliance or certification tool and does not guarantee that all sensitive
data was found or removed.** Detection is pattern- and keyword-based: it proposes candidates, it
does not understand documents. It can miss sensitive data written in an unrecognised form and can
propose matches that are not sensitive. Nothing is removed until a human approves it.

- **Scanned / image-only PDFs are refused, not silently passed.** There is **no OCR**; a PDF with
  no text layer raises `SCANNED_DOCUMENT_UNSUPPORTED`. Per `PDF_LIMITATIONS`: text in PDF form
  fields, annotations, comments and attached files is **not extracted or redacted**; scan pixels are
  only blanked where a text-layer match was found; and **PDF redaction is word-granularity** — a
  match covering part of a word removes the whole word.
- **DOCX** (`DOCX_LIMITATIONS`): text in floating text boxes, Word comments, footnotes, endnotes
  and tracked changes is **reported as a warning but not redacted**; field codes are redacted by
  their cached display text only.
- **Verification** (`VERIFICATION_LIMITATIONS`): only the extractable text layer is checked. It
  searches the exact approved strings — a paraphrase, table cell, chart label or scanned image will
  not be found — and content that cannot be read back (scan, corrupt file) is reported as
  `not_verifiable`, never as a false pass. A string that still appears because you did not approve
  that occurrence is reported as "retained by choice", not a failure. If verification finds residual
  text you approved for removal, the response sets `residual_text_found: true` and warns you not to
  distribute the files.

## 7. Error handling contract

Every expected failure returns one JSON envelope (see `app/errors.py`, wired in `app/main.py`):

```json
{ "error": { "code": "MODEL_UNAVAILABLE", "message": "...", "details": { } } }
```

Stable codes and the actionable next step:

| Failure | HTTP / code | Next step |
| --- | --- | --- |
| Model server down / refused / connect timeout | `503 MODEL_UNAVAILABLE` | Start llama-server on 127.0.0.1 and confirm `/health` returns ok, then retry. No cloud fallback exists. |
| Server up but model still loading (`/health` 503) | `503 MODEL_LOADING` | Wait for `/health` to report ok and retry. |
| Server responds but not with `LLM_MODEL` (404) | `503 MODEL_NOT_LOADED` | Check the llama-server `--alias`, or set `LLM_MODEL` to match. |
| Server rejects auth (401/403) | `502 MODEL_AUTH_ERROR` | If llama-server was started with `--api-key`, set `LLM_API_KEY` to the same value. |
| Read timeout | `504 MODEL_TIMEOUT` | Retry, reduce `LLM_MAX_PROMPT_CHARS`, or raise `LLM_TIMEOUT_SECONDS`. |
| Model returns prose / malformed JSON | `502 MALFORMED_MODEL_RESPONSE` | Retry, or set `INTERPRETATION_MODE=deterministic`. |
| Build rejects `response_format` json_schema | `502 MALFORMED_MODEL_RESPONSE` | Set `LLM_USE_JSON_SCHEMA=false`, or update llama.cpp. |
| Instruction too vague to act on | `422 INSTRUCTION_AMBIGUOUS` | Name the concrete data types to remove and keep. |
| Scanned / encrypted / oversized document | `422 / 413` per `SCANNED_DOCUMENT_UNSUPPORTED`, `ENCRYPTED_DOCUMENT_UNSUPPORTED`, `FILE_TOO_LARGE` | OCR / decrypt / split or reduce the upload. |
| Stale or malformed detection ids at review | `409 STALE_DETECTION_ID` / `422 INVALID_DETECTION_SELECTION` | Re-run `/analyze` and select from the fresh list. |
| Redact without review, or with nothing approved | `409 REVIEW_REQUIRED` / `409 NO_APPROVED_REDACTIONS` | POST a review with at least one approved id first. |
| Expired session / download | `410 SESSION_EXPIRED` / `410 DOWNLOAD_EXPIRED` | Re-run the analysis / redaction to get fresh ids. |

`INTERPRETATION_MODE=deterministic` is the documented way to analyse without the model: it skips
llama-server entirely and uses keyword parsing plus the closed registry only, and says so in the
response. It can also be selected per request via the `interpretation_mode` form field on
`/analyze`.

## 8. Testing

Run from `backend/`:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

All fixtures are generated in-memory from obviously synthetic data (fabricated names, numbers,
addresses); no real document is ever read and nothing is written outside a pytest `tmp_path`. The
default suite runs fully offline: `conftest.py` forces `INTERPRETATION_MODE=deterministic` and
ignores any real `.env`, so no model server is needed.

Tests marked `integration` require a live local llama-server on `127.0.0.1:8080` serving the
`gemma-3-1b-it` alias. `tests/test_model_live.py` is the only such module: it drives the full
analyse -> review -> redact -> download path with real model inference and asserts that the
validated plan still protects kept data. `pytest.ini` deselects them from the default run
(`addopts = ... -m "not integration"`), so a plain `pytest` stays offline; the module also skips
itself if no server answers or the alias is not loaded.

```powershell
.\.venv\Scripts\python.exe -m pytest -m integration        # only model-backed tests
.\.venv\Scripts\python.exe -m pytest -m "not integration"  # the offline suite (the default)
.\.venv\Scripts\python.exe -m pytest                       # same as above
```

The integration run takes a few minutes on an 8 GB laptop because every completion is computed on
CPU.

## 9. Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| API refuses to start: "not a loopback address" | `LLM_BASE_URL` points at a remote host | Point it at `http://127.0.0.1:8080/v1`, or set `LLM_ALLOW_NON_LOOPBACK=true` deliberately. |
| Startup aborts on cloud fallback | `LLM_ALLOW_CLOUD_FALLBACK=true` | Set it back to `false` — cloud inference is unsupported. |
| `MODEL_UNAVAILABLE` on every analyze | llama-server not running / wrong port | Start it per §1 step 3; confirm `GET http://127.0.0.1:8080/health`. |
| Port 8080 already in use | Another llama-server/process holds it | Stop the other process, or change `--port` and `LLM_BASE_URL` together. |
| `MODEL_LOADING` (503) | Model still loading into RAM | Wait for `/health` = ok and `/v1/models` to list the alias. |
| `/v1/models` does not list the model name | llama-server `--alias` ≠ `LLM_MODEL` | Match `--alias gemma-3-1b-it` to `LLM_MODEL`, or change `LLM_MODEL`. |
| `MALFORMED_MODEL_RESPONSE` mentioning `response_format`/`json_schema` | Build lacks constrained-JSON support | Set `LLM_USE_JSON_SCHEMA=false`, or update llama.cpp. |
| `INSTRUCTION_AMBIGUOUS` | No concrete redaction target recognised | Name data types explicitly, e.g. "Remove all student names, phone numbers and home addresses, but keep grades and course names unchanged." |
| Truncation warning on a large PDF | Extraction hit `MAX_EXTRACTED_CHARS` | Raise `MAX_EXTRACTED_CHARS` (8 GB budget!) or split the document. |
| Review rejected: `STALE_DETECTION_ID` | Old ids from a prior analysis | Re-run `/analyze` and use the fresh detection ids. |

## 10. Installed but unused

`sentry-sdk`, `opentelemetry-exporter-otlp-*` and `fastapi-cloud-cli` are present in the venv only
as transitive dependencies of `fastapi-cli`. Nothing in `app/` imports them and no telemetry is
configured. If you want them gone:

```powershell
.\.venv\Scripts\python.exe -m pip uninstall -y sentry-sdk opentelemetry-exporter-otlp-proto-common opentelemetry-exporter-otlp-proto-grpc opentelemetry-exporter-otlp-proto-http fastapi-cloud-cli
```

Decide for yourself whether to run this; removing them is optional and does not affect the app.

## 11. Web UI (static front end)

The repository root holds a dependency-free interface for this API: `index.html`, `styles.css`,
`main.js`, `config.js` and `assets/` (the hero loop and its poster). There is no build step, no
package manager and no framework — open the
page with any static file server while the API is running.

```powershell
# from the repository root, with the API already running on port 8000 (§1 step 5)
python -m http.server 5173 --bind 127.0.0.1
# then open http://127.0.0.1:5173
```

**The model is a separate prerequisite, not an optional extra.** The interface does not probe the
model on load — there is no status readout while the page is idle. If llama-server is not running,
`interpretation_mode=model` requests fail with `MODEL_UNAVAILABLE`, and the result panel then shows
the backend's own hint plus the **Keywords only** fallback. A hosted copy of this page cannot reach a
model running on your laptop — the backend and llama-server have to run on the same machine as each
other, which is what "local-first" means here. `GET /api/v1/health/model` is still there for you to
check by hand (§1 step 4).

### How the UI is configured

`main.js` resolves the API base URL in this order:

1. `?api=<url>` in the page address — for pointing one copy of the UI at a second backend.
2. `window.RedactItConfig.apiBase` in `config.js`.
3. The page's own origin + `/api/v1` — correct when the UI is served by the backend itself.

`config.js` ships with `http://127.0.0.1:8000/api/v1` for the §1 local setup. There are no secrets
in it: model settings, keys and storage paths stay in the backend's environment (§2), and the
browser never learns them. The default CORS allow-list already covers
`http://localhost:5173` / `http://127.0.0.1:5173`; add another origin via `CORS_ALLOW_ORIGINS` if
you serve the page elsewhere.

### Hero background

The hero's background **is** the MotionSites AI Runtime loop, not a reconstruction of it. Four layers
stack inside `.hero__media`, which is `aria-hidden` and has `pointer-events: none`, so assistive
technology and the keyboard never see any of them:

| Layer | What it is |
| --- | --- |
| `.hero__video` | The loop itself — glyph field, rose dome and light wave all come from the footage. |
| `.hero__still` | The loop's own first frame, exported to `assets/hero-loop-poster.jpg`; retired the moment a real frame is playing. |
| `.hero__scrim` | Darkening so the loop's brightest band cannot out-shine the copy, and a landing on black at the section seam. |
| `.hero__grain` | Static 3px dot grid, `mix-blend-mode: overlay`, masked to the lower band to tie the loop to the flat black below it. |

`assets/hero-loop.mp4` (13.8 MB) is served from the repository, so the page normally never contacts
a CDN. `main.js` attaches two `<source>` elements — the local file first, the CloudFront original as
fallback — and the browser walks them in order, so a missing local copy falls through with no manual
error handling. Sources are attached only when `shouldLoadVideo()` passes: not under
`prefers-reduced-motion`, `save-data`, or a 2G-class connection. Because of that gate the 13.8 MB is
never requested by those visitors, and the poster — which is the loop's own opening frame — carries
the whole composition instead. Reduced motion therefore gets a frozen frame of the same design
rather than a different design.

The footage is a saturated magenta that fights every token in the palette, so it is neutralised by a
CSS `filter` (`saturate` / `hue-rotate` / `brightness` / `contrast`) applied identically to the video
and the poster, which makes the handover between them invisible.

Text legibility is measured rather than estimated: the ratios in the `styles.css` §3 comment come
from screenshots of the composited backdrop with the text hidden, at 1440 / 834 / 390 wide. The
headline is the binding constraint — it sits on the dome's rim, the brightest thing the loop shows —
and clears 4.8:1 for white at desktop against a reference headline at roughly 3:1.

The same loop is what the next section is crossfaded from: `.hero::after` guarantees the seam starts
from solid black whatever frame is playing, and `.reference::before` carries a dim rose wash down
from it. Both that wash and `.reference__beam` are clipped by the section's `overflow: hidden`, so
each is masked to fade in rather than being sliced at the boundary.

### What each control calls

| Interface control | Backend call |
| --- | --- |
| **Analyze documents** (files + instruction + mode) | `POST /api/v1/analyze` |
| Per-detection keep/redact, **All redact** / **All keep** | local state, then `POST /api/v1/sessions/{id}/review` |
| **Redact the approved items** | `POST /api/v1/sessions/{id}/redact` |
| **Download** on each output card | `GET /api/v1/downloads/{token}` |
| **Discard and start over** | `DELETE /api/v1/sessions/{id}` |

The UI adds no policy of its own: every detection defaults to **keep**, the review POST sends
exactly the ids you approved, and the residual-text check, notices and limitation list shown per
output come straight from the backend response. Errors render with their machine code
(`MODEL_UNAVAILABLE`, `SESSION_EXPIRED`, `FILE_TOO_LARGE`, …) next to the backend's human message,
and the composer stays usable so you can retry.

### Privacy of the page itself

The browser sends only the files you pick and the instruction you type, and only to the configured
backend origin — never to llama-server on `:8080`. The page loads two third-party resources for the
landing screen (the Inter and Bubbledot web fonts); neither receives document content, and there is
no analytics or telemetry script. The hero loop is same-origin — CloudFront is only a fallback
source and is not requested while `assets/hero-loop.mp4` is present — and it is not fetched at all
under `prefers-reduced-motion`, `save-data` or a 2G-class connection. With JavaScript off the
page still reads completely and a `<noscript>` notice in the interface section explains how to call
the API directly instead.

