# 0005 Serve the models with FastAPI, show them in Streamlit, ship both with Compose

Accepted on 2026-11-26.

The strength, crack and report modules need one interface that a person can try and that
other tools can call. The public demo must run without a GPU and, ideally, without a local
language model.

Serve all models from one FastAPI process. Load each model once in the application
lifespan; a missing or tampered artifact marks only that component unavailable, so its
endpoints answer 503 and `/health` says why, while the rest keeps working. Reuse
`baulens.schemas` for request and response bodies and version the routes under `/api/v1`.
Run CPU-bound inference in the threadpool. Validate uploads before any model sees them
(body size, declared type, Pillow verification, pixel limit, EXIF orientation).

Build the dashboard in Streamlit and let it talk to the API over HTTP only. It never imports
model code, so the browser-facing container stays small (no PyTorch), the API stays the
single place where inputs are validated, and every number the dashboard shows has passed
the same contract that other clients get. Streamlit fits a German-language internal tool
with forms, charts and file upload and needs no separate frontend build.

Put the report generator behind an adapter (`baulens.api.report_service`) with an
API-owned response envelope. The dashboard therefore does not depend on the report module's
internal result types, and the API can ship before that module is merged.

Ship the stack with Docker Compose: `api`, `dashboard` and the official `ollama` image with
a named model volume. Model weights are mounted read-only from `./artifacts` rather than
baked into images or committed. A demo override (`docker-compose.demo.yml`) sets
`BAULENS_LLM_MODE=replay` and drops the Ollama service: the API then refuses live
generation with 403 and serves recorded model output, which the dashboard labels as
"Aufgezeichnete Ausgabe – keine Live-Generierung". This keeps a public demo honest and
cheap: no one sees a recorded reply presented as a live one.

Consequences: the API image carries the full ML stack (~2 GB, mostly PyTorch for Grad-CAM;
the package has one dependency set, so Streamlit is installed there too). A public demo
still needs a way to obtain the model weights, which this decision leaves open. Live
reports take tens of seconds on a CPU and block one threadpool worker each.
