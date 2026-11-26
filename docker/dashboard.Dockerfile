# syntax=docker/dockerfile:1.7
# BauLens dashboard. It only talks HTTP to the API, so it installs the few packages it
# imports (pinned to the versions in uv.lock) instead of the full ML stack.
ARG PYTHON_IMAGE=python:3.12-slim-bookworm

FROM ghcr.io/astral-sh/uv:0.12.23 AS uv

FROM ${PYTHON_IMAGE} AS build
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /src
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv export --locked --no-dev --no-emit-project --no-hashes --format requirements-txt \
        -o /tmp/locked.txt \
    && uv venv /opt/venv \
    && uv pip install --python /opt/venv/bin/python --constraint /tmp/locked.txt \
        streamlit httpx altair pandas

FROM ${PYTHON_IMAGE} AS runtime
RUN groupadd --system --gid 10001 baulens \
    && useradd --system --uid 10001 --gid baulens --create-home baulens
WORKDIR /app
COPY --from=build /opt/venv /opt/venv
COPY apps/dashboard ./
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    BAULENS_API_URL=http://api:8000
USER baulens
EXPOSE 8501
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4)"]
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
