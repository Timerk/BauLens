# syntax=docker/dockerfile:1.7
# BauLens API. glibc base: ONNX Runtime and the CPU PyTorch wheels have no musl builds.
ARG PYTHON_IMAGE=python:3.12-slim-bookworm

FROM ghcr.io/astral-sh/uv:0.12.23 AS uv

FROM ${PYTHON_IMAGE} AS build
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv
WORKDIR /app
# Dependencies first, so source changes do not invalidate the large torch layer.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project
COPY README.md LICENSE ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable

FROM ${PYTHON_IMAGE} AS runtime
RUN groupadd --system --gid 10001 baulens \
    && useradd --system --uid 10001 --gid baulens --create-home baulens
WORKDIR /app
COPY --from=build /app/.venv /app/.venv
COPY configs ./configs
ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib
# Model weights are not baked in: mount ./artifacts read-only at /app/artifacts.
USER baulens
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health', timeout=4)"]
CMD ["baulens", "api", "serve", "--host", "0.0.0.0", "--port", "8000"]
