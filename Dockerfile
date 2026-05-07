FROM python:3.12-slim AS runtime

# Runtime libs the MuJoCo wheel dynamically loads.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libegl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /uvx /usr/local/bin/

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Layer 1: dependencies only — cached unless lockfile changes.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --frozen --no-dev --no-install-project

# Layer 2: project source + install.
COPY pallet_safety ./pallet_safety
COPY data ./data
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/healthz').status == 200 else 1)"

CMD ["uvicorn", "pallet_safety.service.api:app", "--host", "0.0.0.0", "--port", "8000"]
