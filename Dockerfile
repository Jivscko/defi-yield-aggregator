# ============================================================
# Multi-stage Dockerfile for DeFi Yield Aggregator
# ============================================================
# Stage 1 – Builder: install deps into a virtualenv
# Stage 2 – Runtime: copy only what's needed, run as non-root
# ============================================================

# ---------- Stage 1: Builder ----------
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build-time system deps (none needed for pure-Python, but
# keep this layer for future C-extension wheels)
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

# Create a virtualenv so we can copy it cleanly to the runtime stage
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies first (cache-friendly layer)
COPY pyproject.toml README.md LICENSE ./
COPY src/ src/
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e .


# ---------- Stage 2: Runtime ----------
FROM python:3.12-slim AS runtime

LABEL maintainer="DeFi Yield Aggregator Contributors"
LABEL description="Production image for the DeFi Yield Aggregator CLI"

# Copy the pre-built virtualenv from the builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Create a non-root user
RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

WORKDIR /app

# Copy application source and config
COPY src/ src/
COPY config.yaml ./

# Ensure the non-root user owns the app directory
RUN chown -R appuser:appuser /app

USER appuser

ENTRYPOINT ["defi-yield"]
CMD ["--help"]
