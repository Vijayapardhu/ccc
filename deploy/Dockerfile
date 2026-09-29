# CPU image: control plane, identity service, API, enrollment.
FROM python:3.12-slim AS base

# ffmpeg is a hard runtime dependency, not a convenience: without it there is
# no decode. libgl/libglib are the usual headless-openCV requirement.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir -e ".[services,inference,index,dev]"

COPY sql ./sql
COPY configs ./configs

# Run unprivileged. The container needs no write access to the filesystem.
RUN useradd --create-home --uid 10001 campus && chown -R campus:campus /app
USER campus

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

EXPOSE 8080
CMD ["campus", "api", "--config", "configs/system.yaml"]
