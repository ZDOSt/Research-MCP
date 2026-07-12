# syntax=docker/dockerfile:1.7

ARG PYTHON_IMAGE=python:3.11.15-slim-bookworm
ARG CRAWL4AI_IMAGE=unclecode/crawl4ai:0.9.1

FROM ${PYTHON_IMAGE} AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:${PATH}

COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install -r requirements.txt


FROM ${CRAWL4AI_IMAGE} AS crawl4ai-runtime

USER root
COPY --chown=0:0 --chmod=0444 socks5_client.py /app/socks5_client.py
COPY --chown=0:0 --chmod=0444 crawl4ai_egress_proxy.py /app/egress_proxy.py
USER appuser


FROM ${PYTHON_IMAGE} AS runtime

ARG APP_UID=10001
ARG APP_GID=10001

ENV ARTIFACT_DIR=/data/artifacts \
    HF_HOME=/data/models \
    HOME=/home/app \
    MODEL_CACHE_DIR=/data/models \
    PATH=/opt/venv/bin:${PATH} \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates dumb-init \
    && python -m playwright install --with-deps chromium \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/* \
    && groupadd --gid "${APP_GID}" app \
    && useradd --uid "${APP_UID}" --gid "${APP_GID}" --create-home --shell /usr/sbin/nologin app \
    && mkdir -p /data/artifacts /data/models /home/app/.cache /run/research-pdf /run/research-web \
    && chown -R app:app /app /data /home/app /run/research-pdf /run/research-web \
    && chmod -R a+rX /ms-playwright

COPY --chown=app:app *.py ./

USER app

EXPOSE 8001

ENTRYPOINT ["dumb-init", "--"]
CMD ["python", "mcp_server.py"]
