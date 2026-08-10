FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10 \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /app

ARG DATAMIND_SEMANTIC_MODEL=BAAI/bge-small-zh-v1.5
ARG DATAMIND_SEMANTIC_MODEL_REVISION=4e17e244a0fb63bfb78fca8fcf95079fcc664f5c
ARG PYTORCH_CPU_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG PYPI_INDEX_URL=https://pypi.org/simple

COPY pyproject.toml ./

RUN python -m pip install --index-url ${PYTORCH_CPU_INDEX_URL} torch
RUN python -m pip install --index-url ${PYPI_INDEX_URL} hatchling \
    && python -c "import subprocess,sys,tomllib; p=tomllib.load(open('pyproject.toml','rb')); d=p['project']['dependencies']+p['project']['optional-dependencies']['semantic']; subprocess.check_call([sys.executable,'-m','pip','install','--index-url','${PYPI_INDEX_URL}',*d])"
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='${DATAMIND_SEMANTIC_MODEL}', revision='${DATAMIND_SEMANTIC_MODEL_REVISION}', local_dir='/opt/datamind/models/bge-small-zh-v1.5')"

COPY README.md prd.md ./
COPY app ./app
COPY alembic.ini ./
COPY migrations ./migrations

RUN python -m pip install --no-deps --no-build-isolation . \
    && python -m app.semantic.download_model \
       --model ${DATAMIND_SEMANTIC_MODEL} \
       --revision ${DATAMIND_SEMANTIC_MODEL_REVISION} \
       --output /opt/datamind/models/bge-small-zh-v1.5 \
       --verify-only \
    && groupadd --gid 10001 datamind \
    && useradd --uid 10001 --gid 10001 --no-create-home --home-dir /nonexistent datamind \
    && mkdir -p /data/datasets \
    && chown -R 10001:10001 /data

# Keep build identity after dependency/model layers so a new source revision does
# not invalidate the expensive, immutable runtime dependencies.
ARG DATAMIND_BUILD_SHA=local
LABEL org.opencontainers.image.revision=${DATAMIND_BUILD_SHA}
ENV DATAMIND_BUILD_SHA=${DATAMIND_BUILD_SHA}

EXPOSE 8000

USER 10001:10001

CMD ["uvicorn", "app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]

FROM runtime AS test

USER root
ARG PYPI_INDEX_URL=https://pypi.org/simple
RUN python -c "import subprocess,sys,tomllib; p=tomllib.load(open('pyproject.toml','rb')); d=p['project']['optional-dependencies']['dev']; subprocess.check_call([sys.executable,'-m','pip','install','--index-url','${PYPI_INDEX_URL}',*d])"
USER 10001:10001

FROM runtime AS production
