FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONNOUSERSITE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /opt/tacorank
COPY pyproject.toml setup.cfg setup.py requirements.txt ./
COPY src ./src
COPY benchmarks ./benchmarks
RUN python -m pip install --no-cache-dir .

WORKDIR /workspace
ENTRYPOINT ["/usr/local/bin/python3"]
