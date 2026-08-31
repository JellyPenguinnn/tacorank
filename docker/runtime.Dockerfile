FROM python:3.12-slim AS trae-tools

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONNOUSERSITE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install --yes --no-install-recommends binutils git \
    && rm -rf /var/lib/apt/lists/*
COPY requirements-trae.txt /opt/trae-build/requirements-trae.txt
RUN python -m pip install --no-cache-dir \
    --requirement /opt/trae-build/requirements-trae.txt
WORKDIR /usr/local/lib/python3.12/site-packages
RUN pyinstaller --clean --noconfirm \
        --name edit_tool \
        --distpath /tmp/trae-edit-dist \
        --workpath /tmp/trae-edit-build \
        --specpath /tmp/trae-edit-spec \
        trae_agent/tools/edit_tool_cli.py \
    && pyinstaller --clean --noconfirm \
        --name json_edit_tool \
        --hidden-import jsonpath_ng \
        --distpath /tmp/trae-json-dist \
        --workpath /tmp/trae-json-build \
        --specpath /tmp/trae-json-spec \
        trae_agent/tools/json_edit_tool_cli.py \
    && rm -rf trae_agent/dist \
    && mkdir trae_agent/dist \
    && cp /tmp/trae-edit-dist/edit_tool/edit_tool trae_agent/dist/edit_tool \
    && cp /tmp/trae-json-dist/json_edit_tool/json_edit_tool trae_agent/dist/json_edit_tool \
    && cp -R /tmp/trae-json-dist/json_edit_tool/_internal trae_agent/dist/_internal \
    && test -x trae_agent/dist/edit_tool \
    && test -x trae_agent/dist/json_edit_tool \
    && test -d trae_agent/dist/_internal \
    && trae_agent/dist/edit_tool --help >/dev/null \
    && trae_agent/dist/json_edit_tool --help >/dev/null

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONNOUSERSITE=1 \
    PYTHONUNBUFFERED=1

# LightGBM links against the OpenMP runtime, which python:3.12-slim omits.
# Without it the wheel installs cleanly and then fails at import with
# "libgomp.so.1: cannot open shared object file", so the candidate would be
# rejected at Gate A's isolated import rather than at install time.
RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/tacorank
COPY pyproject.toml setup.cfg setup.py requirements.txt ./
COPY src ./src
COPY benchmarks ./benchmarks
# requirements.txt was copied in but never installed, so the candidate
# container held only setup.cfg's install_requires (pydantic, PyYAML). Every
# candidate therefore had to train in pure Python, and a patch that reached
# for numpy failed Gate A's isolated import with ModuleNotFoundError. Install
# the declared requirements so the numeric stack the starter kit assumes is
# actually present; the built image id stays pinned per deployment.
RUN python -m pip install --no-cache-dir --requirement requirements.txt \
    && python -m pip install --no-cache-dir .
COPY --from=trae-tools \
    /usr/local/lib/python3.12/site-packages/trae_agent/dist \
    /opt/tacorank-trae-tools

WORKDIR /workspace
ENTRYPOINT ["/usr/local/bin/python3"]
