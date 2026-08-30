#!/bin/sh

# Start one completely new TacoRank production run.
#
# Required environment:
#   DEEPSEEK_API_KEY  Exported in this shell; it is never written or printed.
#
# Optional environment:
#   TACORANK_PYTHON312  Absolute path to Python 3.12.
#   TACORANK_DOCKER     Absolute path to the Docker executable.
#   TACORANK_RUN_ID     New run identity reserved by a trusted launcher.

set -eu

# Keep trusted Python imports from creating ignored bytecode in the official
# submodule. TacoRank's worktree verifier intentionally treats ignored files as
# dirty even though ordinary `git status` does not show them.
PYTHONDONTWRITEBYTECODE=1
export PYTHONDONTWRITEBYTECODE

die() {
    printf '%s\n' "error: $*" >&2
    exit 1
}

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
repo_root=$script_dir
cd "$repo_root"

git_root=$(git rev-parse --show-toplevel 2>/dev/null) || die "not inside a Git repository"
[ "$git_root" = "$repo_root" ] || die "run this script from the TacoRank repository checkout"

prepare_starter_kit() {
    starter_kit="$repo_root/kuairand-starter-kit"
    if [ ! -e "$starter_kit/.git" ]; then
        git submodule update --init --recursive -- kuairand-starter-kit \
            || die "could not initialize kuairand-starter-kit"
    fi

    expected_starter_commit=$(git rev-parse HEAD:kuairand-starter-kit 2>/dev/null) \
        || die "could not resolve the pinned kuairand-starter-kit commit"
    actual_starter_commit=$(git -C "$starter_kit" rev-parse HEAD 2>/dev/null) \
        || die "kuairand-starter-kit is not a usable Git worktree"
    [ "$actual_starter_commit" = "$expected_starter_commit" ] \
        || die "kuairand-starter-kit is not at the pinned commit"

    starter_changes=$(
        git -C "$starter_kit" status --porcelain=v1 --untracked-files=all
    ) || die "could not inspect kuairand-starter-kit"
    if [ -n "$starter_changes" ]; then
        git -C "$starter_kit" status --short --untracked-files=all >&2
        die "kuairand-starter-kit has tracked or untracked changes; preserving them"
    fi

    # TacoRank intentionally counts ignored files as dirty. Python imports can
    # leave bytecode that ordinary `git status` hides, so remove only that
    # reproducible cache rather than cleaning all ignored submodule content.
    git -C "$starter_kit" clean -q -fdX -- __pycache__ '*.pyc' '*.pyo' \
        || die "could not remove ignored Python bytecode from kuairand-starter-kit"

    starter_strict_status=$(
        git -C "$starter_kit" status \
            --porcelain=v1 --untracked-files=all --ignored=matching
    ) || die "could not verify kuairand-starter-kit cleanliness"
    if [ -n "$starter_strict_status" ]; then
        git -C "$starter_kit" status \
            --short --untracked-files=all --ignored=matching >&2
        die "kuairand-starter-kit contains non-bytecode generated files"
    fi
}

prepare_starter_kit

if [ -n "$(git status --porcelain=v1 --untracked-files=no)" ]; then
    git status --short >&2
    die "tracked changes are present; use a clean checkout for live setup"
fi

[ -n "${DEEPSEEK_API_KEY:-}" ] || die "DEEPSEEK_API_KEY must be exported in this shell"

if [ -x "$repo_root/venv/bin/tacorank" ]; then
    tacorank="$repo_root/venv/bin/tacorank"
    python="$repo_root/venv/bin/python"
    tacorank_mode=installed
elif [ -x "$repo_root/.venv/bin/tacorank" ]; then
    tacorank="$repo_root/.venv/bin/tacorank"
    python="$repo_root/.venv/bin/python"
    tacorank_mode=installed
elif [ -x "$repo_root/venv/bin/python" ]; then
    # A virtualenv can have the project dependencies installed without the
    # editable package itself. Run the checked-out source directly in that
    # case instead of requiring a generated console-script wrapper.
    python="$repo_root/venv/bin/python"
    tacorank_mode=source
elif [ -x "$repo_root/.venv/bin/python" ]; then
    python="$repo_root/.venv/bin/python"
    tacorank_mode=source
else
    die "could not find a TacoRank CLI or Python in venv/.venv; install the development environment first"
fi
[ -x "$python" ] || die "could not find the Python executable beside tacorank"

run_tacorank() {
    if [ "$tacorank_mode" = source ]; then
        PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}" \
            "$python" -m tacorank.cli "$@"
    else
        "$tacorank" "$@"
    fi
}

python312=${TACORANK_PYTHON312:-}
if [ -z "$python312" ]; then
    python312=$(command -v python3.12 2>/dev/null || true)
fi
if [ -z "$python312" ] && [ -x "/opt/homebrew/bin/python3.12" ]; then
    python312=/opt/homebrew/bin/python3.12
fi
if [ -z "$python312" ] && [ -x "/usr/local/bin/python3.12" ]; then
    python312=/usr/local/bin/python3.12
fi
[ -n "$python312" ] || die "Python 3.12 was not found; set TACORANK_PYTHON312"

docker_executable=${TACORANK_DOCKER:-}
if [ -z "$docker_executable" ]; then
    docker_executable=$(command -v docker 2>/dev/null || true)
fi
if [ -z "$docker_executable" ] && [ -x "/Applications/Docker.app/Contents/Resources/bin/docker" ]; then
    docker_executable="/Applications/Docker.app/Contents/Resources/bin/docker"
fi
if [ -z "$docker_executable" ] && [ -x "/Applications/Rancher Desktop.app/Contents/Resources/resources/darwin/bin/docker" ]; then
    docker_executable="/Applications/Rancher Desktop.app/Contents/Resources/resources/darwin/bin/docker"
fi
[ -n "$docker_executable" ] || die "Docker was not found; set TACORANK_DOCKER"

# Docker Desktop resolves its credential helper by name. Preserve the user's
# PATH while ensuring helpers beside the selected Docker executable are visible.
docker_bin_dir=$(dirname "$docker_executable")
case ":$PATH:" in
    *":$docker_bin_dir:"*) ;;
    *) PATH="$docker_bin_dir:$PATH"; export PATH ;;
esac

mkdir -p "$repo_root/.tacorank"
lock_dir="$repo_root/.tacorank/live-run.lock"
if ! mkdir "$lock_dir" 2>/dev/null; then
    existing_pid=$(sed -n '1p' "$lock_dir/pid" 2>/dev/null || true)
    if [ -n "$existing_pid" ] && kill -0 "$existing_pid" 2>/dev/null; then
        die "another script-managed live run is already active (pid $existing_pid)"
    fi
    die "live-run lock exists at $lock_dir; inspect it before removing it"
fi

cleanup() {
    rm -f "$lock_dir/pid"
    rmdir "$lock_dir" 2>/dev/null || true
}
trap cleanup 0
trap 'exit 1' 1 2 3 15
printf '%s\n' "$$" > "$lock_dir/pid"

stamp=$(date -u '+%Y%m%dT%H%M%SZ')
requested_run_id=${TACORANK_RUN_ID:-}
if [ -n "$requested_run_id" ]; then
    case "$requested_run_id" in
        [A-Za-z0-9]*) ;;
        *) die "TACORANK_RUN_ID must start with an alphanumeric character" ;;
    esac
    case "$requested_run_id" in
        *[!A-Za-z0-9._-]*|'') die "TACORANK_RUN_ID is invalid" ;;
    esac
    [ "${#requested_run_id}" -le 128 ] || die "TACORANK_RUN_ID is too long"
    run_id=$requested_run_id
    deployment_dir="$repo_root/.tacorank/deployments/$run_id"
    runtime_dir="$(dirname "$repo_root")/.tacorank-runtime/$(basename "$repo_root")-$run_id"
    run_dir="$repo_root/runs/$run_id"
    [ ! -e "$deployment_dir" ] || die "deployment directory already exists for $run_id"
    [ ! -e "$runtime_dir" ] || die "runtime directory already exists for $run_id"
    [ ! -e "$run_dir" ] || die "run directory already exists for $run_id"
else
    counter=0
    while :; do
        if [ "$counter" -eq 0 ]; then
            run_id="run_${stamp}_$$"
        else
            run_id="run_${stamp}_$$_${counter}"
        fi
        deployment_dir="$repo_root/.tacorank/deployments/$run_id"
        runtime_dir="$(dirname "$repo_root")/.tacorank-runtime/$(basename "$repo_root")-$run_id"
        run_dir="$repo_root/runs/$run_id"
        if [ ! -e "$deployment_dir" ] && [ ! -e "$runtime_dir" ] && [ ! -e "$run_dir" ]; then
            break
        fi
        counter=$((counter + 1))
    done
fi

data_dir="$repo_root/KuaiRand-Pure/data"
config="$deployment_dir/run-config.json"
live_config="$deployment_dir/live-adapters.json"

printf '%s\n' "Starting new TacoRank live run: $run_id"
printf '%s\n' "Repository: $repo_root"

run_tacorank setup-live \
    --repository-root "$repo_root" \
    --deployment-dir "$deployment_dir" \
    --runtime-dir "$runtime_dir" \
    --data-dir "$data_dir" \
    --python312 "$python312" \
    --docker "$docker_executable" \
    --run-id "$run_id" \
    --download-data

# Setup must not normally write into the official submodule, but repeat the
# narrow bytecode cleanup before live preflight so a Python import cannot poison
# the worktree cleanliness invariant.
prepare_starter_kit

run_tacorank preflight \
    --config "$config" \
    --live-config "$live_config"

[ ! -e "$repo_root/runs/$run_id/events.jsonl" ] || die "preflight unexpectedly created a ledger"

run_tacorank run \
    --config "$config" \
    --live-config "$live_config"

status_json=$(run_tacorank status --run-id "$run_id" --repository-root "$repo_root")
printf '%s\n' "$status_json"

"$python" -c '
import json
import sys

status = json.loads(sys.argv[1])
if status.get("status") != "finalized" or status.get("phase") != "finalized":
    raise SystemExit(
        "run did not finalize: status=%r phase=%r"
        % (status.get("status"), status.get("phase"))
    )
if not status.get("final_experiment_id"):
    raise SystemExit("run finalized without final_experiment_id")
' "$status_json"

run_tacorank validate-ledger \
    --run-id "$run_id" \
    --repository-root "$repo_root"

run_tacorank rebuild-views \
    --run-id "$run_id" \
    --repository-root "$repo_root"

printf '%s\n' "TacoRank live run completed and validated: $run_id"
