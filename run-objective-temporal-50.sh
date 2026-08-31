#!/bin/sh

set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
TACORANK_RESEARCH_CAMPAIGN="research/campaigns/objective_temporal_50.json"
export TACORANK_RESEARCH_CAMPAIGN

exec "$script_dir/run-new-live.sh"
