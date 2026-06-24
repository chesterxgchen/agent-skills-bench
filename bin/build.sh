#!/usr/bin/env bash
set -euo pipefail

# Standalone benchmark build entrypoint. Location-independent: it works from any
# checkout location and only needs to be told where the SDK source checkout lives
# (to build the SDK wheel + agent skills baked into the benchmark image).
#
#   ./bin/build.sh --sdk-repo /path/to/<sdk> [build args...]
#   SDK_REPO=/path/to/<sdk> ./bin/build.sh [build args...]
#
# If neither is given, the harness autodetects an SDK checkout from the current
# working directory; if it still can't find one, the build fails fast.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${BENCHMARK_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# Pull --sdk-repo out of the args and expose it as SDK_REPO; forward the rest.
args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --sdk-repo) export SDK_REPO="$2"; shift 2 ;;
    --sdk-repo=*) export SDK_REPO="${1#*=}"; shift ;;
    *) args+=("$1"); shift ;;
  esac
done

exec python3 -m benchmark.harness.host.build ${args[@]+"${args[@]}"}
