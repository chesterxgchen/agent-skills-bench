#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${BENCHMARK_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# Standalone: locate the SDK source checkout for image builds via --sdk-repo or
# SDK_REPO (the harness autodetects from cwd if neither is given). Strip the flag
# from the args before command dispatch and forward the rest unchanged.
_args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --sdk-repo) export SDK_REPO="$2"; shift 2 ;;
    --sdk-repo=*) export SDK_REPO="${1#*=}"; shift ;;
    *) _args+=("$1"); shift ;;
  esac
done
set -- ${_args[@]+"${_args[@]}"}

usage() {
  cat <<EOF
Usage: $(basename "$0") [COMMAND] --prompt PATH [--training-code PATH] [--results-root PATH] [PATH]

Commands:
  pair             Run paired skills/no-skills benchmark cases. This is the default when COMMAND is omitted.
  scenario         Run a compiled scenario YAML.
  report           Rebuild parser artifacts and scenario reports from captured results.
  interactive      Start an interactive benchmark container.

Examples:
  ./bin/run.sh --prompt /path/to/prompt.txt /path/to/job-folder
  ./bin/run.sh pair --prompt /path/to/prompt.txt --training-code /path/to/job-folder
  ./bin/run.sh pair --agent claude --model MODEL --prompt /path/to/prompt.txt /path/to/job-folder
  ./bin/run.sh pair --agent-home /path/to/agent-home --prompt /path/to/prompt.txt /path/to/job-folder
  ./bin/run.sh pair --no-agent-auth-mount --prompt /path/to/prompt.txt /path/to/job-folder
  ./bin/run.sh pair --prompt /path/to/prompt.txt --results-root /path/to/results /path/to/job-folder
  ./bin/run.sh pair --prompt /path/to/prompt.txt --output-dir /path/to/exact-run-dir /path/to/job-folder
  ./bin/run.sh scenario /path/to/scenario.yaml --output-dir /path/to/exact-run-dir
  ./bin/run.sh report /path/to/existing-run-dir
EOF
}

# On macOS, Claude Code stores its API key in the Keychain rather than a file.
# Extract it so passthrough_env can forward it into the benchmark container.
if [[ -z "${ANTHROPIC_API_KEY:-}" ]] && command -v security &>/dev/null; then
  _keychain_key="$(security find-generic-password -s "Claude Code" -w 2>/dev/null || true)"
  if [[ -n "${_keychain_key}" ]]; then
    export ANTHROPIC_API_KEY="${_keychain_key}"
  fi
  unset _keychain_key
fi

# On Linux, an API-key login leaves ~/.claude/.credentials.json empty and keeps
# the key in ~/.claude.json (primaryApiKey); the keyring is not used. Extract it
# so passthrough_env can forward it into the container, mirroring the macOS path.
# Uses only the Python standard library, so the benchmark needs no extra deps.
if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
  _claude_json="${HOME}/.claude.json"
  if [[ -f "${_claude_json}" ]]; then
    _api_key="$(python3 -c 'import json,sys
try:
    key = json.load(open(sys.argv[1])).get("primaryApiKey") or ""
except Exception:
    key = ""
print(key if key.startswith("sk-ant-") else "")' "${_claude_json}" 2>/dev/null || true)"
    if [[ -n "${_api_key}" ]]; then
      export ANTHROPIC_API_KEY="${_api_key}"
    fi
    unset _api_key
  fi
  unset _claude_json
fi

# A single API key is a complete credential. If we have one, drop any OAuth /
# auth-token vars so a stale value cannot be forwarded into the container and
# override the key (symptom: "401 Invalid bearer token", apiKeySource=none).
# This makes the run immune to a stale token cached in the launching shell.
if [[ -n "${ANTHROPIC_API_KEY:-}" ]]; then
  unset CLAUDE_CODE_OAUTH_TOKEN ANTHROPIC_AUTH_TOKEN
fi

if [[ "$#" -eq 0 ]]; then
  usage
  exit 2
fi

if [[ "$1" == "-h" || "$1" == "--help" || "$1" == "help" ]]; then
  command="$1"
  shift
elif [[ "$1" == -* ]]; then
  command="pair"
else
  command="$1"
  shift
fi

case "${command}" in
  pair)
    exec python3 -m benchmark.harness.host.runner pair "$@"
    ;;
  scenario)
    exec python3 -m benchmark.harness.host.runner scenario "$@"
    ;;
  report)
    exec python3 -m benchmark.harness.host.runner report "$@"
    ;;
  interactive|shell)
    exec python3 -m benchmark.harness.host.runner interactive "$@"
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    echo "Unknown command: ${command}" >&2
    usage >&2
    exit 2
    ;;
esac
