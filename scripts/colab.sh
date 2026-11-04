#!/usr/bin/env bash
set -euo pipefail
umask 077

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
state_dir="${XDG_CONFIG_HOME:-$HOME/.config}/automotive-recsys/colab"
mkdir -p "$state_dir/gcloud"
chmod 700 "$state_dir" "$state_dir/gcloud"
export CLOUDSDK_CONFIG="$state_dir/gcloud"
export COLAB_CLI_REQUEST_TIMEOUT="${COLAB_CLI_REQUEST_TIMEOUT:-120}"
export COLAB_CLI_ASSIGN_RETRIES="${COLAB_CLI_ASSIGN_RETRIES:-1}"
export COLAB_CLI_RETRY_BACKOFF="${COLAB_CLI_RETRY_BACKOFF:-2}"

if [[ ! -x "$project_dir/.venv/bin/colab" ]]; then
  echo 'Run: uv sync --locked --group colab' >&2
  exit 1
fi

exec "$project_dir/.venv/bin/python" "$project_dir/scripts/colab_safe.py" \
  --auth adc --config "$state_dir/sessions.json" "$@"
