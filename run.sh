#!/usr/bin/env bash
# Удобный запуск: ./run.sh video.mp4 -o transcript.txt
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -d .venv ]]; then
    echo "Нет .venv. Сначала выполните: ./setup.sh" >&2
    exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate
exec python transcribe.py "$@"
