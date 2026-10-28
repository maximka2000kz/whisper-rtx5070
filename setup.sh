#!/usr/bin/env bash
# Создаёт .venv и ставит зависимости.
#   ./setup.sh                      # быстрый движок (faster-whisper)
#   ./setup.sh --with-transformers  # + резервный движок (torch/transformers)
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python}"

echo "[1/3] Создаю виртуальное окружение .venv"
"$PYTHON" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

echo "[2/3] Обновляю pip"
python -m pip install --upgrade pip

echo "[3/3] Ставлю зависимости"
pip install -r requirements.txt

if [[ "${1:-}" == "--with-transformers" ]]; then
    echo "[+] Ставлю резервный движок (torch cu128 + transformers)"
    pip install torch --index-url https://download.pytorch.org/whl/cu128 \
        || pip install torch
    pip install -r requirements-transformers.txt
fi

echo
echo "Готово. Активируйте окружение и запускайте:"
echo "  source .venv/bin/activate"
echo "  python transcribe.py video.mp4 -o transcript.txt"
