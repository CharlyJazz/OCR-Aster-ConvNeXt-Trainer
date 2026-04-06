#!/usr/bin/env bash
# Start publisher and trainer in background (Linux/macOS)
# Usage: bash scripts/start_all.sh configs/training/my_config.yaml

set -e

CONFIG="${1:-}"
if [[ -z "$CONFIG" ]]; then
    echo "Usage: $0 <config_path>"
    echo "Example: $0 configs/training/aster_v2_curriculum.yaml"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "Starting publisher in background..."
ocr-publish --config "$CONFIG" &
PUB_PID=$!

echo "Publisher PID: $PUB_PID — waiting 8 s for Redis prefill..."
sleep 8

echo "Starting trainer..."
ocr-train --config "$CONFIG"

# When trainer exits, stop publisher
kill "$PUB_PID" 2>/dev/null || true
echo "Done."
