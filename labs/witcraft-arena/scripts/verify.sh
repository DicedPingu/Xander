#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

command -v wasm-tools >/dev/null
rustup target list --installed | grep -qx 'wasm32-wasip2'

cargo test
cargo build --target wasm32-wasip2 --release \
  -p witcraft-greedy-bot \
  -p witcraft-ambusher-bot \
  -p witcraft-spinner-fixture \
  -p witcraft-memory-hog-fixture

GREEDY=target/wasm32-wasip2/release/witcraft_greedy_bot.wasm
AMBUSHER=target/wasm32-wasip2/release/witcraft_ambusher_bot.wasm
SPINNER=target/wasm32-wasip2/release/witcraft_spinner_fixture.wasm
MEMORY_HOG=target/wasm32-wasip2/release/witcraft_memory_hog_fixture.wasm

for component in "$GREEDY" "$AMBUSHER" "$SPINNER" "$MEMORY_HOG"; do
  wasm-tools validate "$component"
done

WIT="$(wasm-tools component wit "$GREEDY")"
printf '%s\n' "$WIT" | grep -q 'export witcraft:arena/bot@0.1.0'
if printf '%s\n' "$WIT" | grep -Eq 'wasi:(filesystem|sockets|http)/'; then
  printf 'unexpected filesystem, socket, or HTTP capability import\n' >&2
  exit 1
fi

cargo run --quiet -- \
  --bot "$GREEDY" --bot "$AMBUSHER" \
  --seed 424242 --turns 60 \
  --output artifacts/replay-424242.json
cargo run --quiet -- \
  --bot "$GREEDY" --bot "$AMBUSHER" \
  --seed 424242 --turns 60 \
  --output artifacts/replay-424242-repeat.json
cmp artifacts/replay-424242.json artifacts/replay-424242-repeat.json

cargo run --quiet -- \
  --bot "$GREEDY" --bot "$SPINNER" \
  --seed 7 --turns 3 --fuel 100000 --timeout-ms 25 \
  --fault-policy require --output artifacts/spinner-contained.json
cargo run --quiet -- \
  --bot "$GREEDY" --bot "$MEMORY_HOG" \
  --seed 8 --turns 2 --memory-mb 16 \
  --fault-policy require --output artifacts/memory-hog-contained.json

printf '(component)' | wasm-tools parse -o artifacts/empty-component.wasm -
if cargo run --quiet -- \
  --bot "$GREEDY" --bot artifacts/empty-component.wasm \
  --turns 1 >artifacts/interface-mismatch.log 2>&1; then
  printf 'empty component unexpectedly passed interface validation\n' >&2
  exit 1
fi
grep -q 'arena-bot@0.1.0 component' artifacts/interface-mismatch.log

cargo run --quiet -- \
  --bot "$GREEDY" --bot "$AMBUSHER" \
  --seed 11 --turns 1 --playback --no-clear --delay-ms 0 \
  --output artifacts/playback-smoke.json >artifacts/playback-smoke.log
grep -q 'turn 1' artifacts/playback-smoke.log

printf 'WITcraft Arena verification passed\n'

