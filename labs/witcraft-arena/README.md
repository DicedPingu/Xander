# WITcraft Arena

WITcraft Arena is a deterministic grid-roguelike tournament for real WebAssembly Component Model bots. Two Rust strategies export the versioned `witcraft:arena/bot@0.1.0` WIT interface and a native Wasmtime host loads them at runtime, executes simultaneous turns, renders terminal playback, and writes content-addressed replay JSON.

This project deliberately uses native `cargo build --target wasm32-wasip2`; there is no `cargo-component` step and no JavaScript or mock-Wasm fallback.

## Build and play

Prerequisites are Rust 1.94 or newer, the `wasm32-wasip2` target, and `wasm-tools`:

```console
rustup target add wasm32-wasip2
cargo install wasm-tools --locked --version 1.253.0 --root "$HOME/.local"
./scripts/verify.sh
```

Run a visible match after verification:

```console
cargo run --release -- \
  --bot target/wasm32-wasip2/release/witcraft_greedy_bot.wasm \
  --bot target/wasm32-wasip2/release/witcraft_ambusher_bot.wasm \
  --seed 424242 --turns 60 --playback --delay-ms 75 \
  --output artifacts/my-match.json
```

The host accepts two to four repeated `--bot` arguments. Use `--json` for a pure replay envelope, or inspect `--help` for board, fuel, memory, timeout, and fault-policy controls.

## Sandbox and determinism

- Every decision gets a fresh fuel budget and epoch deadline.
- `StoreLimits` caps linear memory, tables, instances, and memories.
- WASI is created with closed standard streams, no arguments, no environment, no preopened directory, no TCP/UDP, and no name lookup.
- The shipped components import no WASI filesystem, socket, or HTTP interfaces.
- Bot traps become recorded `hold` decisions; `deny`, `allow`, and `require` fault policies make expectations executable.
- Obstacles, relics, simultaneous movement, scoring, and respawns use a local SplitMix64 stream. The replay hash covers the complete canonical replay payload.

`fixtures/spinner` proves that an infinite loop is contained. `fixtures/memory-hog` proves that a 128 MiB allocation is stopped by the default 16 MiB cap. The verification script also rejects a valid component that lacks the arena interface and compares repeated match files byte for byte.

## Layout

- `wit/arena.wit`: versioned public bot contract.
- `bots/greedy` and `bots/ambusher`: independent Wasm component strategies.
- `crates/arena-core`: deterministic rules, replay schema, hashing, and renderer.
- `crates/arena-host`: Wasmtime loader, sandbox, CLI, and terminal playback.
- `fixtures`: hostile components used as containment proofs.

## Proven lesson

Rust's native `wasm32-wasip2` output retains basic WASI CLI imports even when a bot does no I/O. A correct least-authority host therefore supplies a default-deny WASI context rather than an empty linker, while separately asserting that bot components do not import filesystem, socket, or HTTP capabilities.

