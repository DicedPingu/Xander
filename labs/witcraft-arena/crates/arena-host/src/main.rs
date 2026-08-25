use anyhow::{Context, Result, bail};
use clap::{Parser, ValueEnum};
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::thread;
use std::time::Duration;
use wasmtime::component::{Component, Linker, ResourceTable};
use wasmtime::{Config, Engine, Store, StoreLimits, StoreLimitsBuilder};
use wasmtime_wasi::{WasiCtx, WasiCtxBuilder, WasiCtxView, WasiView};
use witcraft_arena_core::{
    Action, Arena, Decision, Direction, MatchConfig, Observation, Point, ReplayEnvelope,
    render_turn,
};

mod bindings {
    wasmtime::component::bindgen!({
        path: "../../wit",
        world: "arena-bot",
    });
}

use bindings::exports::witcraft::arena::bot as guest;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum FaultPolicy {
    Deny,
    Allow,
    Require,
}

#[derive(Debug, Parser)]
#[command(
    name = "witcraft-arena",
    version,
    about = "Run deterministic roguelike tournaments between sandboxed Wasm components"
)]
struct Args {
    /// WebAssembly Component Model bot. Repeat two to four times.
    #[arg(long = "bot", action = clap::ArgAction::Append, required = true)]
    bots: Vec<PathBuf>,

    #[arg(long, default_value_t = 0x5eed_u64)]
    seed: u64,

    #[arg(long, default_value_t = 80)]
    turns: u32,

    #[arg(long, default_value_t = 17)]
    width: u32,

    #[arg(long, default_value_t = 9)]
    height: u32,

    #[arg(long, default_value_t = 5)]
    relics: usize,

    /// Fuel granted to each bot for one decision.
    #[arg(long, default_value_t = 2_000_000)]
    fuel: u64,

    /// Maximum linear-memory size for each bot store.
    #[arg(long, default_value_t = 16)]
    memory_mb: usize,

    /// Wall-clock deadline for each component call.
    #[arg(long, default_value_t = 100)]
    timeout_ms: u64,

    #[arg(long, value_enum, default_value_t = FaultPolicy::Deny)]
    fault_policy: FaultPolicy,

    /// Write the replay envelope to this path.
    #[arg(long)]
    output: Option<PathBuf>,

    /// Render every recorded turn in the terminal.
    #[arg(long)]
    playback: bool,

    #[arg(long, default_value_t = 45)]
    delay_ms: u64,

    /// Emit only the replay envelope as JSON on stdout.
    #[arg(long)]
    json: bool,

    /// Do not clear the terminal between playback frames.
    #[arg(long)]
    no_clear: bool,
}

struct HostState {
    limits: StoreLimits,
    table: ResourceTable,
    wasi: WasiCtx,
}

impl WasiView for HostState {
    fn ctx(&mut self) -> WasiCtxView<'_> {
        WasiCtxView {
            ctx: &mut self.wasi,
            table: &mut self.table,
        }
    }
}

struct BotRuntime {
    name: String,
    source: PathBuf,
    bindings: bindings::ArenaBot,
    store: Store<HostState>,
    fuel_per_call: u64,
    deadline_ticks: u64,
    disabled_after_fault: Option<String>,
}

impl BotRuntime {
    fn load(
        engine: &Engine,
        source: &Path,
        memory_bytes: usize,
        fuel_per_call: u64,
        deadline_ticks: u64,
    ) -> Result<Self> {
        let component = Component::from_file(engine, source)
            .map_err(anyhow::Error::from)
            .with_context(|| format!("failed to load component {}", source.display()))?;
        let mut linker = Linker::new(engine);
        wasmtime_wasi::p2::add_to_linker_sync(&mut linker).map_err(anyhow::Error::from)?;
        let limits = StoreLimitsBuilder::new()
            .memory_size(memory_bytes)
            .table_elements(20_000)
            .instances(32)
            .tables(16)
            .memories(4)
            .trap_on_grow_failure(true)
            .build();
        let mut wasi = WasiCtxBuilder::new();
        wasi.allow_tcp(false)
            .allow_udp(false)
            .allow_ip_name_lookup(false);
        let mut store = Store::new(
            engine,
            HostState {
                limits,
                table: ResourceTable::new(),
                wasi: wasi.build(),
            },
        );
        store.limiter(|state| &mut state.limits);
        store.set_fuel(fuel_per_call)?;
        store.set_epoch_deadline(deadline_ticks);
        let bindings = bindings::ArenaBot::instantiate(&mut store, &component, &linker)
            .map_err(anyhow::Error::from)
            .with_context(|| {
                format!(
                    "{} is not a capability-free witcraft:arena/arena-bot@0.1.0 component",
                    source.display()
                )
            })?;
        let name = bindings
            .witcraft_arena_bot()
            .call_name(&mut store)
            .map_err(anyhow::Error::from)
            .with_context(|| format!("{} failed while reporting its name", source.display()))?;
        Ok(Self {
            name,
            source: source.to_path_buf(),
            bindings,
            store,
            fuel_per_call,
            deadline_ticks,
            disabled_after_fault: None,
        })
    }

    fn decide(&mut self, state: Observation) -> Decision {
        if let Some(reason) = &self.disabled_after_fault {
            return Decision {
                bot: self.name.clone(),
                action: Action::Hold,
                fuel_used: 0,
                error: Some(format!("bot disabled after sandbox fault: {reason}")),
            };
        }
        let wire_state = to_guest_observation(state);
        let setup = self
            .store
            .set_fuel(self.fuel_per_call)
            .map(|_| self.store.set_epoch_deadline(self.deadline_ticks));
        let outcome = setup.and_then(|_| {
            self.bindings
                .witcraft_arena_bot()
                .call_decide(&mut self.store, &wire_state)
        });
        let remaining = self.store.get_fuel().unwrap_or(0);
        match outcome {
            Ok(action) => Decision {
                bot: self.name.clone(),
                action: from_guest_action(action),
                fuel_used: self.fuel_per_call.saturating_sub(remaining),
                error: None,
            },
            Err(error) => {
                let error = format!("{}: {error:#}", self.source.display());
                self.disabled_after_fault = Some(error.clone());
                Decision {
                    bot: self.name.clone(),
                    action: Action::Hold,
                    fuel_used: self.fuel_per_call.saturating_sub(remaining),
                    error: Some(error),
                }
            }
        }
    }
}

struct EpochClock {
    running: Arc<AtomicBool>,
    worker: Option<thread::JoinHandle<()>>,
}

impl EpochClock {
    const TICK_MS: u64 = 5;

    fn start(engine: Engine) -> Self {
        let running = Arc::new(AtomicBool::new(true));
        let thread_flag = Arc::clone(&running);
        let worker = thread::spawn(move || {
            while thread_flag.load(Ordering::Relaxed) {
                thread::sleep(Duration::from_millis(Self::TICK_MS));
                engine.increment_epoch();
            }
        });
        Self {
            running,
            worker: Some(worker),
        }
    }
}

impl Drop for EpochClock {
    fn drop(&mut self) {
        self.running.store(false, Ordering::Relaxed);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    if !(2..=4).contains(&args.bots.len()) {
        bail!("provide between two and four --bot component paths");
    }
    if args.timeout_ms == 0 {
        bail!("--timeout-ms must be greater than zero");
    }

    let mut config = Config::new();
    config.wasm_component_model(true);
    config.consume_fuel(true);
    config.epoch_interruption(true);
    let engine = Engine::new(&config)?;
    let _clock = EpochClock::start(engine.clone());
    let deadline_ticks = args.timeout_ms.div_ceil(EpochClock::TICK_MS).max(1);
    let memory_bytes = args
        .memory_mb
        .checked_mul(1024 * 1024)
        .context("--memory-mb is too large")?;

    let mut bots = args
        .bots
        .iter()
        .map(|path| BotRuntime::load(&engine, path, memory_bytes, args.fuel, deadline_ticks))
        .collect::<Result<Vec<_>>>()?;
    let names = bots.iter().map(|bot| bot.name.clone()).collect();
    let mut arena = Arena::new(
        args.seed,
        names,
        MatchConfig {
            width: args.width,
            height: args.height,
            turns: args.turns,
            relic_count: args.relics,
        },
    )
    .map_err(anyhow::Error::msg)?;

    for _ in 0..arena.config().turns {
        let decisions = bots
            .iter_mut()
            .enumerate()
            .map(|(index, bot)| bot.decide(arena.observation(index)))
            .collect();
        arena.step(decisions).map_err(anyhow::Error::msg)?;
    }
    let envelope = arena.finish();
    let fault_count = fault_count(&envelope);

    if let Some(output) = &args.output {
        if let Some(parent) = output
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
        fs::write(output, format!("{}\n", envelope.pretty_json()))
            .with_context(|| format!("failed to write {}", output.display()))?;
    }

    if args.playback && !args.json {
        playback(&envelope, args.delay_ms, args.no_clear)?;
    }
    if args.json {
        println!("{}", envelope.pretty_json());
    } else {
        print_summary(&envelope, fault_count, args.output.as_deref());
    }

    match args.fault_policy {
        FaultPolicy::Deny if fault_count > 0 => {
            bail!("{fault_count} sandboxed bot call(s) faulted")
        }
        FaultPolicy::Require if fault_count == 0 => bail!("expected at least one sandbox fault"),
        _ => Ok(()),
    }
}

fn to_guest_observation(state: Observation) -> guest::Observation {
    guest::Observation {
        width: state.width,
        height: state.height,
        turn: state.turn,
        self_index: state.self_index,
        players: state
            .players
            .into_iter()
            .map(|player| guest::Player {
                position: to_guest_point(player.position),
                score: player.score,
            })
            .collect(),
        obstacles: state.obstacles.into_iter().map(to_guest_point).collect(),
        relics: state.relics.into_iter().map(to_guest_point).collect(),
    }
}

fn to_guest_point(point: Point) -> guest::Point {
    guest::Point {
        x: point.x,
        y: point.y,
    }
}

fn from_guest_action(action: guest::Action) -> Action {
    match action {
        guest::Action::Move(direction) => Action::Move(match direction {
            guest::Direction::North => Direction::North,
            guest::Direction::East => Direction::East,
            guest::Direction::South => Direction::South,
            guest::Direction::West => Direction::West,
        }),
        guest::Action::Hold => Action::Hold,
    }
}

fn fault_count(envelope: &ReplayEnvelope) -> usize {
    envelope
        .replay
        .turns
        .iter()
        .flat_map(|turn| &turn.decisions)
        .filter(|decision| decision.error.is_some())
        .count()
}

fn playback(envelope: &ReplayEnvelope, delay_ms: u64, no_clear: bool) -> Result<()> {
    for turn in &envelope.replay.turns {
        if !no_clear {
            print!("\x1b[2J\x1b[H");
        }
        println!("{}", render_turn(&envelope.replay, turn));
        io::stdout().flush()?;
        if delay_ms > 0 {
            thread::sleep(Duration::from_millis(delay_ms));
        }
    }
    Ok(())
}

fn print_summary(envelope: &ReplayEnvelope, faults: usize, output: Option<&Path>) {
    let final_players = envelope
        .replay
        .turns
        .last()
        .map(|turn| turn.players_after.as_slice())
        .unwrap_or(&envelope.replay.initial_players);
    println!("WITcraft Arena replay: {}", envelope.sha256);
    for player in final_players {
        println!(
            "  {:<24} score={:<4} faults={}",
            player.name, player.score, player.faults
        );
    }
    match &envelope.replay.winner {
        Some(winner) => println!("winner: {winner}"),
        None => println!("winner: draw"),
    }
    println!("sandbox faults: {faults}");
    if let Some(output) = output {
        println!("replay: {}", output.display());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deadline_rounds_up_to_clock_ticks() {
        assert_eq!(1_u64.div_ceil(EpochClock::TICK_MS).max(1), 1);
        assert_eq!(11_u64.div_ceil(EpochClock::TICK_MS), 3);
    }

    #[test]
    fn cli_rejects_missing_bots() {
        assert!(Args::try_parse_from(["witcraft-arena"]).is_err());
    }
}
