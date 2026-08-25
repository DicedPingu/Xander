use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub const REPLAY_SCHEMA: &str = "witcraft.replay/v1";

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
pub struct Point {
    pub x: u32,
    pub y: u32,
}

impl Point {
    fn moved(self, direction: Direction, width: u32, height: u32) -> Self {
        match direction {
            Direction::North if self.y > 0 => Self {
                y: self.y - 1,
                ..self
            },
            Direction::East if self.x + 1 < width => Self {
                x: self.x + 1,
                ..self
            },
            Direction::South if self.y + 1 < height => Self {
                y: self.y + 1,
                ..self
            },
            Direction::West if self.x > 0 => Self {
                x: self.x - 1,
                ..self
            },
            _ => self,
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Direction {
    North,
    East,
    South,
    West,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", content = "direction", rename_all = "snake_case")]
pub enum Action {
    Move(Direction),
    Hold,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PlayerView {
    pub position: Point,
    pub score: u32,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Observation {
    pub width: u32,
    pub height: u32,
    pub turn: u32,
    pub self_index: u32,
    pub players: Vec<PlayerView>,
    pub obstacles: Vec<Point>,
    pub relics: Vec<Point>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Decision {
    pub bot: String,
    pub action: Action,
    pub fuel_used: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PlayerState {
    pub name: String,
    pub position: Point,
    pub score: u32,
    pub faults: u32,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct TurnRecord {
    pub turn: u32,
    pub relics_before: Vec<Point>,
    pub decisions: Vec<Decision>,
    pub players_after: Vec<PlayerState>,
    pub relics_after: Vec<Point>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Replay {
    pub schema: String,
    pub seed: u64,
    pub width: u32,
    pub height: u32,
    pub obstacles: Vec<Point>,
    pub initial_relics: Vec<Point>,
    pub initial_players: Vec<PlayerState>,
    pub turns: Vec<TurnRecord>,
    pub winner: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ReplayEnvelope {
    pub sha256: String,
    pub replay: Replay,
}

impl ReplayEnvelope {
    pub fn new(replay: Replay) -> Self {
        let bytes = serde_json::to_vec(&replay).expect("Replay serialization cannot fail");
        let sha256 = format!("{:x}", Sha256::digest(bytes));
        Self { sha256, replay }
    }

    pub fn pretty_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("Replay serialization cannot fail")
    }
}

#[derive(Clone, Copy, Debug)]
pub struct MatchConfig {
    pub width: u32,
    pub height: u32,
    pub turns: u32,
    pub relic_count: usize,
}

impl Default for MatchConfig {
    fn default() -> Self {
        Self {
            width: 17,
            height: 9,
            turns: 80,
            relic_count: 5,
        }
    }
}

#[derive(Clone, Debug)]
struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn below(&mut self, upper: u32) -> u32 {
        (self.next() % u64::from(upper)) as u32
    }
}

pub struct Arena {
    config: MatchConfig,
    seed: u64,
    rng: SplitMix64,
    obstacles: BTreeSet<Point>,
    relics: BTreeSet<Point>,
    players: Vec<PlayerState>,
    initial_relics: Vec<Point>,
    initial_players: Vec<PlayerState>,
    turns: Vec<TurnRecord>,
}

impl Arena {
    pub fn new(seed: u64, bot_names: Vec<String>, config: MatchConfig) -> Result<Self, String> {
        if bot_names.len() < 2 || bot_names.len() > 4 {
            return Err("WITcraft Arena requires two to four bots".into());
        }
        if config.width < 7 || config.height < 7 {
            return Err("Arena dimensions must both be at least seven".into());
        }
        if config.relic_count == 0 {
            return Err("At least one relic is required".into());
        }

        let starts = [
            Point { x: 0, y: 0 },
            Point {
                x: config.width - 1,
                y: config.height - 1,
            },
            Point {
                x: config.width - 1,
                y: 0,
            },
            Point {
                x: 0,
                y: config.height - 1,
            },
        ];
        let players: Vec<_> = bot_names
            .into_iter()
            .enumerate()
            .map(|(index, name)| PlayerState {
                name,
                position: starts[index],
                score: 0,
                faults: 0,
            })
            .collect();
        let initial_players = players.clone();
        let mut arena = Self {
            config,
            seed,
            rng: SplitMix64::new(seed),
            obstacles: BTreeSet::new(),
            relics: BTreeSet::new(),
            players,
            initial_relics: Vec::new(),
            initial_players,
            turns: Vec::new(),
        };
        arena.generate_obstacles();
        let free_cells = (arena.config.width * arena.config.height) as usize
            - arena.obstacles.len()
            - arena.players.len();
        if arena.config.relic_count > free_cells {
            return Err(format!(
                "Requested {} relics, but only {free_cells} cells are free",
                arena.config.relic_count
            ));
        }
        while arena.relics.len() < arena.config.relic_count {
            arena.spawn_relic();
        }
        arena.initial_relics = arena.relics.iter().copied().collect();
        Ok(arena)
    }

    fn generate_obstacles(&mut self) {
        let desired = (self.config.width * self.config.height / 9) as usize;
        let mut attempts = 0;
        while self.obstacles.len() < desired && attempts < desired * 20 {
            attempts += 1;
            let point = Point {
                x: 1 + self.rng.below(self.config.width - 2),
                y: 1 + self.rng.below(self.config.height - 2),
            };
            if self.players.iter().all(|player| player.position != point) {
                self.obstacles.insert(point);
            }
        }
    }

    fn spawn_relic(&mut self) {
        let attempts = (self.config.width * self.config.height * 2) as usize;
        for _ in 0..attempts {
            let point = Point {
                x: self.rng.below(self.config.width),
                y: self.rng.below(self.config.height),
            };
            if !self.obstacles.contains(&point)
                && !self.relics.contains(&point)
                && self.players.iter().all(|player| player.position != point)
            {
                self.relics.insert(point);
                return;
            }
        }
    }

    pub fn config(&self) -> MatchConfig {
        self.config
    }

    pub fn observation(&self, index: usize) -> Observation {
        Observation {
            width: self.config.width,
            height: self.config.height,
            turn: self.turns.len() as u32,
            self_index: index as u32,
            players: self
                .players
                .iter()
                .map(|player| PlayerView {
                    position: player.position,
                    score: player.score,
                })
                .collect(),
            obstacles: self.obstacles.iter().copied().collect(),
            relics: self.relics.iter().copied().collect(),
        }
    }

    pub fn step(&mut self, decisions: Vec<Decision>) -> Result<(), String> {
        if decisions.len() != self.players.len() {
            return Err("Every bot must produce exactly one decision".into());
        }
        let relics_before = self.relics.iter().copied().collect();
        let occupied: BTreeSet<_> = self.players.iter().map(|player| player.position).collect();
        let mut proposed = Vec::with_capacity(decisions.len());
        for (player, decision) in self.players.iter().zip(&decisions) {
            let candidate = match decision.action {
                Action::Move(direction) => {
                    player
                        .position
                        .moved(direction, self.config.width, self.config.height)
                }
                Action::Hold => player.position,
            };
            let candidate = if self.obstacles.contains(&candidate)
                || (candidate != player.position && occupied.contains(&candidate))
            {
                player.position
            } else {
                candidate
            };
            proposed.push(candidate);
        }

        let mut collisions = BTreeMap::<Point, usize>::new();
        for point in &proposed {
            *collisions.entry(*point).or_default() += 1;
        }
        for (index, player) in self.players.iter_mut().enumerate() {
            if collisions[&proposed[index]] == 1 {
                player.position = proposed[index];
            }
            if decisions[index].error.is_some() {
                player.faults += 1;
            }
        }

        let mut collected = Vec::new();
        for player in &mut self.players {
            if self.relics.remove(&player.position) {
                player.score += 10;
                collected.push(player.position);
            }
        }
        for _ in collected {
            self.spawn_relic();
        }

        self.turns.push(TurnRecord {
            turn: self.turns.len() as u32,
            relics_before,
            decisions,
            players_after: self.players.clone(),
            relics_after: self.relics.iter().copied().collect(),
        });
        Ok(())
    }

    pub fn finish(self) -> ReplayEnvelope {
        let top_score = self
            .players
            .iter()
            .map(|player| player.score)
            .max()
            .unwrap_or(0);
        let leaders: Vec<_> = self
            .players
            .iter()
            .filter(|player| player.score == top_score)
            .collect();
        let winner = (leaders.len() == 1).then(|| leaders[0].name.clone());
        ReplayEnvelope::new(Replay {
            schema: REPLAY_SCHEMA.into(),
            seed: self.seed,
            width: self.config.width,
            height: self.config.height,
            obstacles: self.obstacles.into_iter().collect(),
            initial_relics: self.initial_relics,
            initial_players: self.initial_players,
            turns: self.turns,
            winner,
        })
    }
}

pub fn render_turn(replay: &Replay, turn: &TurnRecord) -> String {
    let mut grid = vec![vec![' '; replay.width as usize]; replay.height as usize];
    for obstacle in &replay.obstacles {
        grid[obstacle.y as usize][obstacle.x as usize] = '#';
    }
    for relic in &turn.relics_after {
        grid[relic.y as usize][relic.x as usize] = '*';
    }
    for (index, player) in turn.players_after.iter().enumerate() {
        grid[player.position.y as usize][player.position.x as usize] =
            char::from_digit((index + 1) as u32, 10).unwrap_or('?');
    }
    let border = format!("+{}+", "-".repeat(replay.width as usize));
    let mut output = format!("turn {}\n{border}\n", turn.turn + 1);
    for row in grid {
        output.push('|');
        output.extend(row);
        output.push_str("|\n");
    }
    output.push_str(&border);
    for (index, player) in turn.players_after.iter().enumerate() {
        output.push_str(&format!(
            "\n{}={} score={} faults={}",
            index + 1,
            player.name,
            player.score,
            player.faults
        ));
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scripted_replay(seed: u64) -> ReplayEnvelope {
        let mut arena = Arena::new(
            seed,
            vec!["north".into(), "south".into()],
            MatchConfig {
                turns: 20,
                ..MatchConfig::default()
            },
        )
        .unwrap();
        for turn in 0..arena.config().turns {
            arena
                .step(vec![
                    Decision {
                        bot: "north".into(),
                        action: if turn % 2 == 0 {
                            Action::Move(Direction::East)
                        } else {
                            Action::Move(Direction::South)
                        },
                        fuel_used: 1,
                        error: None,
                    },
                    Decision {
                        bot: "south".into(),
                        action: Action::Move(Direction::West),
                        fuel_used: 1,
                        error: None,
                    },
                ])
                .unwrap();
        }
        arena.finish()
    }

    #[test]
    fn one_hundred_seeds_reproduce_exactly() {
        for seed in 0..100 {
            assert_eq!(scripted_replay(seed).sha256, scripted_replay(seed).sha256);
        }
        assert_ne!(scripted_replay(42).sha256, scripted_replay(43).sha256);
    }

    #[test]
    fn collision_keeps_both_players_in_place() {
        let mut arena =
            Arena::new(9, vec!["a".into(), "b".into()], MatchConfig::default()).unwrap();
        let before: Vec<_> = (0..2)
            .map(|index| arena.observation(index).players[index].position)
            .collect();
        arena
            .step(vec![
                Decision {
                    bot: "a".into(),
                    action: Action::Hold,
                    fuel_used: 0,
                    error: None,
                },
                Decision {
                    bot: "b".into(),
                    action: Action::Hold,
                    fuel_used: 0,
                    error: None,
                },
            ])
            .unwrap();
        let replay = arena.finish();
        let after: Vec<_> = replay.replay.turns[0]
            .players_after
            .iter()
            .map(|player| player.position)
            .collect();
        assert_eq!(before, after);
    }

    #[test]
    fn replay_hash_matches_serialized_payload() {
        let envelope = scripted_replay(7);
        let bytes = serde_json::to_vec(&envelope.replay).unwrap();
        assert_eq!(envelope.sha256, format!("{:x}", Sha256::digest(bytes)));
    }

    #[test]
    fn impossible_relic_count_is_rejected() {
        let error = Arena::new(
            1,
            vec!["a".into(), "b".into()],
            MatchConfig {
                width: 7,
                height: 7,
                turns: 1,
                relic_count: 100,
            },
        )
        .err()
        .expect("configuration should fail");
        assert!(error.contains("cells are free"));
    }
}
