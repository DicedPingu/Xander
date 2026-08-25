wit_bindgen::generate!({
    path: "../../wit",
    world: "arena-bot",
});

use exports::witcraft::arena::bot::{Action, Direction, Guest, Observation, Point};

struct Ambusher;

impl Guest for Ambusher {
    fn name() -> String {
        "Relic Ambusher".into()
    }

    fn decide(state: Observation) -> Action {
        let me_index = state.self_index as usize;
        let me = &state.players[me_index];
        let Some(target) = state.relics.iter().max_by_key(|relic| {
            let my_distance = distance(&me.position, relic);
            let rival_distance = state
                .players
                .iter()
                .enumerate()
                .filter(|(index, _)| *index != me_index)
                .map(|(_, player)| distance(&player.position, relic))
                .min()
                .unwrap_or(u32::MAX);
            (
                i64::from(rival_distance) - i64::from(my_distance),
                -(i64::from(my_distance)),
            )
        }) else {
            return Action::Hold;
        };

        let vertical = if target.y >= me.position.y {
            Direction::South
        } else {
            Direction::North
        };
        let horizontal = if target.x >= me.position.x {
            Direction::East
        } else {
            Direction::West
        };
        [
            vertical,
            horizontal,
            turn_right(vertical),
            turn_left(vertical),
        ]
        .into_iter()
        .find(|direction| open(&state, &me.position, *direction))
        .map(Action::Move)
        .unwrap_or(Action::Hold)
    }
}

fn distance(a: &Point, b: &Point) -> u32 {
    a.x.abs_diff(b.x) + a.y.abs_diff(b.y)
}

fn open(state: &Observation, origin: &Point, direction: Direction) -> bool {
    let candidate = moved(origin, direction, state.width, state.height);
    if same(&candidate, origin) || state.obstacles.iter().any(|point| same(point, &candidate)) {
        return false;
    }
    state.players.iter().enumerate().all(|(index, player)| {
        index == state.self_index as usize || !same(&player.position, &candidate)
    })
}

fn moved(point: &Point, direction: Direction, width: u32, height: u32) -> Point {
    match direction {
        Direction::North if point.y > 0 => Point {
            x: point.x,
            y: point.y - 1,
        },
        Direction::East if point.x + 1 < width => Point {
            x: point.x + 1,
            y: point.y,
        },
        Direction::South if point.y + 1 < height => Point {
            x: point.x,
            y: point.y + 1,
        },
        Direction::West if point.x > 0 => Point {
            x: point.x - 1,
            y: point.y,
        },
        _ => Point {
            x: point.x,
            y: point.y,
        },
    }
}

fn same(a: &Point, b: &Point) -> bool {
    a.x == b.x && a.y == b.y
}

fn turn_left(direction: Direction) -> Direction {
    match direction {
        Direction::North => Direction::West,
        Direction::East => Direction::North,
        Direction::South => Direction::East,
        Direction::West => Direction::South,
    }
}

fn turn_right(direction: Direction) -> Direction {
    match direction {
        Direction::North => Direction::East,
        Direction::East => Direction::South,
        Direction::South => Direction::West,
        Direction::West => Direction::North,
    }
}

export!(Ambusher);
