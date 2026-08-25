wit_bindgen::generate!({
    path: "../../wit",
    world: "arena-bot",
});

use exports::witcraft::arena::bot::{Action, Direction, Guest, Observation, Point};

struct Greedy;

impl Guest for Greedy {
    fn name() -> String {
        "Greedy Gecko".into()
    }

    fn decide(state: Observation) -> Action {
        let me = &state.players[state.self_index as usize];
        let Some(target) = state
            .relics
            .iter()
            .min_by_key(|point| distance(&me.position, point))
        else {
            return Action::Hold;
        };

        let horizontal = if target.x >= me.position.x {
            Direction::East
        } else {
            Direction::West
        };
        let vertical = if target.y >= me.position.y {
            Direction::South
        } else {
            Direction::North
        };
        let directions = if me.position.x.abs_diff(target.x) >= me.position.y.abs_diff(target.y) {
            [
                horizontal,
                vertical,
                turn_left(horizontal),
                turn_right(horizontal),
            ]
        } else {
            [
                vertical,
                horizontal,
                turn_left(vertical),
                turn_right(vertical),
            ]
        };
        directions
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

export!(Greedy);
