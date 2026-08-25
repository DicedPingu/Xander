wit_bindgen::generate!({
    path: "../../wit",
    world: "arena-bot",
});

use exports::witcraft::arena::bot::{Action, Guest, Observation};

struct Spinner;

impl Guest for Spinner {
    fn name() -> String {
        "Infinite Spinner".into()
    }

    fn decide(_: Observation) -> Action {
        loop {
            std::hint::spin_loop();
        }
    }
}

export!(Spinner);
