wit_bindgen::generate!({
    path: "../../wit",
    world: "arena-bot",
});

use exports::witcraft::arena::bot::{Action, Guest, Observation};

struct MemoryHog;

impl Guest for MemoryHog {
    fn name() -> String {
        "Memory Hog".into()
    }

    fn decide(_: Observation) -> Action {
        let mut bytes = vec![0x5a_u8; 128 * 1024 * 1024];
        for offset in (0..bytes.len()).step_by(64 * 1024) {
            bytes[offset] = bytes[offset].wrapping_add(1);
        }
        std::hint::black_box(&bytes);
        Action::Hold
    }
}

export!(MemoryHog);
