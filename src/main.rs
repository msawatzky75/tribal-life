#![forbid(unsafe_code)]

const WINDOW_WIDTH: f64 = 1900.;
const WINDOW_HEIGHT: f64 = 1000.;

fn main() {
    pollster::block_on(hex_cellular_life::run(WINDOW_WIDTH, WINDOW_HEIGHT));
}
