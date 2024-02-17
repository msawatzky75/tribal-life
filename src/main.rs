#![deny(clippy::all)]
#![forbid(unsafe_code)]

mod tribal;

use std::time::Instant;
use error_iter::ErrorIter;
use log::{error};
use pixels::{Error, Pixels, SurfaceTexture};
use winit::{
    dpi::LogicalSize,
    event::{Event, WindowEvent},
    event_loop::{EventLoop},
    window::WindowBuilder,
    keyboard::KeyCode,
};
use winit_input_helper::{WinitInputHelper};
use crate::tribal::hexagon::{Point};
use crate::tribal::TribalHexGrid;

const WINDOW_WIDTH: usize = 1900;
const WINDOW_HEIGHT: usize = 1000;
const RENDER_WIDTH: usize = WINDOW_WIDTH;
const RENDER_HEIGHT: usize = WINDOW_HEIGHT;

fn main() -> Result<(), Error> {
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();
    let mut input = WinitInputHelper::new();

    let window = {
        let size = LogicalSize::new(WINDOW_WIDTH as f64, WINDOW_HEIGHT as f64);
        WindowBuilder::new()
            .with_title("Tribal Life")
            .with_inner_size(size)
            .with_min_inner_size(size)
            .build(&event_loop)
            .unwrap()
    };

    let mut pixels = {
        let window_size = window.inner_size();
        let surface_texture = SurfaceTexture::new(window_size.width, window_size.height, &window);
        Pixels::new(RENDER_WIDTH as u32, RENDER_HEIGHT as u32, surface_texture)?
    };


    let dt_size: Point<f64> = Point { x: RENDER_WIDTH as f64, y: RENDER_HEIGHT as f64 };
    // this is a radius
    let cell_size: Point<f64> = Point { x: 10., y: 10. };

    let mut grid = TribalHexGrid::new_random(dt_size, cell_size);

    let mut paused = false;
    grid.draw();


    let _ = event_loop.run(move |event, elwt| {
        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                println!("The close button was pressed; stopping");
                elwt.exit();
            }
            Event::WindowEvent {
                event: WindowEvent::RedrawRequested,
                ..
            } => {
                // Redraw the application.
                //
                // It's preferable for applications that do not render continuously to render in
                // this event rather than in AboutToWait, since rendering in here allows
                // the program to gracefully handle redraws requested by the OS.


                for (dst, &src) in pixels
                    .frame_mut()
                    .chunks_exact_mut(4)
                    .zip(grid.frame().iter())
                {
                    dst[0] = (src >> 16) as u8;
                    dst[1] = (src >> 8) as u8;
                    dst[2] = src as u8;
                    dst[3] = (src >> 24) as u8;
                }

                if let Err(err) = pixels.render() {
                    log_error("pixels.render", err);
                    elwt.exit();
                    return;
                }
            }
            _ => ()
        }

        // For everything else, for let winit_input_helper collect events to build its state.
        // It returns `true` when it is time to update our game state and request a redraw.
        if input.update(&event) {
            // Close events
            if input.key_pressed(KeyCode::Escape) || input.close_requested() {
                elwt.exit();
                return;
            }
            if input.key_pressed(KeyCode::KeyP) {
                paused = !paused;
            }
            if input.key_pressed(KeyCode::KeyR) {
                grid.randomize()
            }
            if input.key_pressed_os(KeyCode::Space) {
                // Space is frame-step, so ensure we're paused
                paused = true;
            }

            // Resize the window
            if let Some(size) = input.window_resized() {
                if let Err(err) = pixels.resize_surface(size.width, size.height) {
                    log_error("pixels.resize_surface", err);
                    elwt.exit();
                    return;
                }
            }
            if !paused || input.key_pressed_os(KeyCode::Space) {
                let pre_update = Instant::now();
                grid.update();
                let post_update = pre_update.elapsed();

                let pre_draw = Instant::now();
                grid.draw();
                let post_draw = pre_draw.elapsed();
                println!("Draw: {:.2?} Update: {:.2?} Total: {:.2?}", post_draw, post_update, post_draw + post_update);
            }

            window.request_redraw();
        }
    });

    return Ok(());
}

fn log_error<E: std::error::Error + 'static>(method_name: &str, err: E) {
    error!("{method_name}() failed: {err}");
    for source in err.sources().skip(1) {
        error!("  Caused by: {source}");
    }
}
