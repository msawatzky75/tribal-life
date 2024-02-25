use cgmath::*;
use winit::event::*;
use winit::dpi::PhysicalPosition;
use instant::Duration;

#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: Matrix4<f32> = Matrix4::new(
	1.0, 0.0, 0.0, 0.0,
	0.0, 1.0, 0.0, 0.0,
	0.0, 0.0, 0.5, 0.5,
	0.0, 0.0, 0.0, 1.0,
);

#[derive(Debug)]
pub struct Camera {
	pub position: Point3<f32>,
	width: u32,
	height: u32,
	znear: f32,
	zfar: f32,
}

impl Camera {
	pub fn new<
		V: Into<Point3<f32>>,
	>(
		position: V,
		width: u32,
		height: u32,
		znear: f32,
		zfar: f32,
	) -> Self {
		Self {
			position: position.into(),
			width: width.into(),
			height: height.into(),
			znear: znear.into(),
			zfar: zfar.into(),
		}
	}

	pub fn resize(&mut self, new_width: u32, new_height: u32) {
		self.width = new_width;
		self.height = new_height;
	}

	pub fn calc_matrix(&self) -> Matrix4<f32> {
		let proj = ortho(
			0.0,
			self.width as f32,
			0.0,
			self.height as f32,
			self.znear,
			self.zfar,
		);
		let target = Point3 { x: self.position.x, y: self.position.y, z: 0. };
		let view = Matrix4::look_at_rh(self.position, target, Vector3::unit_y());

		OPENGL_TO_WGPU_MATRIX * proj * view
	}
}

#[derive(Debug)]
pub struct CameraController {
	amount_left: f32,
	amount_right: f32,
	amount_up: f32,
	amount_down: f32,
	scroll: f32,
	speed: f32,
}

impl CameraController {
	pub fn new(speed: f32) -> Self {
		Self {
			amount_left: 0.0,
			amount_right: 0.0,
			amount_up: 0.0,
			amount_down: 0.0,
			scroll: 0.0,
			speed,
		}
	}

	pub fn process_keyboard(&mut self, key: &KeyEvent) -> bool {
		use winit::keyboard::{KeyCode, PhysicalKey};

		let amount = if key.state == ElementState::Pressed { 1.0 } else { 0.0 };

		match key.physical_key {
			PhysicalKey::Code(KeyCode::KeyW) | PhysicalKey::Code(KeyCode::ArrowUp) => {
				self.amount_up = amount;
				true
			}
			PhysicalKey::Code(KeyCode::KeyS) | PhysicalKey::Code(KeyCode::ArrowDown) => {
				self.amount_down = amount;
				true
			}
			PhysicalKey::Code(KeyCode::KeyA) | PhysicalKey::Code(KeyCode::ArrowLeft) => {
				self.amount_left = amount;
				true
			}
			PhysicalKey::Code(KeyCode::KeyD) | PhysicalKey::Code(KeyCode::ArrowRight) => {
				self.amount_right = amount;
				true
			}
			_ => false,
		}
	}

	pub fn process_mouse(&mut self, mouse_dx: f64, mouse_dy: f64) {
		self.amount_left = mouse_dx as f32;
		self.amount_up = mouse_dy as f32;
	}

	pub fn process_scroll(&mut self, delta: &MouseScrollDelta) {
		self.scroll = -match delta {
			MouseScrollDelta::LineDelta(_, scroll) => *scroll,
			MouseScrollDelta::PixelDelta(PhysicalPosition {
																		 y: scroll,
																		 ..
																	 }) => *scroll as f32,
		};
	}

	pub fn update_camera(&mut self, camera: &mut Camera, dt: Duration) {
		let dt = dt.as_secs_f32();

		// Move forward/backward and left/right
		camera.position.x += (self.amount_right - self.amount_left) * (camera.position.z / 5.) * self.speed * dt;
		camera.position.y += (self.amount_up - self.amount_down) * (camera.position.z / 5.) * self.speed * dt;
		camera.position.z += (self.scroll) * 0.5;

		// // TODO: just use a scale variable
		// let zoom_change = (self.scroll * zoom_factor);
		// if zoom_change != 0.0 {
		// 	let new_height: i32 = (camera.height as f32 * zoom_change) as i32 + camera.height as i32;
		// 	let new_width: i32 = (camera.width as f32 * zoom_change) as i32 + camera.width as i32;
		// 	if new_height > 0 && new_width > 0 {
		// 		camera.height = new_height as u32;
		// 		camera.width = new_width as u32;
		// 	}
		// }

		self.scroll = 0.;
		self.amount_up = 0.;
		self.amount_down = 0.;
		self.amount_left = 0.;
		self.amount_right = 0.;
	}
}
