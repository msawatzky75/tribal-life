mod tribal;
mod texture;
mod camera;
mod vertex;

use crate::vertex::Vertex;
use cgmath::{Point3, Vector3, Vector4};
use wgpu::util::DeviceExt;
use winit::window::{Window, WindowBuilder};
use winit::event::{DeviceEvent, ElementState, Event, KeyEvent, WindowEvent};
use winit::event_loop::EventLoop;
use winit::keyboard::{Key, NamedKey};
use crate::tribal::{hex, TribalHexGrid};

const WIDTH: f32 = 1900.;
const HEIGHT: f32 = 1000.;
const VERTICES: &[Vertex] = &[
	// Vertex2D { position: [-0.0868241, 0.49240386, 0.], color: [0.5, 1.0, 0.5], tex_coords: [0.4131759, 0.00759614] }, // A
	// Vertex2D { position: [-0.49513406, 0.06958647, 0.], color: [0.5, 1.0, 0.5], tex_coords: [0.0048659444, 0.43041354] }, // B
	// Vertex2D { position: [-0.21918549, -0.44939706, 0.], color: [0.5, 0.0, 1.], tex_coords: [0.28081453, 0.949397] }, // C
	// Vertex2D { position: [0.35966998, -0.3473291, 0.], color: [0.5, 0.0, 0.5], tex_coords: [0.85967, 0.84732914] }, // D
	// Vertex2D { position: [0.44147372, 0.2347359, 0.], color: [1., 0.0, 0.], tex_coords: [0.9414737, 0.2652641] }, // E

	// counter-clockwise
	Vertex { position: [-0.3 * (WIDTH), -0.3 * (HEIGHT), 0.], color: [1., 0., 0.] }, // bottom-left
	Vertex { position: [0.3 * (WIDTH), -0.3 * (HEIGHT), 0.], color: [0., 1., 0.] },
	Vertex { position: [0.3 * (WIDTH), 0.3 * (HEIGHT), 0.], color: [0., 0., 1.] },
	Vertex { position: [-0.3 * (WIDTH), 0.3 * (HEIGHT), 0.], color: [1., 1., 0.] },
];

const INDICES: &[u16] = &[
	// 0, 1, 4,
	// 1, 2, 4,
	// 2, 3, 4,
	0, 1, 2,
	0, 2, 3,
	// 0, 2, 1,
	// 0, 3, 2,
];


#[cfg_attr(target_arch = "wasm32", wasm_bindgen(start))]
pub async fn run(width: f64, height: f64) {
	env_logger::init();
	let event_loop = EventLoop::new().unwrap();

	let window = {
		let size = winit::dpi::LogicalSize::new(width, height);
		WindowBuilder::new()
				.with_title("Tribal Life")
				.with_inner_size(size)
				// .with_min_inner_size(size)
				.build(&event_loop)
				.unwrap()
	};

	let mut paused = false;

	let grid = TribalHexGrid::new_random(
		hex::Point { x: width, y: height },
		hex::Point { x: 40., y: 40. },
	);

	let mut state = State::new(&window, grid).await;
	let mut last_render_time = instant::Instant::now();

	let _ = event_loop.run(move |event, elwt| {
		match event {
			Event::DeviceEvent {
				event: DeviceEvent::MouseMotion { delta },
				..
			} => if state.mouse_pressed {
				state.camera_controller.process_mouse(delta.0, delta.1)
			}
			Event::WindowEvent {
				ref event, window_id
			} if window_id == state.window.id() && !state.input(event) => match event {
				#[cfg(not(target_arch = "wasm32"))]
				WindowEvent::CloseRequested
				| WindowEvent::KeyboardInput {
					event: KeyEvent {
						logical_key: Key::Named(NamedKey::Escape),
						state: ElementState::Pressed,
						..
					},
					..
				} => {
					println!("The close button was pressed; stopping");
					elwt.exit();
				}
				WindowEvent::Resized(physical_size) => {
					state.resize(*physical_size);
				}
				WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
					state.resize(winit::dpi::PhysicalSize {
						width: (state.size.width as f64 * *scale_factor) as u32,
						height: (state.size.height as f64 * *scale_factor) as u32,
					});
				}
				WindowEvent::RedrawRequested => {
					let now = instant::Instant::now();
					let dt = now - last_render_time;
					last_render_time = now;
					state.update(dt);
					match state.render() {
						Ok(_) => {}
						// Reconfigure the surface if lost
						Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
						// The system is out of memory, we should probably quit
						Err(wgpu::SurfaceError::OutOfMemory) => elwt.exit(),
						// All other errors (Outdated, Timeout) should be resolved by the next frame
						Err(e) => eprintln!("{:?}", e),
					}
				}
				WindowEvent::KeyboardInput {
					event: KeyEvent {
						logical_key: Key::Named(NamedKey::Space),
						state: ElementState::Pressed,
						..
					},
					..
				} => {
					paused = !paused
				}
				_ => {}
			},
			Event::AboutToWait {} => {
				// RedrawRequested will only trigger once unless we manually
				// request it.
				state.window().request_redraw();
			}
			_ => {}
		}
	});
}

// We need this for Rust to store our data correctly for the shaders
#[repr(C)]
// This is so we can store this in a buffer
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
	view_position: [f32; 4],
	view_proj: [[f32; 4]; 4],
}

impl CameraUniform {
	fn new() -> Self {
		use cgmath::SquareMatrix;
		Self {
			view_position: [0.0; 4],
			view_proj: cgmath::Matrix4::identity().into(),
		}
	}

	fn update_view_proj(&mut self, camera: &camera::Camera) {
		self.view_position = camera.position.to_homogeneous().into();
		self.view_proj = camera.calc_matrix().into();
	}
}

struct State<'window> {
	surface: wgpu::Surface<'window>,
	device: wgpu::Device,
	queue: wgpu::Queue,
	config: wgpu::SurfaceConfiguration,
	size: winit::dpi::PhysicalSize<u32>,
	render_pipeline: wgpu::RenderPipeline,

	vertex_buffer: wgpu::Buffer,
	num_vertices: u32,

	index_buffer: wgpu::Buffer,
	num_indices: u32,

	instance_buffer: wgpu::Buffer,
	instances: Vec<Instance>,

	diffuse_bind_group: wgpu::BindGroup,
	diffuse_texture: texture::Texture,

	camera: camera::Camera,
	camera_controller: camera::CameraController,
	camera_uniform: CameraUniform,
	camera_buffer: wgpu::Buffer,
	camera_bind_group: wgpu::BindGroup,
	mouse_pressed: bool,

	window: &'window Window,

	grid: TribalHexGrid,
}

impl State<'_> {
	// Creating some of the wgpu types requires async code
	async fn new<'window>(window: &'window Window, grid: TribalHexGrid) -> State<'window> {
		let size = window.inner_size();

		assert!(size.width > 0 && size.height > 0, "Height or width is 0");

		// The instance is a handle to our GPU
		// Backends::all => Vulkan + Metal + DX12 + Browser WebGPU
		let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
			backends: wgpu::Backends::all(),
			..Default::default()
		});

		let surface = instance.create_surface(window).unwrap();

		let adapter = instance.request_adapter(
			&wgpu::RequestAdapterOptions {
				power_preference: wgpu::PowerPreference::default(),
				compatible_surface: Some(&surface),
				force_fallback_adapter: false,
			},
		).await.unwrap();

		let (device, queue) = adapter.request_device(
			&wgpu::DeviceDescriptor {
				required_features: wgpu::Features::empty(),
				// WebGL doesn't support all of wgpu's features, so if
				// we're building for the web, we'll have to disable some.
				required_limits: if cfg!(target_arch = "wasm32") {
					wgpu::Limits::downlevel_webgl2_defaults()
				} else {
					wgpu::Limits::default()
				},
				label: None,
			},
			None, // Trace path
		).await.unwrap();

		let surface_caps = surface.get_capabilities(&adapter);
		// Shader code in this tutorial assumes an sRGB surface texture. Using a different
		// one will result in all the colors coming out darker. If you want to support non
		// sRGB surfaces, you'll need to account for that when drawing to the frame.
		let surface_format = surface_caps.formats.iter()
				.copied()
				.filter(|f| f.is_srgb())
				.next()
				.unwrap_or(surface_caps.formats[0]);

		let config = wgpu::SurfaceConfiguration {
			usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
			format: surface_format,
			width: size.width,
			height: size.height,
			present_mode: surface_caps.present_modes[0],
			alpha_mode: surface_caps.alpha_modes[0],
			view_formats: vec![],
			desired_maximum_frame_latency: 2,
		};
		surface.configure(&device, &config);

		let diffuse_bytes = include_bytes!("happy-tree.png");
		let diffuse_texture = texture::Texture::from_bytes(&device, &queue, diffuse_bytes, "happy-tree.png").unwrap();
		let texture_bind_group_layout =
				device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
					entries: &[
						wgpu::BindGroupLayoutEntry {
							binding: 0,
							visibility: wgpu::ShaderStages::FRAGMENT,
							ty: wgpu::BindingType::Texture {
								multisampled: false,
								view_dimension: wgpu::TextureViewDimension::D2,
								sample_type: wgpu::TextureSampleType::Float { filterable: true },
							},
							count: None,
						},
						wgpu::BindGroupLayoutEntry {
							binding: 1,
							visibility: wgpu::ShaderStages::FRAGMENT,
							// This should match the filterable field of the
							// corresponding Texture entry above.
							ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
							count: None,
						},
					],
					label: Some("texture_bind_group_layout"),
				});

		let diffuse_bind_group = device.create_bind_group(
			&wgpu::BindGroupDescriptor {
				layout: &texture_bind_group_layout,
				entries: &[
					wgpu::BindGroupEntry {
						binding: 0,
						resource: wgpu::BindingResource::TextureView(&diffuse_texture.view),
					},
					wgpu::BindGroupEntry {
						binding: 1,
						resource: wgpu::BindingResource::Sampler(&diffuse_texture.sampler),
					}
				],
				label: Some("diffuse_bind_group"),
			}
		);

		let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
			label: Some("Shader"),
			source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
		});

		let camera = camera::Camera::new((config.width as f32 / -2., config.height as f32 / -2., 6.), config.width, config.height, 1., 10.);
		let camera_controller = camera::CameraController::new(config.width as f32 / 4.0, 0.4);

		let mut camera_uniform = CameraUniform::new();
		camera_uniform.update_view_proj(&camera);

		let camera_buffer = device.create_buffer_init(
			&wgpu::util::BufferInitDescriptor {
				label: Some("Camera Buffer"),
				contents: bytemuck::cast_slice(&[camera_uniform]),
				usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
			}
		);

		let camera_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
			entries: &[
				wgpu::BindGroupLayoutEntry {
					binding: 0,
					visibility: wgpu::ShaderStages::VERTEX,
					ty: wgpu::BindingType::Buffer {
						ty: wgpu::BufferBindingType::Uniform,
						has_dynamic_offset: false,
						min_binding_size: None,
					},
					count: None,
				}
			],
			label: Some("camera_bind_group_layout"),
		});
		let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
			layout: &camera_bind_group_layout,
			entries: &[
				wgpu::BindGroupEntry {
					binding: 0,
					resource: camera_buffer.as_entire_binding(),
				}
			],
			label: Some("camera_bind_group"),
		});

		let render_pipeline_layout =
				device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
					label: Some("Render Pipeline Layout"),
					bind_group_layouts: &[
						&texture_bind_group_layout,
						&camera_bind_group_layout,
					],
					push_constant_ranges: &[],
				});

		let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
			label: Some("Render Pipeline"),
			layout: Some(&render_pipeline_layout),
			vertex: wgpu::VertexState {
				module: &shader,
				entry_point: "vs_main",
				buffers: &[
					Vertex::desc(),
					Instance::desc(),
				],
			},
			fragment: Some(wgpu::FragmentState {
				module: &shader,
				entry_point: "fs_main",
				targets: &[Some(wgpu::ColorTargetState {
					format: config.format,
					blend: Some(wgpu::BlendState::REPLACE),
					write_mask: wgpu::ColorWrites::ALL,
				})],
			}),
			primitive: wgpu::PrimitiveState {
				topology: wgpu::PrimitiveTopology::TriangleList,
				strip_index_format: None,
				front_face: wgpu::FrontFace::Ccw,
				cull_mode: Some(wgpu::Face::Back),
				// Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
				polygon_mode: wgpu::PolygonMode::Fill,
				// Requires Features::DEPTH_CLIP_CONTROL
				unclipped_depth: false,
				// Requires Features::CONSERVATIVE_RASTERIZATION
				conservative: false,
			},
			depth_stencil: None,
			multisample: wgpu::MultisampleState {
				count: 1,
				mask: !0,
				alpha_to_coverage_enabled: false,
			},
			multiview: None,
		});

		let vertex_buffer = device.create_buffer_init(
			&wgpu::util::BufferInitDescriptor {
				label: Some("Vertex Buffer"),
				contents: bytemuck::cast_slice(VERTICES),
				usage: wgpu::BufferUsages::VERTEX,
			}
		);
		let num_vertices = VERTICES.len() as u32;

		let index_buffer = device.create_buffer_init(
			&wgpu::util::BufferInitDescriptor {
				label: Some("Index Buffer"),
				contents: bytemuck::cast_slice(INDICES),
				usage: wgpu::BufferUsages::INDEX,
			}
		);
		let num_indices = INDICES.len() as u32;

		const NUM_INSTANCES_PER_ROW: u32 = 10;
		const INSTANCE_DISPLACEMENT: Vector3<f32> = Vector3::new(NUM_INSTANCES_PER_ROW as f32 * 0.5, NUM_INSTANCES_PER_ROW as f32 * 0.5, 0.0);

		let instances = (0..NUM_INSTANCES_PER_ROW).flat_map(|y| {
			(0..NUM_INSTANCES_PER_ROW).map(move |x| {
				let position = (Vector3 { x: x as f32, y: y as f32, z: 0.0 } - INSTANCE_DISPLACEMENT) * 10.;

				Instance {
					position,
				}
			})
		}).collect::<Vec<_>>();

		let instance_data = instances.iter().map(Instance::to_raw).collect::<Vec<_>>();
		let instance_buffer = device.create_buffer_init(
			&wgpu::util::BufferInitDescriptor {
				label: Some("Index Buffer"),
				contents: bytemuck::cast_slice(&instance_data),
				usage: wgpu::BufferUsages::VERTEX,
			}
		);


		State {
			window,
			surface,
			device,
			queue,
			config,
			size,
			render_pipeline,

			vertex_buffer,
			num_vertices,

			index_buffer,
			num_indices,

			instance_buffer,
			instances,

			diffuse_bind_group,
			diffuse_texture,

			camera,
			camera_controller,
			camera_uniform,
			camera_buffer,
			camera_bind_group,
			mouse_pressed: false,

			grid,
		}
	}

	pub fn window(&self) -> &Window {
		&self.window
	}

	pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
		if new_size.width > 0 && new_size.height > 0 {
			self.camera.resize(new_size.width, new_size.height);
			self.size = new_size;
			self.config.width = new_size.width;
			self.config.height = new_size.height;
			self.surface.configure(&self.device, &self.config);
		}
	}

	#[allow(unused_variables)]
	fn input(&mut self, event: &WindowEvent) -> bool {
		match event {
			WindowEvent::KeyboardInput {
				event,
				..
			} => self.camera_controller.process_keyboard(event),
			WindowEvent::MouseWheel { delta, .. } => {
				self.camera_controller.process_scroll(delta);
				true
			}
			WindowEvent::MouseInput {
				button: winit::event::MouseButton::Left,
				state,
				..
			} => {
				self.mouse_pressed = *state == ElementState::Pressed;
				true
			}
			_ => false,
		}
	}

	fn update(&mut self, dt: instant::Duration) {
		self.camera_controller.update_camera(&mut self.camera, dt);
		self.camera_uniform.update_view_proj(&self.camera);
		self.queue.write_buffer(&self.camera_buffer, 0, bytemuck::cast_slice(&[self.camera_uniform]));

		self.grid.update();
		// triangulate
		// self.grid.
	}

	fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
		let output = self.surface.get_current_texture()?;
		let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
		let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
			label: Some("Render Encoder"),
		});

		{
			let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
				label: Some("Render Pass"),
				color_attachments: &[Some(wgpu::RenderPassColorAttachment {
					view: &view,
					resolve_target: None,
					ops: wgpu::Operations {
						load: wgpu::LoadOp::Clear(wgpu::Color {
							r: 0.1,
							g: 0.2,
							b: 0.3,
							a: 1.0,
						}),
						store: wgpu::StoreOp::Store,
					},
				})],
				depth_stencil_attachment: None,
				occlusion_query_set: None,
				timestamp_writes: None,
			});


			render_pass.set_pipeline(&self.render_pipeline);

			render_pass.set_bind_group(0, &self.diffuse_bind_group, &[]);
			render_pass.set_bind_group(1, &self.camera_bind_group, &[]);

			render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
			render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));

			render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
			render_pass.draw_indexed(0..self.num_indices, 0, 0..self.instances.len() as _);
			// render_pass.draw_indexed(0..self.num_indices, 0, 0..1);
		}

		// submit will accept anything that implements IntoIter
		self.queue.submit(std::iter::once(encoder.finish()));
		self.window.pre_present_notify(); // maybe?
		output.present();

		let camera = self.camera.calc_matrix();
		let projected_origin = Point3::from_homogeneous(camera * Vector4 { x: 0., y: 0., z: 0., w: 1.0 });
		let projected_positive = Point3::from_homogeneous(camera * Vector4 { x: WIDTH, y: HEIGHT, z: 0., w: 1.0 });

		println!("camera position: {:?}", self.camera.position);
		println!("origin: {:?} corner: {:?}", projected_origin, projected_positive);

		Ok(())
	}
}

struct Instance {
	position: Vector3<f32>,
}

impl Instance {
	pub fn desc() -> wgpu::VertexBufferLayout<'static> {
		use std::mem;
		wgpu::VertexBufferLayout {
			array_stride: mem::size_of::<InstanceRaw>() as wgpu::BufferAddress,
			step_mode: wgpu::VertexStepMode::Instance,
			attributes: &[
				wgpu::VertexAttribute {
					offset: 0,
					shader_location: 5,
					format: wgpu::VertexFormat::Float32x4,
				},
				wgpu::VertexAttribute {
					offset: mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
					shader_location: 6,
					format: wgpu::VertexFormat::Float32x4,
				},
				wgpu::VertexAttribute {
					offset: mem::size_of::<[f32; 8]>() as wgpu::BufferAddress,
					shader_location: 7,
					format: wgpu::VertexFormat::Float32x4,
				},
				wgpu::VertexAttribute {
					offset: mem::size_of::<[f32; 12]>() as wgpu::BufferAddress,
					shader_location: 8,
					format: wgpu::VertexFormat::Float32x4,
				},
			],
		}
	}
	fn to_raw(&self) -> InstanceRaw {
		InstanceRaw {
			model: cgmath::Matrix4::from_translation(self.position).into(),
		}
	}
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct InstanceRaw {
	model: [[f32; 4]; 4],
}