mod tribal;


use wgpu::util::DeviceExt;
use winit::window::{Window, WindowBuilder};
use winit::event::{ElementState, Event, KeyEvent, WindowEvent};
use winit::event_loop::EventLoop;
use winit::keyboard::{Key, NamedKey};
use crate::tribal::{hex, TribalHexGrid};
use crate::tribal::hex::Hex;


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

    let _ = event_loop.run(move |event, elwt| {
        match event {
            Event::WindowEvent {
                ref event, window_id
            } if window_id == state.window.id() && !state.input(event) => match event {
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
                    state.update();
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

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });
        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[],
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

        let instance_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Index Buffer"),
                contents: bytemuck::cast_slice(INSTANCES),
                usage: wgpu::BufferUsages::VERTEX,
            }
        );
        let num_instances = INSTANCES.len() as u32;

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
            instances: INSTANCES.to_vec(),

            grid,
        }
    }

    pub fn window(&self) -> &Window {
        &self.window
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    #[allow(unused_variables)]
    fn input(&mut self, event: &WindowEvent) -> bool {
        false
    }

    fn update(&mut self) {
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

            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));

            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            render_pass.draw_indexed(0..self.num_indices, 0, 0..self.instances.len() as _);
        }

        // submit will accept anything that implements IntoIter
        self.queue.submit(std::iter::once(encoder.finish()));
        self.window.pre_present_notify(); // maybe?
        output.present();

        Ok(())
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 2],
    color: [f32; 3],
}

impl Vertex {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 2]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                }
            ],
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Instance {
    position: [f32; 4],
}

impl Instance {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Instance>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        }
    }
}

const WIDTH: f32 = 100.;
const HEIGHT: f32 = 100.;

const VERTICES: &[Vertex] = &[
    // Vertex { position: [-0.0868241, 0.49240386], color: [0.5, 1.0, 0.5] }, // A
    // Vertex { position: [-0.49513406, 0.06958647], color: [0.5, 1.0, 0.5] }, // B
    // Vertex { position: [-0.21918549, -0.44939706], color: [0.5, 0.0, 1.] }, // C
    // Vertex { position: [0.35966998, -0.3473291], color: [0.5, 0.0, 0.5] }, // D
    // Vertex { position: [0.44147372, 0.2347359], color: [1., 0.0, 0.] }, // E

    Vertex { position: [(10. - (WIDTH / 2.)) / (WIDTH / 2.), (90. - (HEIGHT / 2.)) / (HEIGHT / 2.)], color: [1., 0., 0.] }, // a
    Vertex { position: [(10. - (WIDTH / 2.)) / (WIDTH / 2.), (80. - (HEIGHT / 2.)) / (HEIGHT / 2.)], color: [1., 1., 0.] }, // b
    Vertex { position: [(20. - (WIDTH / 2.)) / (WIDTH / 2.), (80. - (HEIGHT / 2.)) / (HEIGHT / 2.)], color: [1., 0., 1.] }, // c
    Vertex { position: [(20. - (WIDTH / 2.)) / (WIDTH / 2.), (90. - (HEIGHT / 2.)) / (HEIGHT / 2.)], color: [0., 1., 1.] }, // d

    // Vertex { position: [-0.9, 0.9], color: [1., 0.0, 0.] }, // a
    // Vertex { position: [-0.9, 0.8], color: [1., 0.0, 0.] }, // b
    // Vertex { position: [-0.8, 0.8], color: [1., 0.0, 0.] }, // c
    // Vertex { position: [-0.8, 0.9], color: [1., 0.0, 0.] }, // d
];

const INDICES: &[u16] = &[
    // 0, 1, 4,
    // 1, 2, 4,
    // 2, 3, 4,
    0, 1, 3,
    1, 2, 3
];

const INSTANCES: &[Instance] = &[
    Instance { position: [1.1, 0.7, 1., 1.] },
    Instance { position: [1., 1., 1., 1.] }
];