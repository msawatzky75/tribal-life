#![allow(dead_code)]

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::hash::Hash;
use cgmath::{Point2};

/// Generate a pseudorandom seed for the game's PRNG.
fn generate_seed() -> (u64, u64) {
	use byteorder::{ByteOrder, NativeEndian};
	use getrandom::getrandom;

	let mut seed = [0_u8; 16];

	getrandom(&mut seed).expect("failed to getrandom");

	(
		NativeEndian::read_u64(&seed[0..8]),
		NativeEndian::read_u64(&seed[8..16]),
	)
}

#[derive(Copy, Clone, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub enum Tribe {
	Red,
	Green,
	Blue,
}


impl Tribe {
	fn all() -> [Tribe; 3] {
		[Tribe::Red, Tribe::Green, Tribe::Blue]
	}
	pub fn get_color(&self) -> &[f32; 4] {
		match self {
			Tribe::Red => &[1., 0., 0., 1.],
			Tribe::Green => &[0., 1., 0., 1.],
			Tribe::Blue => &[0., 0., 1., 1.],
		}
	}
	pub fn beats(&self, other: Self) -> bool {
		match self {
			Tribe::Red => other == Tribe::Green,
			Tribe::Green => other == Tribe::Blue,
			Tribe::Blue => other == Tribe::Red,
		}
	}
}

fn get_next_tribe(tribe: Tribe, n: [(Hex, &Tribe); 6]) -> Tribe {
	let binding = n.map(|(_, tribe)| { *tribe });
	let mut freq = most_frequent(&binding, 2);
	let (count, mut n_tribe) = freq.pop().unwrap();

	if *n_tribe == tribe {
		let next = freq.pop();
		if next.is_none() { return tribe; } else { n_tribe = next.unwrap().1; }
	}
	if n_tribe.beats(tribe) && count >= 2 { *n_tribe } else { tribe }
}

pub struct TribalHexGrid {
	size: Point2<usize>,
	cells: HashMap<Hex, Tribe>,
	scratch_cells: HashMap<Hex, Tribe>,
	pub layout: Layout,
}

impl TribalHexGrid {
	pub fn new_empty(grid_size: Point2<f32>, cell_size: Point2<f32>) -> Self {
		let layout = Layout {
			orientation: Orientation::pointy(),
			size: cell_size,
			origin: Point2 { x: grid_size.x * cell_size.x, y: grid_size.y * cell_size.y } / -2.,
		};

		let width = grid_size.x;
		let height = grid_size.y;

		assert!(width != 0. && height != 0.);

		let left: i32 = 0;
		let right: i32 = width as i32;
		let top: i32 = 0;
		let bottom: i32 = height as i32;

		Self {
			cells: TribalHexGrid::pointy_rect(left, right, top, bottom),
			scratch_cells: TribalHexGrid::pointy_rect(left, right, top, bottom),
			size: Point2 {
				x: width as usize,
				y: height as usize,
			},
			layout,
		}
	}
	pub fn new_random(grid_size: Point2<f32>, cell_size: Point2<f32>) -> Self {
		let mut result = Self::new_empty(grid_size, cell_size);
		// randomize tribes
		result.randomize();
		result
	}
	pub fn pointy_rect<'a>(left: i32, right: i32, top: i32, bottom: i32) -> HashMap<Hex, Tribe> {
		let mut map: HashMap<Hex, Tribe> = HashMap::new();

		for r in top..bottom { // pointy top
			let r_offset = r >> 1;
			for q in (left - r_offset)..(right - r_offset) {
				map.insert(Hex::new(q, r), Tribe::Red);
			}
		}

		map
	}
	pub fn randomize(&mut self) {
		let tribes = Tribe::all();
		let mut rng: randomize::PCG32 = generate_seed().into();
		for (_, tribe) in self.cells.iter_mut() {
			let new_tribe = (randomize::f32_half_open_right(rng.next_u32()) * tribes.iter().count() as f32) as usize;
			*tribe = tribes[new_tribe];
		}
		// // run a few simulation iterations for aesthetics (If we don't, the
		// // noise is ugly)
		// for _ in 0..3 {
		//     self.update();
		// }
	}
	fn get_neighbors(&self, cell: Hex) -> [Hex; 6] {
		Hex::directions().map(|offset| {
			cell.add(offset)
		})
	}
	pub fn update(&mut self) {
		for (cell, tribe) in self.cells.iter() {
			let neighbors = self.get_neighbors(*cell);
			let next = get_next_tribe(*tribe, neighbors.map(|n| {
				return if let Some(tribe) = self.cells.get(&n) {
					(n, tribe)
				} else {
					let wrapped_n: Hex = self.get_wrapped(n);
					if self.cells.contains_key(&wrapped_n) {
						(wrapped_n, self.cells.get(&wrapped_n).unwrap())
					} else { (*cell, tribe) }
				};
			}));

			// Write into scratch_cells, since we're still reading from `self.cells`
			self.scratch_cells.insert(*cell, next);
		}
		std::mem::swap(&mut self.scratch_cells, &mut self.cells);
	}

	pub fn cells(&self) -> &HashMap<Hex, Tribe> { &self.cells }

	fn get_wrapped(&self, bad_neighbor: Hex) -> Hex {
		let mut new = Hex { q: bad_neighbor.q, r: bad_neighbor.r };

		if bad_neighbor.r > self.size.y as i32 { new.r = 0 } else if bad_neighbor.r < 0 { new.r = self.size.y as i32 - 1 }
		if bad_neighbor.q > self.size.x as i32 { new.q = 0 } else if bad_neighbor.q < 0 { new.q = self.size.x as i32 - 1 }

		new
	}
}

pub fn most_frequent<T>(array: &[T], k: usize) -> Vec<(usize, &T)>
	where
			T: Hash + Eq + Ord,
{
	let mut map = HashMap::with_capacity(array.len());
	for x in array {
		*map.entry(x).or_default() += 1;
	}

	let mut heap = BinaryHeap::with_capacity(k + 1);
	for (x, count) in map.into_iter() {
		if heap.len() < k {
			heap.push(Reverse((count, x)));
		} else {
			let &Reverse((min, _)) = heap.peek().unwrap();
			if min < count {
				heap.pop();
				heap.push(Reverse((count, x)));
			}
		}
	}
	heap.into_sorted_vec().into_iter().map(|r| r.0).collect()
}

#[derive(Eq, PartialEq, Copy, Clone, Default, Hash)]
pub struct Hex {
	pub q: i32,
	pub r: i32,
}

impl Hex {
	pub fn new(q: i32, r: i32) -> Self {
		Self { q, r }
	}
	pub fn new_axial(q: i32, r: i32, s: i32) -> Self {
		assert_eq!(q + r + s, 0);
		Self { q, r }
	}

	pub fn directions() -> [Hex; 6] {
		[
			Hex::new(1, 0), Hex::new(1, -1), Hex::new(0, -1),
			Hex::new(-1, 0), Hex::new(-1, 1), Hex::new(0, 1)
		]
	}
	pub fn direction(direction: usize) -> Hex {
		assert!(direction < 6);
		Hex::directions()[direction]
	}

	pub fn s(&self) -> i32 { -self.q - self.r }
	pub fn add(&self, other: Hex) -> Self {
		Self {
			q: self.q + other.q,
			r: self.r + other.r,
		}
	}
	pub fn subtract(&self, other: Hex) -> Self {
		Self {
			q: self.q - other.q,
			r: self.r - other.r,
		}
	}
	pub fn multiply(&self, other: Hex) -> Self {
		Self {
			q: self.q * other.q,
			r: self.r * other.r,
		}
	}

	/// Distance to origin
	pub fn length(&self) -> i32 { (self.q.abs() + self.r.abs() + self.s().abs()) / 2 }
	/// Distance to another cell
	pub fn distance(&self, other: Hex) -> i32 {
		self.subtract(other).length()
	}

	pub fn to_pixel(&self, layout: Layout) -> Point2<f32> {
		let m = &layout.orientation;
		Point2 {
			x: (((m.f[0] * self.q as f32) + (m.f[1] * self.r as f32)) as f32 * layout.size.x) + layout.origin.x,
			y: (((m.f[2] * self.q as f32) + (m.f[3] * self.r as f32)) as f32 * layout.size.y) + layout.origin.y,
		}
	}
	pub fn from_pixel(layout: Layout, p: Point2<f32>) -> Self {
		let m = &layout.orientation;
		let pt: Point2<f32> = Point2 {
			x: (p.x - layout.origin.x) / layout.size.x,
			y: (p.y - layout.origin.y) / layout.size.y,
		};
		Hex {
			q: (m.b[0] as f32 * pt.x + m.b[1] as f32 * pt.y) as i32,
			r: (m.b[2] as f32 * pt.x + m.b[3] as f32 * pt.y) as i32,
		}
	}
	fn corner_offset(&self, layout: Layout, corner: f32) -> Point2<f32> {
		let angle: f32 = 2. * std::f32::consts::PI * (layout.orientation.start_angle + corner) / 6.;

		Point2 {
			x: layout.size.x * angle.cos(),
			y: layout.size.y * angle.sin(),
		}
	}
	pub fn polygon_corners(self, layout: Layout) -> [Point2<f32>; 6] {
		let center = self.to_pixel(layout);
		core::array::from_fn(|i| {
			let offset = self.corner_offset(layout, i as f32);
			Point2 {
				x: center.x + offset.x,
				y: center.y + offset.y,
			}
		})
	}
}

#[derive(Copy, Clone)]
pub struct Orientation {
	pub f: [f32; 4],
	pub b: [f32; 4],
	pub start_angle: f32, // in multiples of 60
}

impl Orientation {
	pub fn pointy() -> Orientation {
		Orientation {
			f: [f32::sqrt(3.), f32::sqrt(3.) / 2., 0., 3. / 2.],
			b: [f32::sqrt(3.) / 3., -1. / 3., 0., 2. / 3.],
			start_angle: 0.5,
		}
	}
	pub fn flat() -> Orientation {
		Orientation {
			f: [3. / 2., 0., f32::sqrt(3.) / 2., f32::sqrt(3.)],
			b: [2. / 3., 0., -1. / 3., f32::sqrt(3.) / 3.],
			start_angle: 0.,
		}
	}
}


#[derive(Copy, Clone)]
pub struct Layout {
	pub orientation: Orientation,
	pub size: Point2<f32>,
	pub origin: Point2<f32>,
}
