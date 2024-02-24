pub mod hex;

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::hash::Hash;
use crate::tribal::hex::*;
use crate::tribal::hex::Point;
use wgpu::Color;

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
struct Tribe {
    id: u8,
}

impl Tribe {
    pub fn colors() -> Vec<Color> {
        vec![
            Color {
                a: (0xff / 0xff) as f64,
                r: (0xff / 0xff) as f64,
                g: 0f64,
                b: 0f64,
            },
            Color {
                a: (0xff / 0xff) as f64,
                r: 0f64,
                g: 0f64,
                b: (0xff / 0xff) as f64,
            },
            Color {
                a: (0xff / 0xff) as f64,
                r: (0xff / 0xff) as f64,
                g: (0xa5 / 0xff) as f64,
                b: 0f64,
            },
        ]
    }
    pub fn red() -> Tribe { Tribe { id: 0 } }
    pub fn blue() -> Tribe { Tribe { id: 1 } }
    pub fn orange() -> Tribe { Tribe { id: 2 } }
    pub fn all() -> [Tribe; 3] { [Tribe::red(), Tribe::blue(), Tribe::orange()] }
    pub fn get_color(&self) -> Color { Self::colors()[self.id as usize] }
    pub fn beats(&self, other: Self) -> bool {
        return match self.id {
            0 => other.id == Self::blue().id,
            1 => other.id == Self::orange().id,
            2 => other.id == Self::red().id,
            _ => false
        };
    }
}

fn get_next_tribe(tribe: Tribe, n: [(Hex, &Tribe); 6]) -> Tribe {
    let binding = n.map(|(_, tribe)| { *tribe });
    let mut freq = most_frequent(&binding, 2);
    let (count, mut n_tribe) = freq.pop().unwrap();

    if n_tribe.id == tribe.id {
        let next = freq.pop();
        if next.is_none() { return tribe; } else { n_tribe = next.unwrap().1; }
    }
    if n_tribe.beats(tribe) && count >= 2 { *n_tribe } else { tribe }
}

pub struct TribalHexGrid {
    size: Point<usize>,
    cells: HashMap<Hex, Tribe>,
    scratch_cells: HashMap<Hex, Tribe>,
    layout: Layout,
}

impl TribalHexGrid {
    pub fn new_empty(dt_size: Point<f64>, cell_size: Point<f64>) -> Self {
        let layout = Layout {
            orientation: Orientation::pointy(),
            size: cell_size,
            origin: Point { x: cell_size.x, y: cell_size.y },
        };

        let width = dt_size.x / (cell_size.x * layout.orientation.f[0]);
        let height = dt_size.y / (cell_size.y * layout.orientation.f[3]);

        assert!(width != 0. && height != 0.);
        // assert!((width * cell_size.x * 2.) + 1. <= dt_size.x, "Grid width too large");
        // assert!((height * cell_size.y * 2.) + 1. <= dt_size.y, "Grid height too large");

        let left: i32 = 0;
        let right: i32 = width as i32;
        let top: i32 = 0;
        let bottom: i32 = height as i32;

        Self {
            cells: TribalHexGrid::pointy_rect(left, right, top, bottom),
            scratch_cells: TribalHexGrid::pointy_rect(left, right, top, bottom),
            size: Point {
                x: width as usize,
                y: height as usize,
            },
            layout,
        }
    }
    pub fn new_random(dt_size: Point<f64>, cell_size: Point<f64>) -> Self {
        let mut result = Self::new_empty(dt_size, cell_size);
        // randomize tribes
        result.randomize();
        result
    }
    fn pointy_rect(left: i32, right: i32, top: i32, bottom: i32) -> HashMap<Hex, Tribe> {
        let mut map: HashMap<Hex, Tribe> = HashMap::new();

        for r in top..bottom { // pointy top
            let r_offset = r >> 1;
            for q in (left - r_offset)..(right - r_offset) {
                map.insert(Hex::new(q, r), Tribe::red());
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