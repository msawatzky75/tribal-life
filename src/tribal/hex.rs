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

    pub fn to_pixel(&self, layout: Layout) -> Point<f64> {
        let m = &layout.orientation;
        Point {
            x: (((m.f[0] * self.q as f64) + (m.f[1] * self.r as f64)) * layout.size.x) + layout.origin.x,
            y: (((m.f[2] * self.q as f64) + (m.f[3] * self.r as f64)) * layout.size.y) + layout.origin.y,
        }
    }
    pub fn from_pixel(layout: Layout, p: Point<f64>) -> Self {
        let m = &layout.orientation;
        let pt: Point<f64> = Point {
            x: (p.x - layout.origin.x) / layout.size.x,
            y: (p.y - layout.origin.y) / layout.size.y,
        };
        Hex {
            q: (m.b[0] * pt.x + m.b[1] * pt.y) as i32,
            r: (m.b[2] * pt.x + m.b[3] * pt.y) as i32,
        }
    }
    fn corner_offset(&self, layout: Layout, corner: f64) -> Point<f64> {
        let angle: f64 = 2. * std::f64::consts::PI * (layout.orientation.start_angle + corner) / 6.;

        Point {
            x: layout.size.x * angle.cos(),
            y: layout.size.y * angle.sin(),
        }
    }
    pub fn polygon_corners(self, layout: Layout) -> [Point<f64>; 6] {
        let center = self.to_pixel(layout);
        core::array::from_fn(|i| {
            let offset = self.corner_offset(layout, i as f64);
            Point {
                x: center.x + offset.x,
                y: center.y + offset.y,
            }
        })
    }
}

#[derive(Copy, Clone)]
pub struct Orientation {
    pub f: [f64; 4],
    pub b: [f64; 4],
    pub start_angle: f64, // in multiples of 60
}

impl Orientation {
    pub fn pointy() -> Orientation {
        Orientation {
            f: [f64::sqrt(3.), f64::sqrt(3.) / 2., 0., 3. / 2.],
            b: [f64::sqrt(3.) / 3., -1. / 3., 0., 2. / 3.],
            start_angle: 0.5,
        }
    }
    pub fn flat() -> Orientation {
        Orientation {
            f: [3. / 2., 0., f64::sqrt(3.) / 2., f64::sqrt(3.)],
            b: [2. / 3., 0., -1. / 3., f64::sqrt(3.) / 3.],
            start_angle: 0.,
        }
    }
}

#[derive(Copy, Clone)]
pub struct Point<T> {
    pub x: T,
    pub y: T,
}

#[derive(Copy, Clone)]
pub struct Layout {
    pub orientation: Orientation,
    pub size: Point<f64>,
    pub origin: Point<f64>,
}
