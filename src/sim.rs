use rand::{distributions::Uniform, prelude::Distribution};

pub struct Particle {
    pub x: f32,
    pub y: f32,
    pub heading: f32,
    pub neighbours: u8,
}

impl Particle {
    pub fn encode(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend(self.x.to_le_bytes());
        bytes.extend(self.y.to_le_bytes());
        bytes.extend(self.heading.to_le_bytes());
        // 24-bit id field unused
        bytes.extend([0u8, 0u8, 0u8]);
        // 8-bit neighbour count
        bytes.extend(self.neighbours.to_le_bytes());
        bytes
    }
}

/// Approximates wrapping x within [0, max].
pub fn wrap(x: f32, max: f32) -> f32 {
    if x > max {
        x - max
    } else if x < 0.0 {
        x + max
    } else {
        x
    }
}

pub fn within_radius(delta_x: f32, delta_y: f32, r: f32) -> bool {
    delta_x * delta_x + delta_y * delta_y <= r * r
}

pub fn is_on_left(delta_x: f32, delta_y: f32, selected_heading: f32) -> bool {
    let (s, c) = f32::sin_cos(selected_heading);
    delta_x * s + delta_y * c >= 0.0
}

// Distributions

pub struct RectDistribution {
    pub width: f32,
    pub height: f32,
}

impl Distribution<Particle> for RectDistribution {
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> Particle {
        let x_distribution = Uniform::new(0.0, self.width);
        let y_distribution = Uniform::new(0.0, self.height);
        let heading_distribution = Uniform::new_inclusive(0.0, 2.0 * std::f32::consts::PI);
        Particle {
            x: x_distribution.sample(rng),
            y: y_distribution.sample(rng),
            heading: heading_distribution.sample(rng),
            neighbours: 0,
        }
    }
}

pub trait Simulation {
    fn dump_to_file(&self, file: &mut std::fs::File) -> std::io::Result<()>;
    fn tick(&mut self);
}
