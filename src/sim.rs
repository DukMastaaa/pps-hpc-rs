use rand::{distributions::Uniform, prelude::Distribution};

// Generic simulation elements

#[derive(Clone, Copy)]
pub struct CommonParams {
    pub world_width: f32,
    pub world_height: f32,
    pub alpha: f32,
    pub beta: f32,
    pub r: f32,
    pub v: f32,
}

pub struct GenericParticle {
    pub x: f32,
    pub y: f32,
    pub heading: f32,
}

pub trait Simulate {
    fn dump_to_file(&self, file: &mut std::fs::File) -> std::io::Result<()>;
    fn tick(&mut self);
}

// Helper functions

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

impl Distribution<GenericParticle> for RectDistribution {
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> GenericParticle {
        let x_distribution = Uniform::new(0.0, self.width);
        let y_distribution = Uniform::new(0.0, self.height);
        let heading_distribution = Uniform::new_inclusive(0.0, 2.0 * std::f32::consts::PI);
        GenericParticle {
            x: x_distribution.sample(rng),
            y: y_distribution.sample(rng),
            heading: heading_distribution.sample(rng),
        }
    }
}
