use rand::{distributions::Uniform, prelude::Distribution};
use std::fs::File;
use std::io::prelude::*;

const MAGIC_BIT_PATTERN: u64 = 0x1234;

struct SimParams {
    tick_count: usize,
    dump_skip_size: usize,
    particle_count: usize,
    world_width: f32,
    world_height: f32,
    alpha: f32,
    beta: f32,
    r: f32,
    v: f32,
    file_name: String,
}

impl SimParams {
    fn encode(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        // These aren't the same size so I can't use [u8; N]::concat()
        bytes.extend(self.tick_count.to_le_bytes());
        bytes.extend(self.particle_count.to_le_bytes());
        bytes.extend(self.world_width.to_le_bytes());
        bytes.extend(self.world_height.to_le_bytes());
        bytes.extend(self.dump_skip_size.to_le_bytes());
        bytes.extend(MAGIC_BIT_PATTERN.to_le_bytes());
        bytes
    }
}

struct Particle {
    x: f32,
    y: f32,
    heading: f32,
    neighbours: u8,
}

impl Particle {
    fn encode(&self) -> Vec<u8> {
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

struct RectDistribution {
    width: f32,
    height: f32
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
            neighbours: 0
        }
    }
}

#[derive(Clone)]
struct SideNeighbourCount {
    left: u8,
    right: u8,
}

/// Approximates wrapping x within [0, max].
fn wrap(x: f32, max: f32) -> f32 {
    if x > max {
        x - max
    } else if x < 0.0 {
        x + max
    } else {
        x
    }
}

fn within_radius(delta_x: f32, delta_y: f32, r: f32) -> bool {
    delta_x * delta_x + delta_y * delta_y <= r * r
}

fn is_on_left(delta_x: f32, delta_y: f32, selected_heading: f32) -> bool {
    let (s, c) = f32::sin_cos(selected_heading);
    delta_x * s + delta_y * c >= 0.0
}

struct SerialSimulation {
    world_width: f32,
    world_height: f32,
    alpha: f32,
    beta: f32,
    r: f32,
    v: f32,
    particles: Vec<Particle>
}

impl SerialSimulation {
    fn dump_to_file(&self, file: &mut File) -> std::io::Result<()> {
        file.write_all(
            self.particles
                .iter()
                .flat_map(Particle::encode)
                .collect::<Vec<u8>>()
                .as_slice(),
        )
    }

    fn calculate_neighbours(&self) -> Vec<SideNeighbourCount> {
        let mut left_right_neighbours =
            vec![SideNeighbourCount { left: 0, right: 0 }; self.particles.len()];
        for (i, (selected, SideNeighbourCount { left, right })) in
            std::iter::zip(&self.particles, &mut left_right_neighbours).enumerate()
        {
            for current in self.particles[..i].iter().chain(&self.particles[i + 1..]) {
                let delta_x = wrap(current.x - selected.x, self.world_width);
                let delta_y = wrap(current.y - selected.y, self.world_height);
                if within_radius(delta_x, delta_y, self.r) {
                    if is_on_left(delta_x, delta_y, selected.heading) {
                        *left += 1;
                    } else {
                        *right += 1;
                    }
                }
            }
        }
        left_right_neighbours
    }

    fn update_particle_data(&mut self, left_right_neighbours: Vec<SideNeighbourCount>) {
        for (p, SideNeighbourCount { left, right }) in
            std::iter::zip(&mut self.particles, &left_right_neighbours)
        {
            let neighbours = left + right;
            p.neighbours = neighbours;
            let delta_heading = self.alpha
                + self.beta
                    * (neighbours as i8 * i8::signum((*right as i8) - (*left as i8))) as f32;
            p.heading = (p.heading + delta_heading) % (2.0 * std::f32::consts::PI);
            let (s, c) = f32::sin_cos(p.heading);
            p.x = wrap(p.x + self.v * c, self.world_width);
            p.y = wrap(p.y + self.v * s, self.world_height);
        }
    }

    fn tick(&mut self) {
        self.update_particle_data(self.calculate_neighbours());
    }
}

fn main() -> std::io::Result<()> {
    let params = SimParams {
        tick_count: 100,
        dump_skip_size: 1,
        particle_count: 1000,
        world_width: 100.0,
        world_height: 100.0,
        alpha: f32::to_radians(180.0),
        beta: f32::to_radians(70.0),
        r: 5.0,
        v: 0.67,
        file_name: "test.dump".to_string(),
    };

    let mut dump_file = File::create(params.file_name.clone())?;
    dump_file.write_all(&params.encode())?;

    let mut rng = rand::thread_rng();
    let dist = RectDistribution {width: params.world_width, height: params.world_height};
    let particles: Vec<Particle> = (0..params.particle_count).map(|_| dist.sample(&mut rng)).collect();

    let mut sim = SerialSimulation {
        world_width: params.world_width,
        world_height: params.world_height,
        alpha: params.alpha,
        beta: params.beta,
        r: params.r,
        v: params.v,
        particles
    };

    for tick in 0..=params.tick_count {
        if tick % params.dump_skip_size == 0 {
            sim.dump_to_file(&mut dump_file)?
        }
        sim.tick();
    }

    Ok(())
}
