use rand::{distributions::Uniform, prelude::Distribution};
use std::f32::consts::PI;
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
    let x_distribution = Uniform::new(0.0, params.world_width);
    let y_distribution = Uniform::new(0.0, params.world_height);
    let heading_distribution = Uniform::new_inclusive(0.0, 2.0 * PI);

    let mut particles: Vec<Particle> = (0..params.particle_count)
        .map(|_| Particle {
            x: x_distribution.sample(&mut rng),
            y: y_distribution.sample(&mut rng),
            heading: heading_distribution.sample(&mut rng),
            neighbours: 0,
        })
        .collect();

    for tick in 0..=params.tick_count {
        // Write to file
        if tick % params.dump_skip_size == 0 {
            dump_file.write_all(
                particles
                    .iter()
                    .flat_map(|p| p.encode())
                    .collect::<Vec<u8>>()
                    .as_slice(),
            )?;
        }

        // Calculate neighbours on left and right
        let mut left_right_neighbours =
            vec![SideNeighbourCount { left: 0, right: 0 }; params.particle_count];
        for (i, (selected, SideNeighbourCount { left, right })) in
            std::iter::zip(&particles, &mut left_right_neighbours).enumerate()
        {
            for current in particles[..i].iter().chain(&particles[i + 1..]) {
                let delta_x = wrap(current.x - selected.x, params.world_width);
                let delta_y = wrap(current.y - selected.y, params.world_height);
                if within_radius(delta_x, delta_y, params.r) {
                    if is_on_left(delta_x, delta_y, selected.heading) {
                        *left += 1;
                    } else {
                        *right += 1;
                    }
                }
            }
        }

        // Update particle data
        for (p, SideNeighbourCount { left, right }) in
            std::iter::zip(&mut particles, &left_right_neighbours)
        {
            let neighbours = left + right;
            p.neighbours = neighbours;
            let delta_heading = params.alpha
                + params.beta
                    * (neighbours as i8 * i8::signum((*right as i8) - (*left as i8))) as f32;
            p.heading = (p.heading + delta_heading) % (2.0 * PI);
            let (s, c) = f32::sin_cos(p.heading);
            p.x = wrap(p.x + params.v * c, params.world_width);
            p.y = wrap(p.y + params.v * s, params.world_height);
        }
    }

    Ok(())
}
