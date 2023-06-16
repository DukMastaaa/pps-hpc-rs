use rand::{distributions::Uniform, prelude::Distribution};
use std::f32::consts::PI;

struct Particle {
    x: f32,
    y: f32,
    heading: f32,
    neighbours: i32,
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

fn main() {
    let tick_count: usize = 100;
    let particle_count: usize = 1000;
    let world_width: f32 = 100.0;
    let world_height: f32 = 100.0;

    let sim_alpha: f32 = f32::to_radians(180.0);
    let sim_beta: f32 = f32::to_radians(17.0);
    let sim_r: f32 = 5.0;
    let sim_v: f32 = 0.67;

    let mut rng = rand::thread_rng();
    let x_distribution = Uniform::new(0.0, world_width);
    let y_distribution = Uniform::new(0.0, world_height);
    let heading_distribution = Uniform::new_inclusive(0.0, 2.0 * PI);

    let mut particles: Vec<Particle> = (0..particle_count)
        .map(|_| Particle {
            x: x_distribution.sample(&mut rng),
            y: y_distribution.sample(&mut rng),
            heading: heading_distribution.sample(&mut rng),
            neighbours: 0,
        })
        .collect();

    for tick in 0..tick_count {
        // Calculate changes in heading
        let mut delta_headings = vec![0.0f32; particle_count];
        for i in 0..particle_count {
            let selected = &particles[i];
            let mut left: i32 = 0;
            let mut right: i32 = 0;
            for j in 0..particle_count {
                if j == i {
                    continue;
                }
                let current = &particles[j];
                let delta_x = wrap(current.x - selected.x, world_width);
                let delta_y = wrap(current.y - selected.y, world_height);
                if within_radius(delta_x, delta_y, sim_r) {
                    if is_on_left(delta_x, delta_y, selected.heading) {
                        left += 1;
                    } else {
                        right += 1;
                    }
                }
            }
            let neighbours = left + right;
            particles[i].neighbours = neighbours;
            delta_headings[i] =
                sim_alpha + sim_beta * (neighbours * i32::signum(right - left)) as f32;
        }

        // Update particle data
        for (mut p, delta_heading) in std::iter::zip(&mut particles, &delta_headings) {
            p.heading = (p.heading + delta_heading) % (2.0 * PI);
            let (s, c) = f32::sin_cos(p.heading);
            p.x = wrap(p.x + sim_v * c, world_width);
            p.y = wrap(p.y + sim_v * s, world_height);
        }
    }
}
