use crate::sim;
use std::{fs::File, io::Write};

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

impl From<sim::GenericParticle> for Particle {
    fn from(value: sim::GenericParticle) -> Self {
        Self {
            x: value.x,
            y: value.y,
            heading: value.heading,
            neighbours: 0,
        }
    }
}

pub struct SpecificParams {}

pub struct SerialSimulation {
    common: sim::CommonParams,
    #[allow(dead_code)]
    specific: SpecificParams,
    particles: Vec<Particle>,
}

impl SerialSimulation {
    pub fn new(
        common: sim::CommonParams,
        specific: SpecificParams,
        particles: Vec<sim::GenericParticle>,
    ) -> Self {
        Self {
            common,
            specific,
            particles: particles.into_iter().map(Particle::from).collect(),
        }
    }

    /// Calculates the number of neighbours on the left and right of the
    /// particle at the specified index.
    fn calculate_neighbours(
        common: &sim::CommonParams,
        particles: &Vec<Particle>,
        particle_idx: usize,
    ) -> sim::SideNeighbourCount {
        let mut left = 0;
        let mut right = 0;
        let selected = &particles[particle_idx];
        for (i, current) in particles.into_iter().enumerate() {
            if i == particle_idx {
                continue;
            }
            let delta_x = sim::wrap(current.x - selected.x, common.world_width);
            let delta_y = sim::wrap(current.y - selected.y, common.world_height);
            if sim::within_radius(delta_x, delta_y, common.r) {
                if sim::is_on_left(delta_x, delta_y, selected.heading) {
                    left += 1;
                } else {
                    right += 1;
                }
            }
        }
        sim::SideNeighbourCount { left, right }
    }

    /// Updates a particle's fields given how many neighbours it has.
    fn update_particle_data(
        common: &sim::CommonParams,
        p: &mut Particle,
        neighbours: &sim::SideNeighbourCount,
    ) {
        let total = neighbours.left + neighbours.right;
        p.neighbours = total;
        let delta_heading = common.alpha
            + common.beta
                * (total as i8 * i8::signum((neighbours.right as i8) - (neighbours.left as i8)))
                    as f32;
        p.heading = (p.heading + delta_heading) % (2.0 * std::f32::consts::PI);
        let (s, c) = f32::sin_cos(p.heading);
        p.x = sim::wrap(p.x + common.v * c, common.world_width);
        p.y = sim::wrap(p.y + common.v * s, common.world_height);
    }
}

impl sim::Simulate for SerialSimulation {
    fn dump_to_file(&self, file: &mut File) -> std::io::Result<()> {
        file.write_all(
            self.particles
                .iter()
                .flat_map(Particle::encode)
                .collect::<Vec<u8>>()
                .as_slice(),
        )
    }

    fn tick(&mut self) {
        let left_right_neighbours: Vec<sim::SideNeighbourCount> = (0..self.particles.len())
            .map(|i| Self::calculate_neighbours(&self.common, &self.particles, i))
            .collect();
        for (p, neighbours) in std::iter::zip(&mut self.particles, &left_right_neighbours) {
            Self::update_particle_data(&self.common, p, neighbours);
        }
    }
}
