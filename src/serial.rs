use crate::sim;
use std::{fs::File, io::Write};

#[derive(Clone)]
struct SideNeighbourCount {
    left: u8,
    right: u8,
}

pub struct SerialSimulation {
    pub world_width: f32,
    pub world_height: f32,
    pub alpha: f32,
    pub beta: f32,
    pub r: f32,
    pub v: f32,
    pub particles: Vec<sim::Particle>,
}

impl SerialSimulation {
    fn calculate_neighbours(&self) -> Vec<SideNeighbourCount> {
        let mut left_right_neighbours =
            vec![SideNeighbourCount { left: 0, right: 0 }; self.particles.len()];
        for (i, (selected, SideNeighbourCount { left, right })) in
            std::iter::zip(&self.particles, &mut left_right_neighbours).enumerate()
        {
            for current in self.particles[..i].iter().chain(&self.particles[i + 1..]) {
                let delta_x = sim::wrap(current.x - selected.x, self.world_width);
                let delta_y = sim::wrap(current.y - selected.y, self.world_height);
                if sim::within_radius(delta_x, delta_y, self.r) {
                    if sim::is_on_left(delta_x, delta_y, selected.heading) {
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
            p.x = sim::wrap(p.x + self.v * c, self.world_width);
            p.y = sim::wrap(p.y + self.v * s, self.world_height);
        }
    }
}

impl sim::Simulation for SerialSimulation {
    fn dump_to_file(&self, file: &mut File) -> std::io::Result<()> {
        file.write_all(
            self.particles
                .iter()
                .flat_map(sim::Particle::encode)
                .collect::<Vec<u8>>()
                .as_slice(),
        )
    }

    fn tick(&mut self) {
        self.update_particle_data(self.calculate_neighbours());
    }
}