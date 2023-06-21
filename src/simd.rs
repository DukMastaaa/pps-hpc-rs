use crate::sim;
use std::{fs::File, io::Write};
use core::arch::x86_64;

pub struct SIMDSimulation {
    params: sim::CommonParams,
    particles: Vec<sim::Particle>,
}

macro_rules! panic_if_x86_features_not_detected {
    ($feature:tt) => {
        if !is_x86_feature_detected!($feature) {
            panic!(concat!($feature, " feature not detected!"));
        }
    };
    ($feature:tt $(, $features:tt)*) => {{
        panic_if_x86_features_not_detected!($feature);
        panic_if_x86_features_not_detected!($($features),*);
    }}
}

impl SIMDSimulation {
    fn check_x86_features_detected() {
        // panic_if_x86_features_not_detected!("avx2");
        panic_if_x86_features_not_detected!("avx2", "fma", "bmi2");
    }

    pub fn new(params: sim::CommonParams, particles: Vec<sim::Particle>) -> Self {
        Self { params, particles }
    }

}

impl sim::Simulation for SIMDSimulation {
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
        // self.update_particle_data(self.calculate_neighbours());
    }
}
