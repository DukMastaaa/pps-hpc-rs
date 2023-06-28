use crate::sim::{self, CommonParams};
use core::arch::x86_64;
use std::{fs::File, io::Write};

macro_rules! panic_if_x86_features_not_detected {
    // thank you @paelias
    ( $( $feature:tt ),* ) => {
        $(
            if !is_x86_feature_detected!($feature) {
                panic!(concat!($feature, " feature not detected!"));
            }
        )*
    };
}

#[repr(C)]
struct Particle {
    x: f32,
    y: f32,
    heading: f32,
    // no 24-bit id field
    neighbours: u32, // 32-bit to pad out to 128 bits
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
        bytes.extend((self.neighbours as u8).to_le_bytes());
        bytes
    }
}

impl From<sim::GenericParticle> for Particle {
    fn from(value: sim::GenericParticle) -> Self {
        Self {
            x: value.x,
            y: value.y,
            heading: value.heading,
            neighbours: 0
        }
    }
}

pub struct SpecificParams {}

pub struct SIMDSimulation {
    common: sim::CommonParams,
    #[allow(dead_code)]
    specific: SpecificParams,
    particles: Vec<Particle>,
}

impl SIMDSimulation {
    pub fn new(
        common: sim::CommonParams,
        specific: SpecificParams,
        particles: Vec<sim::GenericParticle>,
    ) -> Self {
        panic_if_x86_features_not_detected!("avx2", "fma", "bmi2");
        Self {
            common,
            specific,
            particles: particles.into_iter().map(Particle::from).collect(),
        }
    }
}

impl sim::Simulate for SIMDSimulation {
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
        // self.update_particle_data(self.calculate_neighbours());
    }
}
