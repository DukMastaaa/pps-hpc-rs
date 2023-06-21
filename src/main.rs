mod serial;
mod sim;

use std::fs::File;
use std::io::prelude::*;

use rand::prelude::Distribution;

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
    let dist = sim::RectDistribution {
        width: params.world_width,
        height: params.world_height,
    };
    let particles: Vec<sim::Particle> = (0..params.particle_count)
        .map(|_| dist.sample(&mut rng))
        .collect();

    let mut sim: Box<dyn sim::Simulation> = Box::new(serial::SerialSimulation {
        world_width: params.world_width,
        world_height: params.world_height,
        alpha: params.alpha,
        beta: params.beta,
        r: params.r,
        v: params.v,
        particles,
    });

    for tick in 0..=params.tick_count {
        if tick % params.dump_skip_size == 0 {
            sim.dump_to_file(&mut dump_file)?
        }
        sim.tick();
    }

    Ok(())
}
