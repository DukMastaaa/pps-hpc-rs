mod serial;
mod sim;
mod simd;

use std::fs::File;
use std::io::prelude::*;

use rand::prelude::Distribution;

const MAGIC_BIT_PATTERN: u64 = 0x1234;

struct Cli {
    tick_count: usize,
    dump_skip_size: usize,
    particle_count: usize,
    common: sim::CommonParams,
    file_name: String,
}

impl Cli {
    fn encode(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        // These aren't the same size so I can't use [u8; N]::concat()
        bytes.extend(self.tick_count.to_le_bytes());
        bytes.extend(self.particle_count.to_le_bytes());
        bytes.extend(self.common.world_width.to_le_bytes());
        bytes.extend(self.common.world_height.to_le_bytes());
        bytes.extend(self.dump_skip_size.to_le_bytes());
        bytes.extend(MAGIC_BIT_PATTERN.to_le_bytes());
        bytes
    }
}

fn main() -> std::io::Result<()> {
    let common = sim::CommonParams {
        world_width: 100.0,
        world_height: 100.0,
        alpha: f32::to_radians(180.0),
        beta: f32::to_radians(70.0),
        r: 5.0,
        v: 0.67,
    };
    let cli = Cli {
        tick_count: 100,
        dump_skip_size: 1,
        particle_count: 1000,
        common,
        file_name: "test.dump".to_string(),
    };

    let mut dump_file = File::create(cli.file_name.clone())?;
    dump_file.write_all(&cli.encode())?;

    let rng = rand::thread_rng();
    let dist = sim::RectDistribution {
        width: cli.common.world_width,
        height: cli.common.world_height,
    };
    let particles = dist.sample_iter(rng).take(cli.particle_count).collect();

    let mut sim: Box<dyn sim::Simulate> = Box::new(serial::SerialSimulation::new(
        common,
        serial::SpecificParams {},
        particles,
    ));

    for tick in 0..=cli.tick_count {
        if tick % cli.dump_skip_size == 0 {
            sim.dump_to_file(&mut dump_file)?
        }
        sim.tick();
    }

    Ok(())
}
