use crate::sim;
use core::arch::x86_64::{__m256, __m256i};
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

/// Returns a vector with the elements in src selected by the mask,
/// all moved to the right side.
/// (Peter Cordes, 30/4/2016, https://stackoverflow.com/a/36951611)
/// SAFETY: Only use this if you've confirmed AVX2 support on x86_64 arch.
unsafe fn compress256(src: __m256, mask: u32) -> __m256 {
    use std::arch::x86_64::*;
    // unpack each bit to a byte
    let mut expanded_mask: u64 = _pdep_u64(mask as u64, 0x0101010101010101);
    // replicate each bit to fill its byte
    expanded_mask *= 0xFF;
    // the identity shuffle for vpermps, packed to one index per byte
    const IDENTITY_INDICES: u64 = 0x0706050403020100;
    // wanted_indices[i+7:i] is the ith byte of identity_indices
    // whose corresponding element in expanded_mask is high.
    // after some conversions, this gives a permute control vector which
    // sends elements selected by mask to the lower slots.
    let wanted_indices: u64 = _pext_u64(IDENTITY_INDICES, expanded_mask);
    // convert u64 to i128, zero-extending
    let bytevec: __m128i = _mm_cvtsi64_si128(wanted_indices as i64);
    // convert each of the lower bytes of bytevec into 32 bits
    let shufmask: __m256i = _mm256_cvtepu8_epi32(bytevec);
    // finally, permute the input.
    _mm256_permutevar8x32_ps(src, shufmask)
}

/// Vectorised version of sim::wrap, which wraps x within [0, max].
/// SAFETY: Only use this if you've confirmed AVX2 support on x86_64 arch.
unsafe fn wrap256(x: __m256, max: __m256) -> __m256 {
    use std::arch::x86_64::*;
    // bitmask where x > max
    let x_gt_max: __m256 = _mm256_cmp_ps(x, max, _CMP_GT_OQ);
    // bitmask where x < 0
    let x_lt_zero: __m256 = _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_LT_OQ);
    // blend(a, b, mask) chooses from b if mask is high, else a.
    let mut result: __m256 = _mm256_blendv_ps(
        x,
        _mm256_sub_ps(x, max),
        x_gt_max
    );
    result = _mm256_blendv_ps(
        result,
        _mm256_add_ps(x, max),
        x_lt_zero
    );
    result
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
