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
            neighbours: 0,
        }
    }
}

/// Returns a vector with the elements in src selected by the mask,
/// all moved to the right side.
/// (Peter Cordes, 30/4/2016, https://stackoverflow.com/a/36951611)
/// SAFETY: Only use this if you've confirmed AVX2 support on x86_64 arch.
#[warn(clippy::undocumented_unsafe_blocks)]
unsafe fn compress256(src: __m256, mask: u32) -> __m256 {
    use std::arch::x86_64::*;
    // SAFETY: No memory accesses. CPU support assumed.
    unsafe {
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
}

/// Vectorised version of sim::wrap, which wraps x within [0, max].
/// SAFETY: Only use this if you've confirmed AVX2 support on x86_64 arch.
unsafe fn wrap256(x: __m256, max: __m256) -> __m256 {
    use std::arch::x86_64::*;
    // SAFETY: No memory accesses. CPU support assumed.
    unsafe {
        // bitmask where x > max
        let x_gt_max: __m256 = _mm256_cmp_ps(x, max, _CMP_GT_OQ);
        // bitmask where x < 0
        let x_lt_zero: __m256 = _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_LT_OQ);
        // blend(a, b, mask) chooses from b if mask is high, else a.
        let mut result: __m256 = _mm256_blendv_ps(x, _mm256_sub_ps(x, max), x_gt_max);
        result = _mm256_blendv_ps(result, _mm256_add_ps(x, max), x_lt_zero);
        result
    }
}

pub struct SpecificParams {}

/// Stores the common parameters in vectors for ease of access.
struct VectorisedConstants {
    radius_squared: __m256,
    world_width: __m256,
    world_height: __m256,
    x_indices: __m256i,
    y_indices: __m256i,
    particle_count: usize,
}

pub struct SIMDSimulation {
    common: sim::CommonParams,
    #[allow(dead_code)]
    specific: SpecificParams,
    particles: Vec<Particle>,
    vec_consts: VectorisedConstants,
}

const SCALE: i32 = 4;
const CHUNK_SIZE: usize = std::mem::size_of::<__m256>() / std::mem::size_of::<f32>();

impl SIMDSimulation {
    pub fn new(
        common: sim::CommonParams,
        specific: SpecificParams,
        particles: Vec<sim::GenericParticle>,
    ) -> Self {
        use std::arch::x86_64::*;
        panic_if_x86_features_not_detected!("avx2", "fma", "bmi2");
        // SAFETY: No memory accesses. CPU support assumed.
        let radius_squared = unsafe { _mm256_set1_ps(common.r * common.r) };
        let world_width = unsafe { _mm256_set1_ps(common.world_width) };
        let world_height = unsafe { _mm256_set1_ps(common.world_height) };
        let x_indices = unsafe { _mm256_set_epi32(0, 4, 8, 12, 16, 20, 24, 28) };
        let y_indices = unsafe { _mm256_add_epi32(x_indices, _mm256_set1_epi32(1)) };
        let particle_count = particles.len();
        Self {
            common,
            specific,
            particles: particles.into_iter().map(Particle::from).collect(),
            vec_consts: VectorisedConstants {
                radius_squared,
                world_width,
                world_height,
                x_indices,
                y_indices,
                particle_count,
            },
        }
    }

    unsafe fn filter_close_particles_in_chunk(
        &self,
        selected_index_within_chunk: Option<usize>,
        particles_in_chunk: &[Particle; 8],
        delta_x_chunk: &mut [f32; 8],
        delta_y_chunk: &mut [f32; 8],
        x_selected: __m256,
        y_selected: __m256,
    ) -> u8 {
        use std::arch::x86_64::*;

        // SAFETY: blah
        let base_address = particles_in_chunk.as_ptr() as *const f32;
        let x_chunk =
            unsafe { _mm256_i32gather_ps::<SCALE>(base_address, self.vec_consts.x_indices) };
        let y_chunk =
            unsafe { _mm256_i32gather_ps::<SCALE>(base_address, self.vec_consts.y_indices) };

        let particles_in_range: u8 = unsafe {
            let delta_x = wrap256(
                _mm256_sub_ps(x_chunk, x_selected),
                self.vec_consts.world_width,
            );
            let delta_y = wrap256(
                _mm256_sub_ps(y_chunk, y_selected),
                self.vec_consts.world_height,
            );
            let distance_squared =
                _mm256_fmadd_ps(delta_x, delta_x, _mm256_mul_ps(delta_y, delta_y));

            // Check if distances squared are less than radius squared.
            let compare_result =
                _mm256_cmp_ps(distance_squared, self.vec_consts.radius_squared, _CMP_LE_OQ);
            // Convert to an integer mask.
            // If the current chunk contains the selected particle,
            // we don't want the particle to count itself as a neighbour.
            // We must set the specific bit in the mask low.
            let neighbour_particle_mask: u32 = {
                // movemask only sets the lower 8 bits, so this is really a u8.
                let movemask = _mm256_movemask_ps(compare_result) as u32;
                match selected_index_within_chunk {
                    None => movemask,
                    Some(i) => movemask & !(1 << i),
                }
            };

            // Move all the elements in delta for which the mask is HIGH
            // to one side. https://stackoverflow.com/a/36951611
            let rearranged_delta_x = compress256(delta_x, neighbour_particle_mask);
            let rearranged_delta_y = compress256(delta_y, neighbour_particle_mask);
            // Store these coordinates in buffers.
            // The low-mask elements are still present in rearranged_delta,
            // but we write the entire vector as we allocated 8 floats larger than needed.
            _mm256_storeu_ps(delta_x_chunk.as_mut_ptr(), rearranged_delta_x);
            _mm256_storeu_ps(delta_y_chunk.as_mut_ptr(), rearranged_delta_y);

            // The number of particles in range is the number of 1s in the mask,
            // which must be between 0 and 8. This can fit in u8.
            neighbour_particle_mask.count_ones() as u8
        };

        particles_in_range
    }

    /// Writes the x and y coordinates of particles in range with particle_idx to
    /// delta_x_buf and delta_y_buf, and returns how many particles were written.
    ///
    /// SAFETY: Only use this if you've confirmed AVX2 support on x86_64 arch.
    /// Requires delta_x_buf and delta_y_buf to have enough space to store the particles,
    /// with possible overwrite by 8 elements.
    /// Recommended to allocate int(ceil(particle_count/8.0)) to be safe.
    unsafe fn filter_all_close_particles(
        &self,
        particle_idx: usize,
        delta_x_buf: &mut [f32],
        delta_y_buf: &mut [f32],
    ) -> usize {
        use std::arch::x86_64::*;
        let selected = &self.particles[particle_idx];
        let x_selected = unsafe { _mm256_set1_ps(selected.x) };
        let y_selected = unsafe { _mm256_set1_ps(selected.y) };

        let particle_count = self.particles.len();
        let mut filtered_particle_count: usize = 0;

        // We iterate across particles in chunks, and calculate
        // what chunk the selected particle belongs to.
        let chunks = particle_count / CHUNK_SIZE;
        let selected_chunk = particle_idx / CHUNK_SIZE;
        let selected_index_within_chunk = particle_idx % CHUNK_SIZE;
        for chunk in 0..chunks {
            let selected_index_within_chunk = if chunk == selected_chunk {
                Some(selected_index_within_chunk)
            } else {
                None
            };
            let particles_in_chunk = self.particles[chunk * CHUNK_SIZE..(chunk + 1) * CHUNK_SIZE]
                .try_into()
                .unwrap();
            let delta_x_chunk = &mut delta_x_buf
                [filtered_particle_count..filtered_particle_count + 8]
                .try_into()
                .unwrap();
            let delta_y_chunk = &mut delta_y_buf
                [filtered_particle_count..filtered_particle_count + 8]
                .try_into()
                .unwrap();
            // TODO: justify these memory accesses
            unsafe {
                filtered_particle_count += self.filter_close_particles_in_chunk(
                    selected_index_within_chunk,
                    particles_in_chunk,
                    delta_x_chunk,
                    delta_y_chunk,
                    x_selected,
                    y_selected,
                ) as usize;
            }
        }
        // Finish remaining particles serially.
        for i in chunks * CHUNK_SIZE..particle_count {
            if i == particle_idx {
                continue;
            }
            let current = &self.particles[i];
            let delta_x = sim::wrap(current.x - selected.x, self.common.world_width);
            let delta_y = sim::wrap(current.y - selected.y, self.common.world_height);
            if sim::within_radius(delta_x, delta_y, self.common.r) {
                delta_x_buf[filtered_particle_count] = current.x;
                delta_y_buf[filtered_particle_count] = current.y;
                filtered_particle_count += 1;
            }
        }

        filtered_particle_count
    }

    unsafe fn count_left_right_neighbours(
        &self,
        delta_x_buf: &mut [f32],
        delta_y_buf: &mut [f32],
        selected_heading: f32,
        filtered_particle_count: usize,
    ) -> sim::SideNeighbourCount {
        use std::arch::x86_64::*;
        let (s, c) = f32::sin_cos(selected_heading);
        let selected_cos_heading = unsafe { _mm256_set1_ps(c) };
        let selected_sin_heading = unsafe { _mm256_set1_ps(s) };
        let mut left = 0;
        let mut right = 0;
        let chunks = filtered_particle_count / CHUNK_SIZE;
        for chunk in 0..chunks {
            // TODO: justify memory access
            unsafe {
                let delta_x = _mm256_loadu_ps(delta_x_buf.as_ptr().add(chunk * CHUNK_SIZE));
                let delta_y = _mm256_loadu_ps(delta_y_buf.as_ptr().add(chunk * CHUNK_SIZE));
                // Use FMA to calculate sim::is_on_left
                let cross_product = _mm256_fmadd_ps(
                    delta_x,
                    selected_sin_heading,
                    _mm256_mul_ps(delta_y, selected_cos_heading)
                );
                // _mm256_movemask_ps already looks at the MSB of the elements
                // to determine the int8 mask bit. We don't care about behaviour
                // of special cases like -NaN and -0, so we call movemask
                // instead of _mm256_cmp_ps then movemask.
                let mask = _mm256_movemask_ps(cross_product);
                // High bit indicates negative result, meaning particle is on the right.
                let particles_on_right = mask.count_ones() as u8;
                right += particles_on_right;
                left += 8 - particles_on_right;  // 8 bits total; the low bits indicate left
            }
        }
        // Finish remaining particles serially.
        for i in chunks * CHUNK_SIZE..filtered_particle_count {
            if sim::is_on_left(delta_x_buf[i], delta_y_buf[i], selected_heading) {
                left += 1;
            } else {
                right += 1;
            }
        }

        sim::SideNeighbourCount { left, right }
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
