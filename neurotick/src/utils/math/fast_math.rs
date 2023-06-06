/**
 * Wrapper around a library as a single entry point
 */

pub struct FMath;

impl FMath {
    pub fn fast_exponent(value: f32) -> f32 {
        fast_math::exp(value)
    }
}
