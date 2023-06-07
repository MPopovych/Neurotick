/**
 * Wrapper around a library as a single entry point
 */

pub struct FMath;

impl FMath {
    pub fn fast_exponent(value: f32) -> f32 {
        fast_math::exp(value)
    }

    pub fn fast_sigmoid(value: f32) -> f32 {
        1.0 / (1.0 + Self::fast_exponent(-value))
    }

    pub fn fast_tanh(value: f32) -> f32 {
        let d = Self::fast_exponent(2.0 * value);
        return (d - 1.0) / (d + 1.0)
    }

    pub fn eq_approx(lhs: f32, rhs: f32, eps: f32) -> bool {
        let delta = (lhs - rhs).abs();
        return delta <= eps
    }
}


#[cfg(test)]
mod test {
    use super::FMath;


    #[test]
    fn test_eq_approx() {
        let value_1: f32 = 1.0;
        let value_2: f32 = 100.0;
        assert!(!FMath::eq_approx(value_1, value_2, 0.01));
        assert!(FMath::eq_approx(value_1, value_2, 99.0));

        let value_3: f32 = -1.0;
        let value_4: f32 = -100.0;
        assert!(!FMath::eq_approx(value_3, value_4, 0.01));
        assert!(FMath::eq_approx(value_3, value_4, 99.0));

        let value_5: f32 = -100.0;
        let value_6: f32 = 100.0;
        assert!(!FMath::eq_approx(value_5, value_6, 0.01));

        let value_7: f32 = 99.0;
        let value_8: f32 = 100.0;
        assert!(FMath::eq_approx(value_7, value_8, 1.0));
    }

    #[test]
    fn test_exp() {
        let tolerance = 0.006;

        let test_values: [f32; 11] = [-20.0, -5.0, -2.0, -1.0, -0.3, -0.01, 0.0, 0.01, 0.3, 1.0, 2.0];
        let ref_values: [f32; 11] = [2.06e-9, 0.006737, 0.1353, 0.3678, 0.74081, 0.99, 1.0, 1.01, 1.349, 2.7182, 7.3890];

        for i in 0..test_values.len() {
            let v = FMath::fast_exponent(test_values[i]);
            assert!(FMath::eq_approx(v, ref_values[i], tolerance));
        }
    }

    #[test]
    fn test_tanh() {
        let tolerance = 0.001;

        let test_values: [f32; 13] = [-20.0, -5.0, -2.0, -1.0, -0.3, -0.01, 0.0, 0.01, 0.3, 1.0, 2.0, 5.0, 10.0];
        let ref_values: [f32; 13] = [-1.0, -0.9999, -0.9640, -0.76159, -0.2913, -0.00999, 0.0, 0.00999, 0.2913, 0.76159, 0.9640, 0.9999, 0.9999];

        for i in 0..test_values.len() {
            let v = FMath::fast_tanh(test_values[i]);
            assert!(FMath::eq_approx(v, ref_values[i], tolerance));
        }
    }
}