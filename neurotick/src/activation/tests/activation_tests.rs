#[cfg(test)]
pub mod test {
    use crate::{
        activation::{abs::Activation, lerelu::LeakyReLu, relu::ReLu, softmax::SoftMax},
        matrix::nmatrix::NDMatrix,
    };

    #[test]
    fn test_relu() {
        let cap = 2.0;
        let relu = ReLu { cap: cap };
        let neg_array = NDMatrix::from_raw_vec(3, 2, vec![-2.0, -1.0, -0.1, -3.0, -1.2, -0.001]);
        let neg_array_result = relu.apply(&neg_array);
        dbg!(&neg_array_result);
        let all_zero = neg_array_result.iter_all().all(|f| f.eq(&0.0));
        assert!(all_zero);

        let pos_array = NDMatrix::from_raw_vec(3, 2, vec![2.0, 1.0, 0.1, 3.0, 1.2, 0.001]);
        let pos_array_result = relu.apply(&pos_array);
        dbg!(&pos_array_result);
        let all_non_zero = pos_array_result.iter_all().all(|f| *f > 0.0 && *f <= cap);
        assert!(all_non_zero)
    }

    #[test]
    fn test_lerelu() {
        let lerelu = LeakyReLu { beta: 0.03 };
        let neg_array = NDMatrix::from_raw_vec(3, 2, vec![-2.0, -1.0, -0.1, -3.0, -1.2, -0.001]);
        let neg_array_result = lerelu.apply(&neg_array);
        dbg!(&neg_array_result);
        let all_zero = neg_array_result.iter_all().all(|f| *f < 0.0);
        assert!(all_zero);

        let pos_array = NDMatrix::from_raw_vec(3, 2, vec![2.0, 1.0, 0.1, 3.0, 1.2, 0.001]);
        let pos_array_result = lerelu.apply(&pos_array);
        dbg!(&pos_array_result);
        let all_non_zero = pos_array_result.iter_all().all(|f| *f > 0.0);
        assert!(all_non_zero)
    }

    #[test]
    fn test_softmax() {
        let softmax = SoftMax {};
        let multi_row = NDMatrix::from_raw_vec(3, 2, vec![-2.0, 2.0, -0.1, 107.0, -1.2, -0.001]);
        let multi_row_result = softmax.apply(&multi_row);
        dbg!(&multi_row_result);
    }
}
