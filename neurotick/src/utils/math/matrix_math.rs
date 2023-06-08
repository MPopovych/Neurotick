use crate::matrix::nmatrix::NDMatrix;

use super::fast_math::FMath;

pub struct MatrixMath;

impl MatrixMath {
    pub fn softmax_per_row(matrix: &NDMatrix) -> NDMatrix {
        let flattened = matrix
            .iter_rows()
            .map(|row| {
                let max = row
                    .iter()
                    .max_by(|x, y| x.total_cmp(y))
                    .unwrap_or_else(|| &0.0);
                let mut_row = row
                    .iter()
                    .map(|f| {
                        return FMath::fast_exponent(*f - max);
                    })
                    .collect::<Vec<f32>>();

                let sum: f32 = mut_row.iter().sum();
                if sum == 0.0 {
                    return vec![0.; mut_row.len()];
                }

                mut_row
                    .into_iter()
                    .map(|exp| exp / sum)
                    .collect::<Vec<f32>>()
            })
            .flatten()
            .collect::<Vec<f32>>();

        return NDMatrix::from_raw_vec(matrix.width, matrix.height, flattened);
    }
}
