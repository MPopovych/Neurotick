#[cfg(test)]
mod tests {
    use crate::matrix::nmatrix::NDMatrix;

    #[test]
    fn test_from_raw_vec() {
        let vector = vec![1., 2., 3., 4., 5., 6.];
        let a = NDMatrix::from_raw_vec(3, 2, vector.clone());
        let mut index = 0;
        a.iter_all()
            .map(|v| {
                let t = (index, v);
                index += 1;
                return t;
            })
            .for_each(|(index, matrix_value)| assert!(vector[index].eq(matrix_value)));
    }

    #[test]
    fn test_row_iter() {
        let vector = vec![0., 0., 0., 1., 1., 1., 2., 2., 2.];
        let a = NDMatrix::from_raw_vec(3, 3, vector.clone());
        let mut index = 0;
        a.iter_rows()
            .map(|v| {
                let t = (index, v);
                index += 1;
                return t;
            })
            .for_each(|(index, row)| {
                let matching = row.iter().all(|f| f.eq(&(index as f32)));
                assert!(matching)
            });
    }

    #[test]
    fn test_col_iter() {
        let vector = vec![0., 1., 2., 0., 1., 2., 0., 1., 2.];
        let a = NDMatrix::from_raw_vec(3, 3, vector.clone());
        let mut index = 0;
        a.iter_columns()
            .map(|v| {
                let t = (index, v);
                index += 1;
                return t;
            })
            .for_each(|(index, row)| {
                let matching = row.iter().all(|f| f.eq(&(index as f32)));
                assert!(matching)
            });
    }

    #[test]
    fn test_multiply_simple() {
        let a = NDMatrix::constant(32, 10, 1.);
        let b = NDMatrix::constant(2, 32, 1.0 / 32 as f32); // 2 features from 32 features
        let d = &a * &b;
        d.iter_all().for_each(|f| {
            assert!(f.eq(&1.0))
        });
    }

    #[test]
    fn test_add_simple() {
        let a = NDMatrix::constant(32, 2, 1.);
        let b = NDMatrix::constant(32, 1, 1.);
        let d = &a + &b;
        d.iter_all().for_each(|f| {
            assert!(f.eq(&2.0))
        });
    }

}
