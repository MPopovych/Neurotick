#[cfg(test)]
mod tests {
    use crate::matrix::nmatrix::NDMatrix;
    use std::time::Instant;

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
        dbg!(d);
        // TODO assert
    }

    #[test]
    fn test_add_simple() {
        let a = NDMatrix::constant(32, 2, 1.);
        let b = NDMatrix::constant(32, 1, 1.);
        let d = &a + &b;
        dbg!(d);

        // TODO assert
    }

    #[test]
    fn test_multiply_simple_speed() {
        // M1 took Elapsed: 86.78ms
        let size = 100usize;
        let iterations = 100000000 / (size * size);

        let now = Instant::now();
        let mut shown = false;
        for _ in 0..iterations {
            let a = NDMatrix::constant(size, size, 1.);
            let b = NDMatrix::constant(size, size, 1.0 / size as f32);
            let _d = &a * &b;

            if !shown {
                shown = true;
                dbg!(_d);
            }
        }

        println!("Elapsed: {:.2?}", now.elapsed());
        dbg!(NDMatrix::constant(10, 10, 1.));
    }
}
