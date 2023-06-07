#[cfg(test)]
mod test {
    use crate::{
        activation::{relu::ReLu, tanh::Tanh},
        builder::builder::ModelBuilder,
        layer::{dense::Dense, input::Input, concat::Concat},
        matrix::{meta::shape::Shape, nmatrix::NDMatrix},
        suppliers::suppliers::{GlorothNormalSupplier, ZeroSupplier, RandomUniformSupplier}, utils::extensions::eq_vecs,
    };

    #[test]
    pub fn model_test_gloroth_weight() {
        let input_1 = Input::new(Shape::Const(10), Shape::Repeat);

        let d1 = Dense::builder(20, || &input_1)
            .with_activation(ReLu::default())
            .build();

        let d2 = Dense::builder(3, || &d1)
            .with_activation(Tanh::default())
            .with_bias_init(ZeroSupplier::new())
            .with_weight_init(GlorothNormalSupplier::new())
            .build();

        let mb = ModelBuilder::from_straight(input_1, d2);

        let model = mb.build();

        let input = NDMatrix::constant(10, 3, 1.0);
        let output = model.propagate_single(input);
        dbg!(&output);

        let rows = output
            .iter_rows()
            .map(|i| i.to_vec())
            .collect::<Vec<Vec<f32>>>();

        assert!(rows.len() > 1);

        let first_row = rows[0].clone();
        rows.iter().for_each(|row| {
            assert!(eq_vecs(row, &first_row));
        });
    }


    #[test]
    pub fn model_test_concat() {
        let input_1 = Input::new(Shape::Const(10), Shape::Repeat);

        let d1 = Dense::builder(5, || &input_1)
            .with_activation(ReLu::default())
            .build();

        let d2 = Dense::builder(3, || &input_1)
            .with_activation(Tanh::default())
            .with_bias_init(RandomUniformSupplier::new(1.0, -1.0))
            .with_weight_init(GlorothNormalSupplier::new())
            .build();

        let cocnat = Concat::new(|| vec![&d1, &d2]);

        let mb = ModelBuilder::from_straight(input_1, cocnat);

        let model = mb.build();

        let input = NDMatrix::from_supply(10, 3, RandomUniformSupplier::new(1.0, -1.0));
        let output = model.propagate_single(input);
        dbg!(&output);

        let rows = output
            .iter_rows()
            .map(|i| i.to_vec())
            .collect::<Vec<Vec<f32>>>();

        assert!(rows.len() > 1);

        let first_row = rows[0].clone();
        rows.iter().skip(1).for_each(|row| {
            assert!(!eq_vecs(row, &first_row));
        });
    }


}
