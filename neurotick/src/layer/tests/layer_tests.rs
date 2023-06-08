#[cfg(test)]
mod test {
    use indexmap::IndexMap;

    use crate::{
        builder::builder::ModelBuilder,
        layer::{
            abs::LayerRef, concat::Concat, dense::Dense, direct::Direct, flatten::Flatten,
            input::Input,
        },
        map,
        matrix::{meta::shape::Shape, nmatrix::NDMatrix},
    };

    #[test]
    fn dense_layer_test() {
        let input = Input::new(Shape::Const(3), Shape::Const(2));
        let dense = Dense::new(4, || &input);
        let mb = ModelBuilder::from_straight(input, dense);
        let model = mb.build();

        let input_data = NDMatrix::constant(3, 2, 1.0);
        let output_data = model.propagate_single(input_data);
        assert!(output_data.width == 4 && output_data.height == 2);
        dbg!(&output_data);
    }

    #[test]
    fn direct_layer_test() {
        let input = Input::new(Shape::Const(3), Shape::Const(2));
        let direct = Direct::new(|| &input);
        let mb = ModelBuilder::from_straight(input, direct);
        let model = mb.build();

        let input_data = NDMatrix::constant(3, 2, 1.0);
        let output_data = model.propagate_single(input_data);
        assert!(output_data.width == 3 && output_data.height == 2);
        dbg!(&output_data);
    }

    #[test]
    fn concat_layer_test() {
        let input_1 = Input::new(Shape::Const(3), Shape::Const(2));
        let input_2 = Input::new(Shape::Const(5), Shape::Const(2));

        let concat = Concat::new(|| vec![&input_1, &input_2]);

        let input_map: IndexMap<LayerRef, String> = map! {
            input_1 => "1".to_owned(),
            input_2 => "2".to_owned()
        };

        let mb = ModelBuilder::from_single_o(input_map, concat);
        let model = mb.build();

        let input_data_1 = NDMatrix::constant(3, 2, 1.0);
        let input_data_2 = NDMatrix::constant(5, 2, -1.0);
        let input_data_map = map! {
            "1".to_owned() => input_data_1,
            "2".to_owned() => input_data_2,
        };

        let output_data = model.propagate_single_output(input_data_map);
        assert!(output_data.width == 8 && output_data.height == 2);
        dbg!(&output_data);
    }

    #[test]
    fn flatten_full_const_layer_test() {
        let input = Input::new(Shape::Const(3), Shape::Const(2));
        let flatten = Flatten::new(|| &input);
        let mb = ModelBuilder::from_straight(input, flatten);
        let model = mb.build();

        let input_data = NDMatrix::constant(3, 2, 1.0);
        let output_data = model.propagate_single(input_data);
        assert!(output_data.width == 6 && output_data.height == 1);
        dbg!(&output_data);
    }

    #[test]
    fn flatten_const_variable_layer_test() {
        let input = Input::new(Shape::Const(3), Shape::Variable);
        let flatten = Flatten::new(|| &input);

        assert!(
            matches!(flatten.get_shape().0, Shape::Variable)
                && flatten.get_shape().1.unwrap_to_conts() == 1
        );

        let mb = ModelBuilder::from_straight(input, flatten);
        let model = mb.build();

        let input_data = NDMatrix::constant(3, 3, 1.0);
        let output_data = model.propagate_single(input_data);
        assert!(output_data.width == 9 && output_data.height == 1);
        dbg!(&output_data);
    }

    #[test]
    fn flatten_full_variable_layer_test() {
        let input = Input::new(Shape::Variable, Shape::Variable);
        let flatten = Flatten::new(|| &input);

        assert!(
            matches!(flatten.get_shape().0, Shape::Variable)
                && flatten.get_shape().1.unwrap_to_conts() == 1
        );

        let mb = ModelBuilder::from_straight(input, flatten);
        let model = mb.build();

        let input_data = NDMatrix::constant(3, 3, 1.0);
        let output_data = model.propagate_single(input_data);
        assert!(output_data.width == 9 && output_data.height == 1);
        dbg!(&output_data);
    }
}
