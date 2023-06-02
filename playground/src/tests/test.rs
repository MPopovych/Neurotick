#[cfg(test)]
pub mod test {
    use std::collections::HashMap;

    use neurotick::{
        builder::builder::ModelBuilder,
        layer::{dense::Dense, input::Input},
        map,
        matrix::{meta::shape::Shape, nmatrix::NDMatrix}, serial::model_serial::ModelSerialized,
    };

    #[test]
    fn test_mb() {
        let input_1 = Input::new(Shape::Const(10000), Shape::Repeat);
        let d1 = Dense::new(300, || &input_1);
        let d2 = Dense::new(64, || &input_1);

        let d1_o = Dense::new(2, || &d1);
        let d2_o = Dense::new(3, || &d2);

        let input_2 = Input::new(Shape::Const(300), Shape::Repeat);
        let d3_o = Dense::new(4, || &input_2);

        let inputs = map! {
            input_1 => "input1".to_owned(),
            input_2 => "input2".to_owned(),
        };

        let outputs = map! {
            d1_o => "out1".to_owned(),
            d2_o => "out2".to_owned(),
            d3_o => "out3".to_owned(),
        };

        let mb = ModelBuilder::from(inputs, outputs);
        dbg!(&mb);
        let model = mb.build();
        let input_data = map! {
            "input1".to_owned() => NDMatrix::constant(10000, 6, 1.0),
            "input2".to_owned() => NDMatrix::constant(300, 5, 1.0),
        };
        let output_data: HashMap<String, NDMatrix> = model.propagate(&input_data);
        dbg!(&output_data);
    }

    #[test]
    fn test_small_mb() {
        let input_1 = Input::new(Shape::Const(10), Shape::Repeat);
        let d1 = Dense::new(20, || &input_1);
        let d2 = Dense::new(10, || &input_1);

        let d1_o = Dense::new(2, || &d1);
        let d2_o = Dense::new(3, || &d2);

        let input_2 = Input::new(Shape::Const(10), Shape::Repeat);
        let d3_o = Dense::new(4, || &input_2);

        let inputs = map! {
            input_1 => "input1".to_owned(),
            input_2 => "input2".to_owned(),
        };

        let outputs = map! {
            d1_o => "out1".to_owned(),
            d2_o => "out2".to_owned(),
            d3_o => "out3".to_owned(),
        };

        let mb = ModelBuilder::from(inputs, outputs);
        
        let model = mb.build();
        let serialized: ModelSerialized = model.to_serialized_model();
        let json = serialized.to_json_pretty();
        println!("{}", &json);
        let deserialized: ModelSerialized = serde_json::from_str(&json).unwrap();

        dbg!(&deserialized);
    }
}
