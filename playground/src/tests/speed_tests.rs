#[cfg(test)]
mod test {
    use std::time::Instant;

    use neurotick::{
        activation::{relu::ReLu, tanh::Tanh},
        builder::builder::ModelBuilder,
        layer::{dense::Dense, input::Input},
        matrix::{meta::shape::Shape, nmatrix::NDMatrix},
        model::model::Model,
        suppliers::suppliers::{GlorothNormalSupplier, ZeroSupplier},
    };

    fn create_simple_model() -> Model {
        let input_1 = Input::new(Shape::Const(100), Shape::Repeat);

        let d1 = Dense::builder(20, || &input_1)
            .with_activation(ReLu::default())
            .build();

        let d2 = Dense::builder(10, || &d1)
            .with_activation(Tanh::default())
            .with_bias_init(ZeroSupplier::new())
            .with_weight_init(GlorothNormalSupplier::new())
            .build();

        let mb = ModelBuilder::from_straight(input_1, d2);

        return mb.build();
    }

    #[test]
    fn test_simple_model_speed() {
        // M1 took Elapsed: 4.13ms
        let iterations = 1000;
        let model = create_simple_model();
        let input = NDMatrix::constant(100, 3, 1.0);

        let now = Instant::now();
        let mut shown = false;
        for _ in 0..iterations {
            let output = model.propagate_single(input.clone());

            if !shown {
                shown = true;
                dbg!(output);
            }
        }

        println!("Elapsed: {:.2?}", now.elapsed());
    }

    #[test]
    fn test_multiply_simple_speed() {
        // M1 took Elapsed: 2.11ms
        let size = 1000usize;
        let iterations = 1000;

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
    }

    #[test]
    fn test_hadamard_simple_speed() {
        // M1 took Elapsed: 0.92ms
        let size = 1000usize;
        let iterations = 1000;

        let now = Instant::now();
        let mut shown = false;
        for _ in 0..iterations {
            let a = NDMatrix::constant(size, size, 1.);
            let b = NDMatrix::constant(size, size, 1.0 / size as f32);
            let _d = &a & &b;

            if !shown {
                shown = true;
                dbg!(_d);
            }
        }

        println!("Elapsed: {:.2?}", now.elapsed());
    }
}
