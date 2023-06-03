# Neurotick
Rust library for neural networks with a DSL of keras-style.
People are welcome to contribute.
<br><br>

## Features
- Matrix operations are done via NDArray's and BLAS
- Dense and Input layer added
- Custom implementations of layers and activations functions
<br><br>

## Plans
- Add serialisation ✅ and deserialisation ✅
- Add activation functions ✅
- Add weight inits and randomisers
- Add several common layer implementations
- Add several custom layer implementations (compatible with GA but lack backpropagation)
<br><br>
- Implement Binary GA algorithm as a separate module (neurotick_ga)
- Reasearch back propagation algorithms in the current architecture
- Implement PSO algorithm together in the GA module
- Possibly implement SGD or Ada-related optimiser + PPO
<br><br>
- Restructure the library, enable features, experimental addons
- Simplify core trait objects and internal model data
- More unit tests
<br><br>

## Contribution
Feel free to contact me via: rufalgin@gmail.com or Telegram at @MakkiMax with your ideas to collaborate.

### Inspiration
The goal is to develop a framework for RL in rust

# Samples
Example of building a model

``` rust
use neurotick::{
    builder::builder::ModelBuilder, layer::{dense::Dense, input::Input},
    map, matrix::{meta::shape::Shape, nmatrix::NDMatrix},
};

/***/

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
let model = mb.build();
```

Example of calling a model

``` rust
let output_data: HashMap<String, NDMatrix> = model.propagate(&input_data);
```