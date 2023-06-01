use std::collections::HashMap;

use indexmap::IndexMap;

use crate::{
    builder::builder::ModelBuilderRc, layer::abs::GraphPropagationNode, matrix::nmatrix::NDMatrix,
};

pub struct Model {
    pub model_builder: ModelBuilderRc,
    pub input_layer_to_data_name: IndexMap<String, String>,
    pub output_layer_to_data_name: IndexMap<String, String>,
    pub sequential_prop: IndexMap<String, GraphPropagationNode>,
}

impl Model {
    pub fn propagate(&self, inputs: &HashMap<String, NDMatrix>) -> HashMap<String, NDMatrix> {
        let mut data_buffer: HashMap<String, NDMatrix> = HashMap::new();

        self.sequential_prop.iter().for_each(|seq| match seq.1 {
            GraphPropagationNode::DeadEnd(callable) => {
                let data_name = self.input_layer_to_data_name.get(seq.0);
                let data = match data_name {
                    Some(name) => inputs.get(name).unwrap(),
                    None => panic!("Missing branch layer: {}", seq.0),
                };
                let result = callable.propagate(data);
                data_buffer.insert(seq.0.clone(), result);
            }
            GraphPropagationNode::SingleInput(parent, callable) => {
                let data = data_buffer.get(parent).unwrap();
                let result = callable.propagate(data);
                data_buffer.insert(seq.0.clone(), result);
            }
            GraphPropagationNode::MultipleInput(parents, callable) => {
                let data: Vec<&NDMatrix> = parents
                    .iter()
                    .map(|p| &data_buffer.get(p).unwrap() as &NDMatrix)
                    .collect();
                let result = callable.propagate_multi(&data);
                data_buffer.insert(seq.0.clone(), result);
            }
        });

        let output = self
            .output_layer_to_data_name
            .iter()
            .map(
                |layer_and_data| match data_buffer.remove(layer_and_data.0) {
                    Some(data) => (layer_and_data.1.clone(), data),
                    None => panic!(
                        "Missing output for {} layer: {}",
                        layer_and_data.1, layer_and_data.0
                    ),
                },
            )
            .collect();
        return output;
    }
}
