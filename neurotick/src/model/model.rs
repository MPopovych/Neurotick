use std::collections::HashMap;

use indexmap::IndexMap;

use crate::{
    builder::{graph_elements::{BuilderNode, ModelPropagationNode}, builder::ModelBuilder},
    matrix::nmatrix::NDMatrix,
    serial::model_serial::{ModelGraph, ModelIO, ModelMeta, ModelSerialized},
    utils::json_wrap::JsonWrap, map,
};

pub struct Model {
    pub input_layer_to_data_name: IndexMap<String, String>,
    pub output_layer_to_data_name: IndexMap<String, String>,
    pub sequential_prop: IndexMap<String, ModelPropagationNode>,
    pub builder_ref: IndexMap<String, BuilderNode>,
}

impl Model {

    pub fn propagate_single(&self, input: NDMatrix) -> NDMatrix {
        return self.propagate_single_input(input).remove(ModelBuilder::SINGLE_IO).unwrap()
    }

    pub fn propagate_single_input(&self, input: NDMatrix) -> HashMap<String, NDMatrix> {
        let input_map: HashMap<String, NDMatrix> = map! {
            ModelBuilder::SINGLE_IO.to_string() => input
        };
        return self.propagate(&input_map)
    }

    pub fn propagate_single_output(&self, inputs: HashMap<String, NDMatrix>) -> NDMatrix {
        return self.propagate(&inputs).remove(ModelBuilder::SINGLE_IO).unwrap()
    }

    pub fn propagate(&self, inputs: &HashMap<String, NDMatrix>) -> HashMap<String, NDMatrix> {
        let mut data_buffer: HashMap<String, NDMatrix> = HashMap::new();

        self.sequential_prop.iter().for_each(|seq| match seq.1 {
            ModelPropagationNode::DeadEnd(callable) => {
                let data_name = self.input_layer_to_data_name.get(seq.0);
                let data = match data_name {
                    Some(name) => inputs.get(name).unwrap(),
                    None => panic!("Missing branch layer: {}", seq.0),
                };
                let result = callable.propagate(data);
                data_buffer.insert(seq.0.clone(), result);
            }
            ModelPropagationNode::SingleInput(parent, callable) => {
                let data = data_buffer.get(parent).unwrap();
                let result = callable.propagate(data);
                data_buffer.insert(seq.0.clone(), result);
            }
            ModelPropagationNode::MultipleInput(parents, callable) => {
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

    pub fn to_serialized_model(&self) -> ModelSerialized {
        let io = ModelIO {
            inputs: self.input_layer_to_data_name.clone(),
            outputs: self.output_layer_to_data_name.clone(),
        };

        let graph = ModelGraph {
            graph: self.builder_ref.clone(),
        };

        let meta_map: IndexMap<String, JsonWrap> = self
            .sequential_prop
            .iter()
            .map(|node| (node.0.clone(), node.1.to_json()))
            .collect();
        let meta = ModelMeta { meta: meta_map };
        return ModelSerialized {
            io: io,
            graph: graph,
            meta: meta,
        };
    }

    pub fn to_json(&self) -> String {
        return self.to_serialized_model().to_json();
    }
    pub fn to_json_pretty(&self) -> String {
        return self.to_serialized_model().to_json_pretty();
    }
}
