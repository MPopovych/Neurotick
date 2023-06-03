use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::{
    builder::graph_elements::BuilderNode,
    layer::abs::{ModelPropagationNode, LayerPropagateEnum},
    model::model::Model,
    utils::json_wrap::JsonWrap,
};

use super::model_reader::ModelReader;

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct ModelGraph {
    #[serde(flatten)]
    pub graph: IndexMap<String, BuilderNode>,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct ModelIO {
    pub inputs: IndexMap<String, String>,
    pub outputs: IndexMap<String, String>,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct ModelMeta {
    pub meta: IndexMap<String, JsonWrap>,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct ModelSerialized {
    pub io: ModelIO,
    pub graph: ModelGraph,
    #[serde(flatten)]
    pub meta: ModelMeta,
}

impl ModelSerialized {
    pub fn to_json(&self) -> String {
        return serde_json::to_string(self).unwrap();
    }
    pub fn to_json_pretty(&self) -> String {
        return serde_json::to_string_pretty(self).unwrap();
    }

    pub fn build_model(&self, reader: &ModelReader) -> Model {
        let node_meta_graph: IndexMap<String, ModelPropagationNode> = self
            .meta
            .meta
            .iter()
            .map(|meta| {
                let layer_name = meta.0.to_string();

                let parent_type_descriptor = self.graph.graph.get(&layer_name).unwrap();
                let type_name = parent_type_descriptor.type_name();

                let prop_enum: LayerPropagateEnum =
                    *reader.get_layer_di().create(&type_name, meta.1, reader);

                let node_num: ModelPropagationNode = match parent_type_descriptor {
                    BuilderNode::DeadEnd(_) => {
                        let cast = if let LayerPropagateEnum::SingleInput(single) = prop_enum {
                            single
                        } else {
                            panic!("Not a dead end builder node")
                        };
                        ModelPropagationNode::DeadEnd(cast)
                    }
                    BuilderNode::SingleParent(c) => {
                        let cast = if let LayerPropagateEnum::SingleInput(single) = prop_enum {
                            single
                        } else {
                            panic!("Not a single input builder node")
                        };
                        ModelPropagationNode::SingleInput(c.parent_name.clone(), cast)
                    }
                    BuilderNode::MultipleParent(c) => {
                        let cast = if let LayerPropagateEnum::MultipleInput(single) = prop_enum {
                            single
                        } else {
                            panic!("Not a multi input builder node")
                        };
                        ModelPropagationNode::MultipleInput(c.parent_names.clone(), cast)
                    }
                };

                return (layer_name, node_num);
            })
            .collect();

        Model {
            input_layer_to_data_name: self.io.inputs.clone(),
            output_layer_to_data_name: self.io.outputs.clone(),
            sequential_prop: node_meta_graph,
            builder_ref: self.graph.graph.clone(),
        }
    }
}
