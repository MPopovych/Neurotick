use std::fmt::Debug;

use indexmap::IndexMap;

use crate::{
    layer::abs::{LayerPropagateEnum, LayerRef},
    map,
    matrix::meta::node::LayerType,
    model::model::Model,
};

use super::graph_elements::{
    BuilderNode, DeadEndStruct, ModelPropagationNode, MultipleParentStruct, SingleParentStruct,
};

/**
 * Builds the neural model with default values
 */
pub struct ModelBuilder {
    /**
     * Maps the input layer reference into a named input
     */
    inputs: IndexMap<LayerRef, String>,
    /**
     * Maps the output layer reference into a named ouput
     */
    outputs: IndexMap<LayerRef, String>,
    /**
     * Holds the layer references into a processed builder node
     */
    graph: IndexMap<LayerRef, BuilderNode>,
}

impl ModelBuilder {
    pub fn build(&self) -> Model {
        let mut inputs: IndexMap<String, String> = IndexMap::new();
        let mut outputs: IndexMap<String, String> = IndexMap::new();

        let serialized: IndexMap<String, ModelPropagationNode> = self
            .graph
            .iter()
            .map(|entry| {
                let name = entry.1.layer_name();

                if let Some(key_value) = self.inputs.get_key_value(entry.0) {
                    inputs.insert(name.clone(), key_value.1.clone());
                }

                if let Some(key_value) = self.outputs.get_key_value(entry.0) {
                    outputs.insert(name.clone(), key_value.1.clone());
                }

                let instance = entry.0.borrow_ref().create_instance(name.clone());
                let graph_node = match entry.1 {
                    BuilderNode::DeadEnd(_) => match instance {
                        LayerPropagateEnum::SingleInput(b) => ModelPropagationNode::DeadEnd(b),
                        _ => panic!("{} is not a dead end graph node", name),
                    },
                    BuilderNode::SingleParent(s) => match instance {
                        LayerPropagateEnum::SingleInput(b) => {
                            ModelPropagationNode::SingleInput(s.parent_name.clone(), b)
                        }
                        _ => panic!("{} is not a dead end graph node", name),
                    },
                    BuilderNode::MultipleParent(s) => match instance {
                        LayerPropagateEnum::MultipleInput(b) => {
                            ModelPropagationNode::MultipleInput(s.parent_names.clone(), b)
                        }
                        _ => panic!("{} is not a dead end graph node", name),
                    },
                };
                (name, graph_node)
            })
            .collect::<IndexMap<_, _>>();

        let builder_ref: IndexMap<String, BuilderNode> = self
            .graph
            .iter()
            .map(|n| (n.1.layer_name(), n.1.clone()))
            .collect();

        return Model {
            input_layer_to_data_name: inputs,
            output_layer_to_data_name: outputs,
            sequential_prop: serialized,
            builder_ref: builder_ref,
        };
    }
}

impl ModelBuilder {
    /**
     * Helper constant for skipping definition if key in a single branched input/output model
     */
    pub const SINGLE_IO: &str = "DEF_IO";

    pub fn from_straight(input: LayerRef, output: LayerRef) -> ModelBuilder {
        return Self::from(
            map!(input => Self::SINGLE_IO.to_owned()),
            map!(output => Self::SINGLE_IO.to_owned()),
        );
    }

    pub fn from_single_o(inputs: IndexMap<LayerRef, String>, output: LayerRef) -> ModelBuilder {
        return Self::from(inputs, map!(output => Self::SINGLE_IO.to_owned()));
    }

    pub fn from_single_i(input: LayerRef, outputs: IndexMap<LayerRef, String>) -> ModelBuilder {
        return Self::from(map!(input => Self::SINGLE_IO.to_owned()), outputs);
    }

    pub fn from(
        inputs: IndexMap<LayerRef, String>,
        outputs: IndexMap<LayerRef, String>,
    ) -> ModelBuilder {
        inputs.iter().for_each(|input_entry| {
            if let LayerType::DeadEnd = input_entry.0.borrow_ref().get_node() {
            } else {
                panic!("Bad input node, should be a DeadEnd node impl")
            }
        });
        let mut graph: IndexMap<LayerRef, BuilderNode> = IndexMap::new();

        outputs.iter().for_each(|entry| {
            Self::iterate_nodes(&mut graph, entry.0, 0);
        });

        return ModelBuilder {
            inputs,
            outputs,
            graph,
        }
        .into();
    }

    fn iterate_nodes(
        graph: &mut IndexMap<LayerRef, BuilderNode>,
        current_layer: &LayerRef,
        depth: usize,
    ) -> String {
        if let Some(existing) = graph.get(current_layer) {
            return existing.layer_name();
        }
        match &current_layer.borrow_ref().get_node() {
            LayerType::DeadEnd => {
                let name = format!("{}_{}", current_layer.type_name(), graph.len());
                let builder_node = DeadEndStruct {
                    layer_name: name.clone(),
                    type_name: current_layer.type_name().to_string(),
                };
                graph.insert(current_layer.clone(), BuilderNode::DeadEnd(builder_node));
                return name;
            }
            LayerType::SingleParent(parent) => {
                let parent_name = Self::iterate_nodes(graph, parent, depth + 1);
                let name = format!("{}_{}", current_layer.type_name(), graph.len());
                let builder_node = SingleParentStruct {
                    layer_name: name.clone(),
                    type_name: current_layer.type_name().to_string(),
                    parent_name,
                };
                graph.insert(
                    current_layer.clone(),
                    BuilderNode::SingleParent(builder_node),
                );
                return name;
            }
            LayerType::MultipleParent(parents) => {
                let parent_names: Vec<String> = parents
                    .iter()
                    .map(|p| Self::iterate_nodes(graph, p, depth + 1))
                    .collect();
                let name = format!("{}_{}", current_layer.type_name(), graph.len());
                let builder_node = MultipleParentStruct {
                    layer_name: name.clone(),
                    type_name: current_layer.type_name().to_string(),
                    parent_names,
                };
                graph.insert(
                    current_layer.clone(),
                    BuilderNode::MultipleParent(builder_node),
                );
                return name;
            }
        }
    }
}

impl Debug for ModelBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelBuilder")
            .field(
                "inputs",
                &self
                    .inputs
                    .iter()
                    .map(|i| (i.1, self.graph.get(i.0).unwrap().layer_name()) as (&String, String))
                    .collect::<IndexMap<_, _>>(),
            )
            .field(
                "outputs",
                &self
                    .outputs
                    .iter()
                    .map(|i| (self.graph.get(i.0).unwrap().layer_name(), i.1) as (String, &String))
                    .collect::<IndexMap<_, _>>(),
            )
            .field("graph", &self.graph.iter().map(|e| e.1).collect::<Vec<_>>())
            .finish()
    }
}
