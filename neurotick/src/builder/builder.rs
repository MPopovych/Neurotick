use std::{fmt::Debug, rc::Rc};

use indexmap::IndexMap;

use crate::{
    layer::abs::{GraphPropagationNode, LBRef, LayerPropagateEnum},
    map,
    matrix::meta::node::LBNode,
    model::model::Model,
};

/**
 * Internal structure, encapsulated
 * Builds the neural model with default values
 */
pub struct ModelBuilder {
    inputs: IndexMap<LBRef, String>,
    outputs: IndexMap<LBRef, String>,
    graph: IndexMap<LBRef, BuilderNode>,
}

/**
 * Exposed builder, models can be easily reverted to a builder via the reference
 */
#[derive(Clone)]
pub struct ModelBuilderRc {
    r: Rc<ModelBuilder>,
}

impl Into<ModelBuilderRc> for ModelBuilder {
    fn into(self) -> ModelBuilderRc {
        ModelBuilderRc::wrap(self)
    }
}

impl ModelBuilderRc {
    pub fn wrap(builder: ModelBuilder) -> ModelBuilderRc {
        return ModelBuilderRc {
            r: Rc::new(builder),
        };
    }

    pub fn build(&self) -> Model {
        let mut inputs: IndexMap<String, String> = IndexMap::new();
        let mut outputs: IndexMap<String, String> = IndexMap::new();

        let serialized: IndexMap<String, GraphPropagationNode> = self
            .r
            .graph
            .iter()
            .map(|entry| {
                let name = entry.1.name();

                if let Some(key_value) = self.r.inputs.get_key_value(entry.0) {
                    inputs.insert(name.clone(), key_value.1.clone());
                }

                if let Some(key_value) = self.r.outputs.get_key_value(entry.0) {
                    outputs.insert(name.clone(), key_value.1.clone());
                }

                let instance = entry.0.borrow_ref().create_instance(name.clone());
                let graph_node = match entry.1 {
                    BuilderNode::DeadEnd(_) => match instance {
                        LayerPropagateEnum::SingleInput(b) => GraphPropagationNode::DeadEnd(b),
                        _ => panic!("{} is not a dead end graph node", name),
                    },
                    BuilderNode::SingleParent(s) => match instance {
                        LayerPropagateEnum::SingleInput(b) => {
                            GraphPropagationNode::SingleInput(s.parent_name.clone(), b)
                        }
                        _ => panic!("{} is not a dead end graph node", name),
                    },
                    BuilderNode::MultipleParent(s) => match instance {
                        LayerPropagateEnum::MultipleInput(b) => {
                            GraphPropagationNode::MultipleInput(s.parent_names.clone(), b)
                        }
                        _ => panic!("{} is not a dead end graph node", name),
                    },
                };
                (name, graph_node)
            })
            .collect::<IndexMap<_, _>>();

        return Model {
            model_builder: self.clone(),
            input_layer_to_data_name: inputs,
            output_layer_to_data_name: outputs,
            sequential_prop: serialized,
        };
    }
}

impl ModelBuilder {
    /**
     * Helper constant for skipping definition if key in a single branched input/output model
     */
    const SINGLE_IO: &str = "DEF_IO";

    pub fn from_straight(input: LBRef, output: LBRef) -> ModelBuilderRc {
        return Self::from(
            map!(input => Self::SINGLE_IO.to_owned()),
            map!(output => Self::SINGLE_IO.to_owned()),
        );
    }

    pub fn from_single_o(inputs: IndexMap<LBRef, String>, output: LBRef) -> ModelBuilderRc {
        return Self::from(inputs, map!(output => Self::SINGLE_IO.to_owned()));
    }

    pub fn from_single_i(input: LBRef, outputs: IndexMap<LBRef, String>) -> ModelBuilderRc {
        return Self::from(map!(input => Self::SINGLE_IO.to_owned()), outputs);
    }

    pub fn from(
        inputs: IndexMap<LBRef, String>,
        outputs: IndexMap<LBRef, String>,
    ) -> ModelBuilderRc {
        inputs.iter().for_each(|input_entry| {
            if let LBNode::DeadEnd = input_entry.0.borrow_ref().get_node() {
            } else {
                panic!("Bad input node, should be a DeadEnd node impl")
            }
        });
        let mut graph: IndexMap<LBRef, BuilderNode> = IndexMap::new();

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
        graph: &mut IndexMap<LBRef, BuilderNode>,
        current_layer: &LBRef,
        depth: usize,
    ) -> String {
        if let Some(existing) = graph.get(current_layer) {
            return existing.name();
        }
        match &current_layer.borrow_ref().get_node() {
            LBNode::DeadEnd => {
                let name = format!("{}_{}", current_layer.type_name(), graph.len());
                let builder_node = DeadEndStruct { name: name.clone() };
                graph.insert(current_layer.clone(), BuilderNode::DeadEnd(builder_node));
                return name;
            }
            LBNode::SingleParent(parent) => {
                let parent_name = Self::iterate_nodes(graph, parent, depth + 1);
                let name = format!("{}_{}", current_layer.type_name(), graph.len());
                let builder_node = SingleParentStruct {
                    name: name.clone(),
                    parent_name,
                };
                graph.insert(
                    current_layer.clone(),
                    BuilderNode::SingleParent(builder_node),
                );
                return name;
            }
            LBNode::MultipleParent(parents) => {
                let parent_names: Vec<String> = parents
                    .iter()
                    .map(|p| Self::iterate_nodes(graph, p, depth + 1))
                    .collect();
                let name = format!("{}_{}", current_layer.type_name(), graph.len());
                let builder_node = MultipleParentStruct {
                    name: name.clone(),
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

    pub fn build(&self) -> Model {
        todo!()
    }
}

enum BuilderNode {
    DeadEnd(DeadEndStruct),
    SingleParent(SingleParentStruct),
    MultipleParent(MultipleParentStruct),
}

impl BuilderNode {
    fn name(&self) -> String {
        match self {
            BuilderNode::DeadEnd(s) => s.name.clone(),
            BuilderNode::SingleParent(s) => s.name.clone(),
            BuilderNode::MultipleParent(s) => s.name.clone(),
        }
    }
}

impl Debug for BuilderNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DeadEnd(arg0) => f.debug_struct(&arg0.name).finish(),
            Self::SingleParent(arg0) => f
                .debug_struct(&arg0.name)
                .field("parent", &arg0.parent_name)
                .finish(),
            Self::MultipleParent(arg0) => f
                .debug_struct(&arg0.name)
                .field("parents", &arg0.parent_names)
                .finish(),
        }
    }
}

struct DeadEndStruct {
    name: String,
}

struct SingleParentStruct {
    name: String,
    parent_name: String,
}

struct MultipleParentStruct {
    name: String,
    parent_names: Vec<String>,
}

impl Debug for ModelBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelBuilder")
            .field(
                "inputs",
                &self
                    .inputs
                    .iter()
                    .map(|i| (i.1, self.graph.get(i.0).unwrap().name()) as (&String, String))
                    .collect::<IndexMap<_, _>>(),
            )
            .field(
                "outputs",
                &self
                    .outputs
                    .iter()
                    .map(|i| (self.graph.get(i.0).unwrap().name(), i.1) as (String, &String))
                    .collect::<IndexMap<_, _>>(),
            )
            .field("graph", &self.graph.iter().map(|e| e.1).collect::<Vec<_>>())
            .finish()
    }
}

impl Debug for ModelBuilderRc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.r.fmt(f)
    }
}
