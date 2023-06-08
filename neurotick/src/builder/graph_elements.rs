use serde::{Deserialize, Serialize};
use std::fmt::Debug;

use crate::{
    layer::abs::{LayerMultiInput, LayerSingleInput},
    utils::json_wrap::JsonWrap,
};

#[derive(Clone, Serialize, Deserialize)]
pub enum BuilderNode {
    DeadEnd(DeadEndStruct),
    SingleParent(SingleParentStruct),
    MultipleParent(MultipleParentStruct),
}

impl BuilderNode {
    pub fn layer_name(&self) -> String {
        match self {
            BuilderNode::DeadEnd(s) => s.layer_name.clone(),
            BuilderNode::SingleParent(s) => s.layer_name.clone(),
            BuilderNode::MultipleParent(s) => s.layer_name.clone(),
        }
    }

    pub fn type_name(&self) -> String {
        match self {
            BuilderNode::DeadEnd(s) => s.type_name.clone(),
            BuilderNode::SingleParent(s) => s.type_name.clone(),
            BuilderNode::MultipleParent(s) => s.type_name.clone(),
        }
    }
}

impl Debug for BuilderNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DeadEnd(arg0) => f.debug_struct(&arg0.layer_name).finish(),
            Self::SingleParent(arg0) => f
                .debug_struct(&arg0.layer_name)
                .field("parent", &arg0.parent_name)
                .finish(),
            Self::MultipleParent(arg0) => f
                .debug_struct(&arg0.layer_name)
                .field("parents", &arg0.parent_names)
                .finish(),
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct DeadEndStruct {
    pub layer_name: String,
    pub type_name: String,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct SingleParentStruct {
    pub layer_name: String,
    pub type_name: String,
    pub parent_name: String,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct MultipleParentStruct {
    pub layer_name: String,
    pub type_name: String,
    pub parent_names: Vec<String>,
}

pub enum ModelPropagationNode {
    DeadEnd(Box<dyn LayerSingleInput>),
    SingleInput(String, Box<dyn LayerSingleInput>),
    MultipleInput(Vec<String>, Box<dyn LayerMultiInput>),
}

impl ModelPropagationNode {
    pub fn to_json(&self) -> JsonWrap {
        return match self {
            ModelPropagationNode::DeadEnd(r) => r.to_json(),
            ModelPropagationNode::SingleInput(_, r) => r.to_json(),
            ModelPropagationNode::MultipleInput(_, r) => r.to_json(),
        };
    }
}
