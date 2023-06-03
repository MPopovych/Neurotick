use serde::{Serialize, Deserialize};

use crate::{matrix::{
    meta::{node::LBNode, shape::Shape},
    nmatrix::NDMatrix,
}, serial::model_reader::ModelReader};

use super::abs::{
    LBRef, Layer, LayerBase, LayerPropagateEnum, LayerSingleInput, TypedLayer,
};

#[derive(Clone)]
pub struct Input {
    features: Shape,
    size: Shape,
}

impl Input {
    pub const NAME: &str = "Input";

    pub fn new(features: Shape, size: Shape) -> LBRef {
        let input = Input { features, size };
        return LBRef::pin(input);
    }
}

impl TypedLayer for Input {
    fn type_name(&self) -> String {
        return Self::NAME.to_string();
    }
}

impl Layer for Input {
    fn get_shape(&self) -> (Shape, Shape) {
        return (self.features.clone(), self.size.clone());
    }

    fn get_node(&self) -> LBNode {
        return LBNode::DeadEnd;
    }

    fn create_instance(&self, name: String) -> LayerPropagateEnum {
        let instance = InputImpl {
            id: name,
            features: self.features.clone(),
            size: self.size.clone(),
        };
        LayerPropagateEnum::SingleInput(Box::new(instance))
    }
}

pub struct InputImpl {
    id: String,
    features: Shape,
    size: Shape,
}

impl LayerBase for InputImpl {
    fn init(&mut self) {}

    fn create_from_ser(json: &String, _model_reader: &ModelReader) -> Self where Self: Sized {
        let deserialized: InputSerialization = serde_json::from_str(&json).unwrap();
        return InputImpl {
            id: deserialized.id,
            features: deserialized.features,
            size: deserialized.size
        }
    }

    fn to_json(&self) -> String {
        serde_json::to_string(&InputSerialization {
            id: self.id.clone(),
            features: self.features.clone(),
            size: self.size.clone(),
        }).unwrap()
    }
}

impl LayerSingleInput for InputImpl {
    fn propagate(&self, input: &NDMatrix) -> NDMatrix {
        return input.clone();
    }
}

/**
 * Serialization
 */

#[derive(Serialize, Deserialize, Debug)]
struct InputSerialization {
    id: String,
    features: Shape,
    size: Shape,
}