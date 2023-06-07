use serde::{Serialize, Deserialize};

use crate::{matrix::{
    meta::{node::LayerType, shape::Shape},
    nmatrix::NDMatrix,
}, serial::model_reader::ModelReader, utils::json_wrap::JsonWrap};

use super::abs::{
    LayerRef, Layer, LayerBase, LayerPropagateEnum, LayerSingleInput,
};

pub struct Input {
    features: Shape,
    size: Shape,
}

impl Input {
    pub const NAME: &str = "Input";

    pub fn new(features: Shape, size: Shape) -> LayerRef {
        let input = Input { features, size };
        return LayerRef::pin(input);
    }
}

impl Layer for Input {
    fn type_name(&self) -> &'static str {
        return Self::NAME;
    }

    fn get_shape(&self) -> (Shape, Shape) {
        return (self.features.clone(), self.size.clone());
    }

    fn get_node(&self) -> LayerType {
        return LayerType::DeadEnd;
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

    fn create_from_ser(json: &JsonWrap, _model_reader: &ModelReader) -> LayerPropagateEnum {
        let deserialized: InputSerialization = json.to().unwrap();
        let impl_ref = InputImpl {
            id: deserialized.id,
            features: deserialized.features,
            size: deserialized.size
        };
        return LayerPropagateEnum::SingleInput(
            Box::new(impl_ref)
        );
    }

    fn to_json(&self) -> JsonWrap {
        let serial = InputSerialization {
            id: self.id.clone(),
            features: self.features.clone(),
            size: self.size.clone(),
        };
        return JsonWrap::from(serial).unwrap()
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