use serde::{Deserialize, Serialize};

use crate::{
    matrix::{
        meta::{node::LayerType, shape::Shape},
        nmatrix::NDMatrix,
    },
    serial::model_reader::ModelReader,
    utils::json_wrap::JsonWrap,
};

use super::abs::{Layer, LayerBase, LayerPropagateEnum, LayerRef, LayerSingleInput};

pub struct Flatten {
    parent: LayerRef,
}

impl Flatten {
    pub const NAME: &str = "Flatten";

    pub fn new<'a, F>(uplink: F) -> LayerRef
    where
        F: Fn() -> &'a LayerRef,
    {
        let flatten = Flatten {
            parent: uplink().clone(),
        };
        return LayerRef::pin(flatten);
    }
}

impl Layer for Flatten {
    fn type_name(&self) -> &'static str {
        return Self::NAME;
    }

    fn get_shape(&self) -> (Shape, Shape) {
        let parent_shape = self.parent.get_shape();
        let feat_count = match parent_shape.0 {
            Shape::Const(x) => match parent_shape.1 {
                Shape::Const(y) => Shape::Const(x * y),
                Shape::Repeat => Shape::Variable,
                Shape::Variable => Shape::Variable,
            },
            Shape::Repeat => Shape::Variable,
            Shape::Variable => Shape::Variable,
        };

        return (feat_count, Shape::Const(1));
    }

    fn get_node(&self) -> LayerType {
        return LayerType::SingleParent(self.parent.clone());
    }

    fn create_instance(&self, id: String) -> LayerPropagateEnum {
        let parent_feats = self.parent.get_shape().0.unwrap_to_conts();
        if parent_feats <= 0 {
            panic!(
                "Zero or negative features in parent is not allowed, by: {}",
                id
            );
        }

        let instance = FlattenImpl { id: id };
        LayerPropagateEnum::SingleInput(Box::new(instance))
    }
}

pub struct FlattenImpl {
    id: String,
}

impl LayerBase for FlattenImpl {
    fn init(&mut self) {}

    fn create_from_ser(json: &JsonWrap, _model_reader: &ModelReader) -> LayerPropagateEnum {
        let deserialized: FlattenSerialization = json.to().unwrap();
        let impl_ref = FlattenImpl {
            id: deserialized.id,
        };
        return LayerPropagateEnum::SingleInput(Box::new(impl_ref));
    }

    fn to_json(&self) -> JsonWrap {
        let serial = FlattenSerialization {
            id: self.id.clone(),
        };
        return JsonWrap::from(serial).unwrap();
    }
}

impl LayerSingleInput for FlattenImpl {
    fn propagate(&self, input: &NDMatrix) -> NDMatrix {
        let features = input.width * input.height;
        return NDMatrix::from_raw_vec(features, 1, input.values.clone().into_raw_vec());
    }
}

/**
 * Serialization
 */

#[derive(Serialize, Deserialize, Debug)]
struct FlattenSerialization {
    id: String,
}
