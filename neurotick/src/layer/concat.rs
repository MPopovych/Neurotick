use serde::{Deserialize, Serialize};

use crate::{
    matrix::{
        meta::{node::LayerType, shape::Shape},
        nmatrix::NDMatrix,
    },
    serial::model_reader::ModelReader,
    utils::{json_wrap::JsonWrap, extensions::Distinct},
};

use super::abs::{
    Layer, LayerBase, LayerMultiInput, LayerPropagateEnum, LayerRef,
};

#[derive(Clone)]
pub struct Concat {
    parents: Vec<LayerRef>,
    features: Shape,
    size: Shape,
}

impl Concat {
    pub const NAME: &str = "Concat";

    pub fn new<'a, F>(uplinks: F) -> LayerRef
    where
        F: Fn() -> Vec<&'a LayerRef>,
    {
        let uplink_vec = uplinks();
        let features = uplink_vec
            .iter()
            .map(|l| l.get_shape().0.unwrap_to_conts())
            .sum();

        let all_const = uplink_vec.iter().all(|l| l.get_shape().1.is_const());
        let size = if all_const {
            let set = uplink_vec
                .iter()
                .distinct_vec(|l| l.get_shape().1.unwrap_to_conts());
            if set.len() != 1 {
                panic!("Different sizes")
            }
            Shape::Const(set[0])
        } else {
            Shape::Variable
        };

        let concat = Concat {
            parents: uplink_vec.iter().map(|u| (*u).clone()).collect(),
            features: Shape::Const(features),
            size: size,
        };
        return LayerRef::pin(concat);
    }
}

impl Layer for Concat {
    fn type_name(&self) -> &'static str {
        return Self::NAME;
    }

    fn get_shape(&self) -> (Shape, Shape) {
        return (self.features.clone(), self.size.clone());
    }

    fn get_node(&self) -> LayerType {
        return LayerType::MultipleParent(self.parents.clone());
    }

    fn create_instance(&self, name: String) -> LayerPropagateEnum {
        let instance = ConcatImpl {
            id: name,
            features: self.features.clone(),
            size: self.size.clone(),
        };
        LayerPropagateEnum::MultipleInput(Box::new(instance))
    }
}

pub struct ConcatImpl {
    id: String,
    features: Shape,
    size: Shape,
}

impl LayerBase for ConcatImpl {
    fn init(&mut self) {}

    fn create_from_ser(json: &JsonWrap, _model_reader: &ModelReader) -> LayerPropagateEnum {
        let deserialized: ConcatSerialization = json.to().unwrap();
        let impl_ref = ConcatImpl {
            id: deserialized.id,
            features: deserialized.features,
            size: deserialized.size,
        };
        return LayerPropagateEnum::MultipleInput(Box::new(impl_ref));
    }

    fn to_json(&self) -> JsonWrap {
        let serial = ConcatSerialization {
            id: self.id.clone(),
            features: self.features.clone(),
            size: self.size.clone(),
        };
        return JsonWrap::from(serial).unwrap();
    }
}

impl LayerMultiInput for ConcatImpl {
    fn propagate_multi(&self, inputs: &Vec<&NDMatrix>) -> NDMatrix {
        NDMatrix::concat_horizontal(&inputs[..])
    }
}

/**
 * Serialization
 */

#[derive(Serialize, Deserialize, Debug)]
struct ConcatSerialization {
    id: String,
    features: Shape,
    size: Shape,
}
