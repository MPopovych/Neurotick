use serde::{Deserialize, Serialize};

use crate::{
    activation::{abs::{Activation, ActivationSerialised}, relu::ReLu},
    matrix::{
        meta::{node::LBNode, shape::Shape},
        nmatrix::NDMatrix,
    }, serial::model_reader::ModelReader,
};

use super::abs::{LBRef, Layer, LayerBase, LayerPropagateEnum, LayerSingleInput, TypedLayer};

#[derive(Clone)]
pub struct Dense {
    features: usize,
    parent: LBRef,
}

impl Dense {
    pub const NAME: &str = "Dense";

    pub fn new<'a, F>(features: usize, uplink: F) -> LBRef
    where
        F: Fn() -> &'a LBRef,
    {
        let dense = Dense {
            features,
            parent: uplink().clone(),
        };
        return LBRef::pin(dense);
    }
}

impl TypedLayer for Dense {
    fn type_name(&self) -> String {
        return Self::NAME.to_string();
    }
}

impl Layer for Dense {
    fn get_shape(&self) -> (Shape, Shape) {
        return (
            Shape::Const(self.features),
            self.parent.get_shape().1.clone(),
        );
    }

    fn get_node(&self) -> LBNode {
        return LBNode::SingleParent(self.parent.clone());
    }

    fn create_instance(&self, id: String) -> LayerPropagateEnum {
        let parent_feats = self.parent.get_shape().0.unwrap_to_conts();
        if parent_feats <= 0 {
            panic!(
                "Zero or negative features in parent is not allowed, by: {}",
                id
            );
        }
        let instance = DenseImpl {
            id: id,
            features: self.features,
            weight: NDMatrix::constant(self.features, parent_feats, 1.0 / parent_feats as f32),
            bias: NDMatrix::constant(self.features, 1, 0.0),
            activation: Box::new(ReLu::default()),
        };
        LayerPropagateEnum::SingleInput(Box::new(instance))
    }
}

pub struct DenseImpl {
    id: String,
    features: usize,
    weight: NDMatrix,
    bias: NDMatrix,
    activation: Box<dyn Activation>,
}

impl LayerBase for DenseImpl {
    fn init(&mut self) {}

    fn create_from_ser(json: String, model_reader: ModelReader) -> Self {
        let deserialized: DenseSerialization = serde_json::from_str(&json).unwrap();
        return DenseImpl {
            id: deserialized.id,
            features: deserialized.features,
            weight: deserialized.weight,
            bias: deserialized.bias,
            activation: model_reader.get_activation_di().deserialize(&deserialized.activation)
        };
    }

    fn to_json(&self) -> String {
        serde_json::to_string(&DenseSerialization {
            id: self.id.clone(),
            features: self.features.clone(),
            weight: self.weight.clone(),
            bias: self.bias.clone(),
            activation: self.activation.as_serialized()
        })
        .unwrap()
    }
}

impl LayerSingleInput for DenseImpl {
    fn propagate(&self, input: &NDMatrix) -> NDMatrix {
        let weighted = &(input * &self.weight);
        let with_bias = weighted + &self.bias;
        return self.activation.apply(&with_bias);
    }
}

/**
 * Serialization
 */

#[derive(Serialize, Deserialize, Debug)]
struct DenseSerialization {
    id: String,
    features: usize,
    weight: NDMatrix,
    bias: NDMatrix,
    activation: ActivationSerialised
}
