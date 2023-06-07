use serde::{Deserialize, Serialize};

use crate::{
    activation::{abs::{Activation, ActivationSerialised}, none::NoneAct},
    matrix::{
        meta::{node::LayerType, shape::Shape},
        nmatrix::NDMatrix,
    }, serial::model_reader::ModelReader, utils::json_wrap::JsonWrap, suppliers::suppliers::{Suppliers, GlorothNormalSupplier, Supplier, ZeroSupplier},
};

use super::abs::{LayerRef, Layer, LayerBase, LayerPropagateEnum, LayerSingleInput};


pub struct Dense {
    features: usize,
    parent: LayerRef,
    activation: Box<dyn Activation>,
    weight_init: Suppliers,
    bias_init: Suppliers,
}

impl Dense {
    pub const NAME: &str = "Dense";

    pub fn new<'a, F>(features: usize, uplink: F) -> LayerRef
    where
        F: Fn() -> &'a LayerRef,
    {
        let dense = Self::builder(features, uplink);
        return LayerRef::pin(dense);
    }

    pub fn builder<'a, F>(features: usize, uplink: F) -> Dense
    where
        F: Fn() -> &'a LayerRef,
    {
        return Dense {
            features,
            parent: uplink().clone(),
            activation: Box::new(NoneAct::default()),
            weight_init: GlorothNormalSupplier::new().into_enum(),
            bias_init: ZeroSupplier::new().into_enum(),
        };
    }

    pub fn with_activation(mut self, activation: impl Activation) -> Dense {
        self.activation = Box::new(activation);
        return self;
    }

    pub fn with_weight_init(mut self, supplier: impl Supplier) -> Dense {
        self.weight_init = supplier.into_enum();
        return self;
    }

    pub fn with_bias_init(mut self, supplier: impl Supplier) -> Dense {
        self.bias_init = supplier.into_enum();
        return self;
    }

    pub fn build(self) -> LayerRef {
        return LayerRef::pin(self);
    }
}

impl Layer for Dense {
    fn type_name(&self) -> &'static str {
        return Self::NAME;
    }

    fn get_shape(&self) -> (Shape, Shape) {
        return (
            Shape::Const(self.features),
            self.parent.get_shape().1.clone(),
        );
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

        let weight_m = self.weight_init.supply_matrix(self.features, parent_feats);
        let bias_m = self.bias_init.supply_matrix(self.features, 1);
        let instance = DenseImpl {
            id: id,
            features: self.features,
            weight: weight_m,
            bias: bias_m,
            activation: self.activation.act_clone(),
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

    fn create_from_ser(json: &JsonWrap, model_reader: &ModelReader) -> LayerPropagateEnum {
        let deserialized: DenseSerialization = json.to().unwrap();
        let activation_ser = &deserialized.activation;
        let impl_ref = DenseImpl {
            id: deserialized.id,
            features: deserialized.features,
            weight: deserialized.weight,
            bias: deserialized.bias,
            activation: model_reader.get_activation_di().create(&activation_ser.name, &activation_ser.json, model_reader)
        };

        return LayerPropagateEnum::SingleInput(
            Box::new(impl_ref)
        )
    }

    fn to_json(&self) -> JsonWrap {
        let serial = DenseSerialization {
            id: self.id.clone(),
            features: self.features.clone(),
            weight: self.weight.clone(),
            bias: self.bias.clone(),
            activation: self.activation.as_serialized()
        };
        return JsonWrap::from(serial).unwrap()
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
