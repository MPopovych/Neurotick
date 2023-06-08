use serde::{Deserialize, Serialize};

use crate::{
    activation::{abs::{Activation, ActivationSerialised}, none::NoneAct},
    matrix::{
        meta::{node::LayerType, shape::Shape},
        nmatrix::NDMatrix,
    }, serial::model_reader::ModelReader, utils::json_wrap::JsonWrap, suppliers::suppliers::{Suppliers, GlorothNormalSupplier, Supplier, ZeroSupplier},
};

use super::abs::{LayerRef, Layer, LayerBase, LayerPropagateEnum, LayerSingleInput};


pub struct Direct {
    parent: LayerRef,
    activation: Box<dyn Activation>,
    weight_init: Suppliers,
    bias_init: Suppliers,
}

impl Direct {
    pub const NAME: &str = "Direct";

    pub fn new<'a, F>(uplink: F) -> LayerRef
    where
        F: Fn() -> &'a LayerRef,
    {
        let direct = Self::builder(uplink);
        return LayerRef::pin(direct);
    }

    pub fn builder<'a, F>(uplink: F) -> Direct
    where
        F: Fn() -> &'a LayerRef,
    {
        return Direct {
            parent: uplink().clone(),
            activation: Box::new(NoneAct::default()),
            weight_init: GlorothNormalSupplier::new().into_enum(),
            bias_init: ZeroSupplier::new().into_enum(),
        };
    }

    pub fn with_activation(mut self, activation: impl Activation) -> Direct {
        self.activation = Box::new(activation);
        return self;
    }

    pub fn with_weight_init(mut self, supplier: impl Supplier) -> Direct {
        self.weight_init = supplier.into_enum();
        return self;
    }

    pub fn with_bias_init(mut self, supplier: impl Supplier) -> Direct {
        self.bias_init = supplier.into_enum();
        return self;
    }

    pub fn build(self) -> LayerRef {
        return LayerRef::pin(self);
    }
}

impl Layer for Direct {
    fn type_name(&self) -> &'static str {
        return Self::NAME;
    }

    fn get_shape(&self) -> (Shape, Shape) {
        return (
            self.parent.get_shape().0.clone(),
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

        let weight_m = self.weight_init.supply_matrix(parent_feats, 1);
        let bias_m = self.bias_init.supply_matrix(parent_feats, 1);
        let instance = DirectImpl {
            id: id,
            weight: weight_m,
            bias: bias_m,
            activation: self.activation.act_clone(),
        };
        LayerPropagateEnum::SingleInput(Box::new(instance))
    }
}

pub struct DirectImpl {
    id: String,
    weight: NDMatrix,
    bias: NDMatrix,
    activation: Box<dyn Activation>,
}

impl LayerBase for DirectImpl {
    fn init(&mut self) {}

    fn create_from_ser(json: &JsonWrap, model_reader: &ModelReader) -> LayerPropagateEnum {
        let deserialized: DirectSerialization = json.to().unwrap();
        let activation_ser = &deserialized.activation;
        let impl_ref = DirectImpl {
            id: deserialized.id,
            weight: deserialized.weight,
            bias: deserialized.bias,
            activation: model_reader.get_activation_di().create(&activation_ser.name, &activation_ser.json, model_reader)
        };

        return LayerPropagateEnum::SingleInput(
            Box::new(impl_ref)
        )
    }

    fn to_json(&self) -> JsonWrap {
        let serial = DirectSerialization {
            id: self.id.clone(),
            weight: self.weight.clone(),
            bias: self.bias.clone(),
            activation: self.activation.as_serialized()
        };
        return JsonWrap::from(serial).unwrap()
    }
}

impl LayerSingleInput for DirectImpl {
    fn propagate(&self, input: &NDMatrix) -> NDMatrix {
        let weight_hadamard = NDMatrix::hadamard_row_wise(input, &self.weight);
        let with_bias = NDMatrix::add(&weight_hadamard, &self.bias);
        return self.activation.apply(&with_bias);
    }
}

/**
 * Serialization
 */

#[derive(Serialize, Deserialize, Debug)]
struct DirectSerialization {
    id: String,
    weight: NDMatrix,
    bias: NDMatrix,
    activation: ActivationSerialised
}
