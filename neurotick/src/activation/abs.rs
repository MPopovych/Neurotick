use std::{any::Any, fmt::Debug};

use serde::{Deserialize, Serialize};

use crate::{
    matrix::nmatrix::NDMatrix,
    utils::{as_any::AsAny, json_wrap::JsonWrap},
};

pub trait Activation: Debug + AsAny {
    fn apply(&self, array: &NDMatrix) -> NDMatrix;
    fn as_serialized(&self) -> ActivationSerialised;
    fn act_clone(&self) -> Box<dyn Activation>;
}

impl<T: 'static + Activation> AsAny for T {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

pub trait ActivationSerialize {
    fn as_serialized(&self) -> ActivationSerialised;
}

pub trait ActivationVirtual {
    fn from_json(json: &JsonWrap) -> Box<dyn Activation>;
    fn type_name() -> &'static str;
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ActivationSerialised {
    pub name: String,
    pub json: JsonWrap,
}
