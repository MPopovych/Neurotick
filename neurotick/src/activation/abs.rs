use std::{fmt::Debug, any::Any};

use serde::{Deserialize, Serialize};

use crate::{matrix::nmatrix::NDMatrix, utils::{json_wrap::JsonWrap, as_any::AsAny}};

pub trait Activation: NamedActivation + Debug + AsAny {
    fn apply(&self, array: &NDMatrix) -> NDMatrix;
    fn as_serialized(&self) -> ActivationSerialised;
}

impl<T: 'static + Activation> AsAny for T {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

pub trait ActivationSerialize {
    fn as_serialized(&self) -> ActivationSerialised;
}

pub trait NamedActivation {
    fn name(&self) -> &'static str;
}

pub trait ActivationVirtual {
    fn from_json(json: &JsonWrap) -> Box<dyn Activation>;
    fn name() -> &'static str;
}


#[derive(Debug, Serialize, Deserialize)]
pub struct ActivationSerialised {
    pub name: String,
    pub json: JsonWrap,
}
