use std::{fmt::Debug, any::Any};

use serde::{Serialize, Deserialize};

use crate::matrix::nmatrix::NDMatrix;

pub trait Activation: NamedActivation + Debug + ActivationAsAny + ActivationSerialize { 
    fn apply(&self, array: &NDMatrix) -> NDMatrix;
    fn as_json(&self) -> String;
}
pub trait ActivationHandler: NamedActivation { 
    fn read(&self, json: &String) -> Box<dyn Activation>;
}

pub trait ActivationAsAny: 'static {
    fn as_any(&self) -> &dyn Any;
}

impl <T: 'static + Activation> ActivationAsAny for T {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

pub trait ActivationSerialize {
    fn as_serialized(&self) -> ActivationSerialised;
}

impl <T: Activation> ActivationSerialize for T {
    fn as_serialized(&self) -> ActivationSerialised {
        ActivationSerialised {
            name: self.name(),
            json: self.as_json()
        }
    }
}

pub trait NamedActivation {
    fn name(&self) -> String;
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ActivationSerialised {
    pub name: String, 
    pub json: String
}
