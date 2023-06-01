use std::any::Any;

use crate::matrix::nmatrix::NDMatrix;

pub trait Activation: ForwardActivation + NamedActivation + Parser + Writer + AToAny { }

pub trait AToAny: 'static {
    fn as_any(&self) -> &dyn Any;
}

impl<T: 'static> AToAny for T {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

pub trait ForwardActivation {
    fn apply(&self, array: &NDMatrix) -> NDMatrix;
}

pub trait NamedActivation {
    fn name(&self) -> String;
}

#[derive(Debug)]
pub struct ActivationSerialised {
    pub name: String, 
    pub json: String
}

pub trait Parser {
    fn read_json(&self, json: &String) -> Box<dyn Activation> ;
}

pub trait Writer {
    fn write_json(&self, act: &Box<dyn Activation>) -> ActivationSerialised;
}