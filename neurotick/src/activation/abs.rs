use std::{fmt::Debug};

use crate::matrix::nmatrix::NDMatrix;

pub trait Activation: NamedActivation + Debug { 
    fn apply(&self, array: &NDMatrix) -> NDMatrix;
    fn as_json(&self) -> String;
}
pub trait ActivationHandler: NamedActivation { 
    fn read(&self, json: &String) -> Box<dyn Activation>;
}

pub trait NamedActivation {
    fn name(&self) -> String;
}

#[derive(Debug)]
pub struct ActivationSerialised {
    pub name: String, 
    pub json: String
}
