use std::hash::Hash;
use std::hash::Hasher;
use std::rc::Rc;

use crate::matrix::{
    meta::{node::LBNode, shape::Shape},
    nmatrix::NDMatrix,
};

#[derive(Clone)]
pub struct LBRef {
    pub reference: Rc<dyn Layer>,
}
impl LBRef {
    pub fn get_shape(&self) -> (Shape, Shape) {
        return self.reference.get_shape();
    }

    pub fn type_name(&self) -> String {
        return self.reference.as_ref().type_name();
    }
}

impl PartialEq for LBRef {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self.reference.as_ref(), other.reference.as_ref())
    }
}
impl Eq for LBRef {}

impl Hash for LBRef {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::ptr::hash(self.reference.as_ref(), state)
    }
}

pub trait TypedLayer {
    fn type_name(&self) -> String;
}

pub trait Layer: TypedLayer {
    fn get_shape(&self) -> (Shape, Shape);
    fn get_node(&self) -> LBNode;
    fn create_instance(&self, name: String) -> LayerPropagateEnum;
}

// Implementation for layers, forward propagation

/**
 * Init
 */
pub trait LayerPropagateBase {
    fn init(&self);
}

pub trait LayerSingleInput: LayerPropagateBase {
    fn propagate(&self, input: &NDMatrix) -> NDMatrix;
}

pub trait LayerMultiInput: LayerPropagateBase {
    fn propagate_multi(&self, inputs: &Vec<&NDMatrix>) -> NDMatrix;
}

pub enum LayerPropagateEnum {
    SingleInput(Box<dyn LayerSingleInput>),
    MultipleInput(Box<dyn LayerMultiInput>),
}

pub enum GraphPropagationNode {
    DeadEnd(Box<dyn LayerSingleInput>),
    SingleInput(String, Box<dyn LayerSingleInput>),
    MultipleInput(Vec<String>, Box<dyn LayerMultiInput>),
}
