use std::hash::Hash;
use std::hash::Hasher;
use std::rc::Rc;

use crate::matrix::{
    meta::{node::LBNode, shape::Shape},
    nmatrix::NDMatrix,
};
use crate::serial::model_reader::ModelReader;

/**
 * Struct for building out a graph of layers. Pre-instancing
 * Serialization is done on layer instances
 */
#[derive(Clone)]
pub struct LBRef {
    reference: Rc<dyn Layer>,
}
impl LBRef {
    pub fn pin<T: Layer + Sized + 'static>(layer: T) -> LBRef {
        return LBRef {
            reference: Rc::new(layer),
        };
    }

    pub fn get_shape(&self) -> (Shape, Shape) {
        return self.reference.get_shape();
    }

    pub fn type_name(&self) -> String {
        return self.reference.as_ref().type_name();
    }

    pub fn borrow_ref(&self) -> &dyn Layer {
        return self.reference.as_ref();
    }
}
/**
 * Helpers for navigating the graph on a pointer basis
 */
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

/**
 * Represents the identity type of the layer, should be unique other-wise lead to a panic
 */
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
pub trait LayerBase {
    fn init(&mut self);
    fn create_from_ser(json: String, model_reader: ModelReader) -> Self
    where
        Self: Sized;
    fn to_json(&self) -> String;
}

pub trait LayerSingleInput: LayerBase {
    fn propagate(&self, input: &NDMatrix) -> NDMatrix;
}

pub trait LayerMultiInput: LayerBase {
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

impl GraphPropagationNode {
    pub fn to_json(&self) -> String {
        return match self {
            GraphPropagationNode::DeadEnd(r) => r.to_json(),
            GraphPropagationNode::SingleInput(_, r) => r.to_json(),
            GraphPropagationNode::MultipleInput(_, r) => r.to_json(),
        };
    }
}
