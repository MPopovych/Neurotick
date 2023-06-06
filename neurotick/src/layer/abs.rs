use std::hash::Hash;
use std::hash::Hasher;
use std::rc::Rc;

use crate::matrix::{
    meta::{node::LayerType, shape::Shape},
    nmatrix::NDMatrix,
};
use crate::serial::model_reader::ModelReader;
use crate::utils::json_wrap::JsonWrap;

/**
 * Struct for building out a graph of layers. Pre-instancing
 * Serialization is done on layer instances
 */
#[derive(Clone)]
pub struct LayerRef {
    reference: Rc<dyn Layer>,
}
impl LayerRef {
    pub fn pin<T: Layer + Sized + 'static>(layer: T) -> LayerRef {
        return LayerRef {
            reference: Rc::new(layer),
        };
    }

    pub fn get_shape(&self) -> (Shape, Shape) {
        return self.reference.get_shape();
    }

    pub fn type_name(&self) -> &'static str {
        return self.reference.as_ref().type_name();
    }

    pub fn borrow_ref(&self) -> &dyn Layer {
        return self.reference.as_ref();
    }
}


pub trait Layer {
    /**
     * Represents the identity type of the layer, should be unique other-wise lead to a panic
     */
    fn type_name(&self) -> &'static str;
    
    fn get_shape(&self) -> (Shape, Shape);
    fn get_node(&self) -> LayerType;
    fn create_instance(&self, name: String) -> LayerPropagateEnum;
}

// Implementation for layers, forward propagation

/**
 * Init
 */
pub trait LayerBase {
    fn init(&mut self);
    fn to_json(&self) -> JsonWrap;

    fn create_from_ser(json: &JsonWrap, model_reader: &ModelReader) -> LayerPropagateEnum
    where
        Self: Sized;
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

/**
 * Helpers for navigating the graph on a pointer basis
 */
impl PartialEq for LayerRef {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self.reference.as_ref(), other.reference.as_ref())
    }
}
impl Eq for LayerRef {}

impl Hash for LayerRef {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::ptr::hash(self.reference.as_ref(), state)
    }
}
