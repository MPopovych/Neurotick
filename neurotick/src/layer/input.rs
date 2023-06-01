use std::rc::Rc;

use crate::matrix::{
    meta::{node::LBNode, shape::Shape},
    nmatrix::NDMatrix,
};

use super::abs::{
    LBRef, Layer, LayerPropagateBase, LayerPropagateEnum, LayerSingleInput, TypedLayer,
};

#[derive(Clone)]
pub struct Input {
    features: Shape,
    size: Shape,
}

impl Input {
    pub fn new(features: Shape, size: Shape) -> LBRef {
        let input = Input { features, size };
        let rc = Rc::new(input);
        return LBRef { reference: rc };
    }
}

impl TypedLayer for Input {
    fn type_name(&self) -> String {
        return "Input".to_string();
    }
}

impl Layer for Input {
    fn get_shape(&self) -> (Shape, Shape) {
        return (self.features.clone(), self.size.clone());
    }

    fn get_node(&self) -> LBNode {
        return LBNode::DeadEnd;
    }

    fn create_instance(&self, name: String) -> LayerPropagateEnum {
        let instance = InputImpl {
            _id: name,
            _input: self.clone(),
        };
        LayerPropagateEnum::SingleInput(Box::new(instance))
    }
}

pub struct InputImpl {
    _id: String,
    _input: Input,
}

impl LayerPropagateBase for InputImpl {
    fn init(&self) {}
}

impl LayerSingleInput for InputImpl {
    fn propagate(&self, input: &NDMatrix) -> NDMatrix {
        return input.clone();
    }
}
