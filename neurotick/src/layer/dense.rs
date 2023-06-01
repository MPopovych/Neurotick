use std::rc::Rc;

use crate::matrix::{
    meta::{node::LBNode, shape::Shape},
    nmatrix::NDMatrix,
};

use super::abs::{
    LBRef, Layer, LayerPropagateBase, LayerPropagateEnum, LayerSingleInput, TypedLayer,
};

#[derive(Clone)]
pub struct Dense {
    features: usize,
    parent: LBRef,
}

impl Dense {
    pub fn new<'a, F>(features: usize, uplink: F) -> LBRef
    where
        F: Fn() -> &'a LBRef,
    {
        let dense = Dense {
            features,
            parent: uplink().clone(),
        };
        let rc = Rc::new(dense);
        return LBRef { reference: rc };
    }
}

impl TypedLayer for Dense {
    fn type_name(&self) -> String {
        return "Dense".to_string();
    }
}

impl Layer for Dense {
    fn get_shape(&self) -> (Shape, Shape) {
        return (
            Shape::Const(self.features),
            self.parent.get_shape().1.clone(),
        );
    }

    fn get_node(&self) -> LBNode {
        return LBNode::SingleParent(self.parent.clone());
    }

    fn create_instance(&self, id: String) -> LayerPropagateEnum {
        let parent_feats = self.parent.get_shape().0.unwrap_to_conts();
        if parent_feats <= 0 {
            panic!("Zero or negative features in parent is not allowed, by: {}", id);
        }
        let instance = DenseImpl {
            _id: id,
            _dense: self.clone(),
            weight: NDMatrix::constant(self.features, parent_feats, 1.0 / parent_feats as f32),
            bias: NDMatrix::constant(self.features, 1, 0.0),
        };
        LayerPropagateEnum::SingleInput(Box::new(instance))
    }
}

pub struct DenseImpl {
    _id: String,
    _dense: Dense,
    weight: NDMatrix,
    bias: NDMatrix,
}

impl LayerPropagateBase for DenseImpl {
    fn init(&self) {}
}

impl LayerSingleInput for DenseImpl {
    fn propagate(&self, input: &NDMatrix) -> NDMatrix {
        let weighted = &(input * &self.weight);
        return weighted + &self.bias;
    }
}
