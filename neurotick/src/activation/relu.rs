use serde::{Deserialize, Serialize};

use crate::matrix::nmatrix::NDMatrix;

use super::abs::{Activation, ActivationHandler, NamedActivation};

const RELU_NAME: &str = "ReLu";

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct ReLu {
    pub cap: f32,
}

impl Default for ReLu {
    fn default() -> Self {
        Self { cap: 10.0 }
    }
}

impl Activation for ReLu {
    fn apply(&self, array: &NDMatrix) -> NDMatrix {
        let data = array.values.map(|f| f.max(0.0).min(self.cap));
        return NDMatrix::with(array.width, array.height, data);
    }

    fn as_json(&self) -> String {
        serde_json::to_string(&self).unwrap()
    }
}

impl NamedActivation for ReLu {
    fn name(&self) -> String {
        return RELU_NAME.to_owned();
    }
}

pub struct ReLuHandler;

impl NamedActivation for ReLuHandler {
    fn name(&self) -> String {
        return RELU_NAME.to_owned();
    }
}

impl ActivationHandler for ReLuHandler {
    fn read(&self, json: &String) -> Box<dyn Activation> {
        return Box::new(serde_json::from_str::<ReLu>(&json).unwrap());
    }
}
