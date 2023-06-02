use serde::{Deserialize, Serialize};

use crate::matrix::nmatrix::NDMatrix;

use super::abs::{Activation, ActivationHandler, NamedActivation};

const LERELU_NAME: &str = "LeakyReLu";

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct LeakyReLu {
    pub beta: f32,
}

impl Default for LeakyReLu {
    fn default() -> Self {
        Self { beta: 0.03 }
    }
}

impl Activation for LeakyReLu {
    fn apply(&self, array: &NDMatrix) -> NDMatrix {
        let data = array.values.map(|f| (f * self.beta).min(0.).max(*f));
        return NDMatrix::with(array.width, array.height, data);
    }

    fn as_json(&self) -> String {
        serde_json::to_string(&self).unwrap()
    }
}

impl NamedActivation for LeakyReLu {
    fn name(&self) -> String {
        return LERELU_NAME.to_owned();
    }
}

pub struct LeReLuHandler;

impl NamedActivation for LeReLuHandler {
    fn name(&self) -> String {
        return LERELU_NAME.to_owned();
    }
}

impl ActivationHandler for LeReLuHandler {
    fn read(&self, json: &String) -> Box<dyn Activation> {
        return Box::new(serde_json::from_str::<LeakyReLu>(&json).unwrap());
    }
}
