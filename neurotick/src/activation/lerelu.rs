use serde::{Deserialize, Serialize};

use crate::matrix::nmatrix::NDMatrix;

use super::abs::{
    Activation, ActivationSerialised, ForwardActivation, NamedActivation, Parser, Writer,
};

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct LeakyReLu {
    pub beta: f32,
}

impl Default for LeakyReLu {
    fn default() -> Self {
        Self { beta: 0.03 }
    }
}

impl ForwardActivation for LeakyReLu {
    fn apply(&self, array: &NDMatrix) -> NDMatrix {
        let data = array.values.map(|f| (f * self.beta).min(0.).max(*f));
        return NDMatrix::with(array.width, array.height, data);
    }
}

impl NamedActivation for LeakyReLu {
    fn name(&self) -> String {
        return "LeReLu".to_owned();
    }
}

impl Parser for LeakyReLu {
    fn read_json(&self, json: &String) -> Box<dyn Activation> {
        let lerelu = serde_json::from_str::<LeakyReLu>(&json).unwrap();
        return Box::new(lerelu); // create new
    }
}
impl Writer for LeakyReLu {
    fn write_json(&self, act: &Box<dyn Activation>) -> ActivationSerialised {
        let lerelu = act.as_any().downcast_ref::<LeakyReLu>().unwrap();
        return ActivationSerialised {
            name: self.name(),
            json: serde_json::to_string(lerelu).unwrap(),
        };
    }
}

impl Activation for LeakyReLu {}
