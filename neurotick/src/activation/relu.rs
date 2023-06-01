use serde::{Deserialize, Serialize};

use crate::matrix::nmatrix::NDMatrix;

use super::abs::{
    Activation, ActivationSerialised, ForwardActivation, NamedActivation, Parser, Writer,
};

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct ReLu {
    pub mock: u32,
}

impl ForwardActivation for ReLu {
    fn apply(&self, array: &NDMatrix) -> NDMatrix {
        let data = array.values.map(|f| f.max(0.0));
        return NDMatrix::with(array.width, array.height, data);
    }
}

impl NamedActivation for ReLu {
    fn name(&self) -> String {
        return "ReLu".to_owned();
    }
}

impl Parser for ReLu {
    fn read_json(&self, json: &String) -> Box<dyn Activation> {
        let relu = serde_json::from_str::<ReLu>(&json).unwrap();
        return Box::new(relu); // create new
    }
}
impl Writer for ReLu {
    fn write_json(&self, act: &Box<dyn Activation>) -> ActivationSerialised {
        let relu = act.as_any().downcast_ref::<ReLu>().unwrap();
        return ActivationSerialised {
            name: self.name(),
            json: serde_json::to_string(relu).unwrap(),
        };
    }
}
impl Activation for ReLu {}
