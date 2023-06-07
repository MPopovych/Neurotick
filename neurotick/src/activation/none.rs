use serde::{Deserialize, Serialize};

use crate::{matrix::nmatrix::NDMatrix, utils::json_wrap::JsonWrap};

use super::abs::{Activation, ActivationVirtual, ActivationSerialised};

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct NoneAct;

impl NoneAct {
    pub const NAME: &str = "NoneAct";
}

impl Default for NoneAct {
    fn default() -> Self {
        Self {}
    }
}

impl Activation for NoneAct {
    fn apply(&self, array: &NDMatrix) -> NDMatrix {
        return array.clone()
    }
    fn as_serialized(&self) -> ActivationSerialised {
        ActivationSerialised {
            name: Self::NAME.to_string(),
            json: JsonWrap::from(&self).unwrap()
        }
    }
    fn act_clone(&self) -> Box<dyn Activation> {
        return Box::new(self.clone())
    }

}

impl ActivationVirtual for NoneAct {
    fn from_json(_json: &JsonWrap) -> Box<dyn Activation> {
        Box::new(NoneAct::default())
    }

    fn type_name() -> &'static str {
        return Self::NAME
    }
}