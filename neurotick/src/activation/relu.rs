use serde::{Deserialize, Serialize};

use crate::{matrix::nmatrix::NDMatrix, utils::json_wrap::JsonWrap};

use super::abs::{Activation, ActivationVirtual, ActivationSerialised};

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct ReLu {
    pub cap: f32,
}

impl ReLu {
    pub const NAME: &str = "ReLu";
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

impl ActivationVirtual for ReLu {
    fn from_json(json: &JsonWrap) -> Box<dyn Activation> {
        Box::new(json.to::<ReLu>().unwrap())
    }

    fn type_name() -> &'static str {
        return Self::NAME
    }
}