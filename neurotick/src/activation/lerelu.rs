use serde::{Deserialize, Serialize};

use crate::{matrix::nmatrix::NDMatrix, utils::json_wrap::JsonWrap};

use super::abs::{Activation, ActivationSerialised, ActivationVirtual};

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct LeakyReLu {
    pub beta: f32,
}

impl Default for LeakyReLu {
    fn default() -> Self {
        Self { beta: 0.03 }
    }
}

impl LeakyReLu {
    pub const NAME: &str = "LeakyReLu";
}

impl Activation for LeakyReLu {
    fn apply(&self, array: &NDMatrix) -> NDMatrix {
        let data = array.values.map(|f| (f * self.beta).min(0.).max(*f));
        return NDMatrix::with(array.width, array.height, data);
    }

    fn as_serialized(&self) -> ActivationSerialised {
        ActivationSerialised {
            name: Self::NAME.to_string(),
            json: JsonWrap::from(&self).unwrap(),
        }
    }

    fn act_clone(&self) -> Box<dyn Activation> {
        return Box::new(self.clone());
    }
}

impl ActivationVirtual for LeakyReLu {
    fn from_json(json: &JsonWrap) -> Box<dyn Activation> {
        Box::new(json.to::<LeakyReLu>().unwrap())
    }

    fn type_name() -> &'static str {
        return Self::NAME;
    }
}
