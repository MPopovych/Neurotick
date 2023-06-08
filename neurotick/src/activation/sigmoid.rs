use serde::{Deserialize, Serialize};

use crate::{
    matrix::nmatrix::NDMatrix,
    utils::{json_wrap::JsonWrap, math::fast_math::FMath},
};

use super::abs::{Activation, ActivationSerialised, ActivationVirtual};

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct Sigmoid;

impl Sigmoid {
    pub const NAME: &str = "Sigmoid";
}

impl Default for Sigmoid {
    fn default() -> Self {
        Self {}
    }
}

impl Activation for Sigmoid {
    fn apply(&self, array: &NDMatrix) -> NDMatrix {
        let data = array.values.map(|f| FMath::fast_sigmoid(*f));
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

impl ActivationVirtual for Sigmoid {
    fn from_json(_json: &JsonWrap) -> Box<dyn Activation> {
        Box::new(Sigmoid::default())
    }

    fn type_name() -> &'static str {
        return Self::NAME;
    }
}
