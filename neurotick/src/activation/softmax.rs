use serde::{Deserialize, Serialize};

use crate::{matrix::nmatrix::NDMatrix, utils::{json_wrap::JsonWrap, math::matrix_math::MatrixMath}};

use super::abs::{Activation, ActivationVirtual, ActivationSerialised};

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct SoftMax {}

impl SoftMax {
    pub const NAME: &str = "SoftMax";
}

impl Default for SoftMax {
    fn default() -> Self {
        Self { }
    }
}

impl Activation for SoftMax {
    fn apply(&self, array: &NDMatrix) -> NDMatrix {
        return MatrixMath::softmax_per_row(array);
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

impl ActivationVirtual for SoftMax {
    fn from_json(json: &JsonWrap) -> Box<dyn Activation> {
        Box::new(json.to::<SoftMax>().unwrap())
    }

    fn type_name() -> &'static str {
        return Self::NAME
    }
}