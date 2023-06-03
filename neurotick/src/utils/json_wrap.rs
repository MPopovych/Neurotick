use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::{Error, Value};

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct JsonWrap {
    #[serde(flatten)]
    value: Value,
}

impl JsonWrap {
    pub fn from<T: Serialize>(model: T) -> Result<JsonWrap, Error> {
        return Ok(JsonWrap {
            value: serde_json::to_value(model)?,
        });
    }

    pub fn to<T>(&self) -> Result<T, Error>
    where
        T: DeserializeOwned,
    {
        return serde_json::from_value(self.value.clone());
    }

    pub fn to_string(&self) -> Result<String, Error> {
        return serde_json::to_string(&self.value)
    }
}
