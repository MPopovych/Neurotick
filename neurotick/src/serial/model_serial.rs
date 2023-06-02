use indexmap::IndexMap;
use serde::{Serialize, Deserialize};

use crate::builder::builder::BuilderNode;

use super::visitor::raw_value_map;

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct ModelGraph {
    #[serde(flatten)]
    pub graph: IndexMap<String, BuilderNode>
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct ModelIO {
    pub inputs: IndexMap<String, String>,
    pub outputs: IndexMap<String, String>,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct ModelMeta {
    #[serde(with = "raw_value_map")]
    pub meta: IndexMap<String, String>
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct ModelSerialized {
    pub io: ModelIO,
    pub graph: ModelGraph,
    #[serde(flatten)]
    pub meta: ModelMeta
}

impl ModelSerialized {
    pub fn to_json(&self) -> String {
        return serde_json::to_string(self).unwrap()
    }
    pub fn to_json_pretty(&self) -> String {
        return serde_json::to_string_pretty(self).unwrap()
    }
}
