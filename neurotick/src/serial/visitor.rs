pub struct RawStringVisitor;

/**
 * Not a safe implementation of formatting the an escaped json as raw string
 */
pub mod raw_string {
    use serde::{ser::SerializeMap, Deserialize, Deserializer, Serializer};
    use serde_json::{Map, Value};

    pub fn serialize<S>(data: &String, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match serde_json::from_str::<Map<String, Value>>(&data) {
            Ok(r) => {
                let mut mapper = serializer.serialize_map(Some(r.len()))?;

                for entry in r {
                    mapper.serialize_entry(&entry.0, &entry.1)?;
                }

                return mapper.end();
            }
            Err(_) => return serializer.serialize_str(&data),
        };
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<String, D::Error>
    where
        D: Deserializer<'de>,
    {
        let map = Map::deserialize(deserializer).unwrap();

        Ok(serde_json::to_string(&map).unwrap())
    }
}

pub mod raw_value_map {
    use indexmap::IndexMap;
    use serde::{ser::SerializeMap, Deserialize, Deserializer, Serializer};
    use serde_json::{Map, Value};

    pub fn serialize<S>(data: &IndexMap<String, String>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut mapper = serializer.serialize_map(Some(data.len()))?;
        for entry in data {
            mapper.serialize_key(entry.0)?;

            match serde_json::from_str::<Map<String, Value>>(&entry.1) {
                Ok(r) => mapper.serialize_value(&r)?,
                Err(_) => mapper.serialize_value(&entry.1)?,
            }
        }

        mapper.end()
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<IndexMap<String, String>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let map: Map<String, Value> = Deserialize::deserialize(deserializer)?;
        let result = map.into_iter().map(|e| {
            let enclosed_local = e.1.as_object().unwrap();
            (e.0.clone(), serde_json::to_string(&enclosed_local).unwrap())
        }).collect();
        Ok(result)
    }
}
