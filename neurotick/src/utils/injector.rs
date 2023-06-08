use std::collections::HashMap;

pub struct GenericInjector<T: ?Sized, J, C> {
    map: HashMap<String, Box<dyn Fn(&J, &C) -> Box<T>>>,
}

impl<T: ?Sized, J, C> GenericInjector<T, J, C> {
    pub fn new() -> GenericInjector<T, J, C> {
        return GenericInjector {
            map: HashMap::new(),
        };
    }

    pub fn register<F: 'static>(&mut self, name: &str, call: F)
    where
        F: Fn(&J, &C) -> Box<T>,
    {
        self.map.insert(name.to_string(), Box::new(call));
    }

    pub fn create(&self, name: &str, json: &J, context: &C) -> Box<T> {
        let call = self.map.get(name).unwrap();
        return call(&json, context);
    }
}

#[cfg(test)]
pub mod test {
    use crate::{
        activation::{
            abs::{Activation, ActivationSerialised},
            relu::ReLu,
        },
        serial::model_reader::ModelReader,
    };

    #[test]
    pub fn inject_activation() {
        let model_reader = ModelReader::default();

        let test_cap_value = 3333.0;

        let injector = model_reader.get_activation_di();
        let sample = ReLu {
            cap: test_cap_value,
        };
        let serialized = sample.as_serialized();
        let deserialised = injector.create(ReLu::NAME, &serialized.json, &model_reader);
        dbg!(&deserialised);

        assert! {
            deserialised.as_ref().as_any().is::<ReLu>() &&
            deserialised.as_ref().as_any().downcast_ref::<ReLu>().unwrap().cap.eq(&test_cap_value)
        }

        let serialized = sample.as_serialized();
        let json = serde_json::to_string(&serialized).unwrap();
        dbg!(&json);
        let reverse_json: ActivationSerialised = serde_json::from_str(&json).unwrap();
        dbg!(&reverse_json);
    }
}
