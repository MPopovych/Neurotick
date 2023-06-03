use std::collections::HashMap;

pub struct GenericInjector<T: ?Sized, C> {
    map: HashMap<String, Box<dyn Fn(&String, &C) -> Box<T>>>,
}

impl<T: ?Sized, C> GenericInjector<T, C> {
    pub fn new() -> GenericInjector<T, C> {
        return GenericInjector {
            map: HashMap::new(),
        };
    }

    pub fn register<F: 'static>(&mut self, name: &str, call: F)
    where
        F: Fn(&String, &C) -> Box<T>,
    {
        self.map.insert(name.to_string(), Box::new(call));
    }

    pub fn create(&self, name: &str, json: &String, context: &C) -> Box<T> {
        let call = self.map.get(name).unwrap();
        return call(&json, context);
    }
}

#[cfg(test)]
pub mod test {
    use crate::activation::{
        abs::{Activation, ActivationHandler, ActivationSerialised, ActivationSerialize},
        lerelu::{LeReLuHandler, LeakyReLu},
        relu::{ReLu, ReLuHandler},
    };

    use super::GenericInjector;

    #[test]
    pub fn inject_activation() {
        let test_cap_value = 3333.0;

        let mut injector: GenericInjector<dyn Activation, String> = GenericInjector::new();
        injector.register(ReLu::NAME, |json, _| {
            let handler = ReLuHandler {};
            return handler.read(json);
        });
        injector.register(LeakyReLu::NAME, |json, _| {
            let handler = LeReLuHandler {};
            return handler.read(json);
        });

        let sample = ReLu {
            cap: test_cap_value,
        };
        let json = sample.as_json();
        let deserialised = injector.create(ReLu::NAME, &json, &"".to_string());
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
