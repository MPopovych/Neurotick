use std::collections::HashMap;

use super::{
    abs::{Activation, ActivationHandler, ActivationSerialised},
    lerelu::LeReLuHandler,
    relu::ReLuHandler,
};

/**
 *  This a helper for introducing custom activation functions.
 *  By registering and properly implementing handlers the ModelReader will restore the structures as traits.
 *  Parsing may and should crash when encountering an unknown activation object.
 *  Deserializing return a boxed trait, which than can be applied to a layer.
 */
pub struct ActivationInjector {
    map: HashMap<String, Box<dyn ActivationHandler>>,
}

impl ActivationInjector {
    pub fn empty() -> ActivationInjector {
        return ActivationInjector {
            map: HashMap::new(),
        };
    }

    /**
     * Returns the injector pre-filled with library supported activation functions
     */
    pub fn default_injector() -> ActivationInjector {
        let mut injector: ActivationInjector = ActivationInjector::empty();

        injector.register(ReLuHandler {});
        injector.register(LeReLuHandler {});
        return injector;
    }

    pub fn register(&mut self, parser: impl ActivationHandler + 'static) {
        self.map.insert(parser.name(), Box::new(parser));
    }

    pub fn serialize(&self, activation: &impl Activation) -> ActivationSerialised {
        return activation.as_serialized();
    }

    pub fn deserialize(&self, ser: &ActivationSerialised) -> Box<dyn Activation> {
        match self.map.get(&ser.name) {
            Some(parser) => {
                return parser.read(&ser.json);
            }
            None => panic!("Activation {:?} not supported", ser.name),
        }
    }
}

#[cfg(test)]
pub mod test {
    use crate::activation::{relu::ReLu, abs::ActivationSerialised};

    use super::ActivationInjector;

    #[test]
    fn test_default_injector() {
        let test_cap_value = 3333.0;

        assert! {
            !ReLu::default().cap.eq(&test_cap_value)
        }

        let injector: ActivationInjector = ActivationInjector::default_injector();

        let serialised = injector.serialize(&ReLu {
            cap: test_cap_value,
        });
        let deserialised = injector.deserialize(&serialised);
        dbg!(&deserialised);

        assert! {
            serialised.name == ReLu::NAME &&
            deserialised.as_ref().as_any().is::<ReLu>() &&
            deserialised.as_ref().as_any().downcast_ref::<ReLu>().unwrap().cap.eq(&test_cap_value)
        }

        let json = serde_json::to_string(&serialised).unwrap();
        dbg!(&json);
        let reverse_json: ActivationSerialised = serde_json::from_str(&json).unwrap();
        dbg!(&reverse_json);
    }
}
