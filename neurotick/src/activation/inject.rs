use std::collections::HashMap;

use super::{
    abs::{ActivationSerialised, Activation, ActivationHandler},
    relu::{ReLuHandler}, lerelu::LeReLuHandler,
};

pub struct InjectionWrap(Box<dyn ActivationHandler>);

pub struct ActivationInjector {
    map: HashMap<String, InjectionWrap>,
}

impl ActivationInjector {
    pub fn empty() -> ActivationInjector {
        return ActivationInjector {
            map: HashMap::new(),
        };
    }

    pub fn default_injector() -> ActivationInjector {
        let mut injector: ActivationInjector = ActivationInjector::empty();

        injector.register(ReLuHandler {});
        injector.register(LeReLuHandler {});
        return injector;
    }

    pub fn register(&mut self, parser: impl ActivationHandler + 'static) {
        self.map.insert(
            parser.name(),
            InjectionWrap {
                0: Box::new(parser),
            },
        );
    }

    pub fn serialize(&self, activation: &impl Activation) -> ActivationSerialised {
        return ActivationSerialised {
            name: activation.name(),
            json: activation.as_json(),
        };
    }

    pub fn deserialize(&self, ser: &ActivationSerialised) -> Box<dyn Activation> {
        match self.map.get(&ser.name) {
            Some(parser) => {
                let act = parser.0.read(&ser.json);
                return act;
            }
            None => panic!("Activation {:?} not supported", ser.name),
        }
    }
}

#[cfg(test)]
pub mod test {
    use crate::activation::{relu::ReLu};

    use super::ActivationInjector;

    #[test]
    fn test_default_injector() {
        let injector: ActivationInjector = ActivationInjector::default_injector();

        let serialised = injector.serialize(&ReLu::default());
        dbg!(&serialised);
        let deserialised = injector.deserialize(&serialised);
        dbg!(&deserialised);
    }
}
