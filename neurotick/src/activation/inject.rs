use std::collections::HashMap;

use super::{
    abs::{Activation, ActivationSerialised},
    lerelu::LeakyReLu,
    relu::ReLu,
};

pub struct InjectionWrap(Box<dyn Activation>);

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

        injector.inject(ReLu::default());
        injector.inject(LeakyReLu::default());
        return injector;
    }

    pub fn inject<T: Activation>(&mut self, parser: T) {
        self.map.insert(
            parser.name(),
            InjectionWrap {
                0: Box::new(parser),
            },
        );
    }

    pub fn serialize(&self, activation: &Box<dyn Activation>) -> ActivationSerialised {
        match self.map.get(&activation.name()) {
            Some(parser) => return parser.0.write_json(activation),
            None => panic!("Activation {:?} not supported", activation.name()),
        }
    }

    pub fn parse_ser(&self, ser: &ActivationSerialised) -> Box<dyn Activation> {
        match self.map.get(&ser.name) {
            Some(parser) => parser.0.read_json(&ser.json),
            None => panic!("Activation {:?} not supported", ser.name),
        }
    }
}

#[cfg(test)]
pub mod test {
    use crate::activation::{abs::Activation, relu::ReLu};

    use super::ActivationInjector;

    #[test]
    fn test_default_injector() {
        let injector: ActivationInjector = ActivationInjector::default_injector();

        let b1 = Box::new(ReLu::default()) as Box<dyn Activation>;

        let serialised = injector.serialize(&b1);
        dbg!(&serialised);
        let binding = injector.parse_ser(&serialised);
        let deserialised = binding.as_any().downcast_ref::<ReLu>().unwrap();
        dbg!(deserialised);
    }
}
