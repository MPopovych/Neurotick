use std::collections::HashMap;

use crate::serial::model_reader::ModelReader;

use super::{abs::LayerBase, dense::{Dense, DenseImpl}, input::{InputImpl, Input}};

pub struct LayerInjector {
    map: HashMap<String, Box<dyn Fn(&String, &ModelReader) -> Box<dyn LayerBase>>>
}

impl LayerInjector {
    pub fn default() -> LayerInjector{
        let mut def = LayerInjector { map: HashMap::new() };

        def.register(Dense::NAME.to_string(), |json: &String, mr: &ModelReader| {
            Box::new(DenseImpl::create_from_ser(json, mr))
        });

        def.register(Input::NAME.to_string(), |json, mr: &ModelReader| {
            Box::new(InputImpl::create_from_ser(json, mr))
        });

        return def
    }

    pub fn register<F: 'static>(&mut self, name: String, call: F) where F: Fn(&String, &ModelReader) -> Box<dyn LayerBase> {
        self.map.insert(name, Box::new(call));
    }

    pub fn create(&self, name: &String, json: &String, model_reader: &ModelReader) -> Box<dyn LayerBase> {
        let call = self.map.get(name).unwrap();
        return call(&json, model_reader)
    }
}
