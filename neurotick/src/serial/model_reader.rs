use crate::{
    activation::{
        abs::{Activation, ActivationVirtual},
        lerelu::LeakyReLu,
        none::NoneAct,
        relu::ReLu,
        softmax::SoftMax,
    },
    layer::{
        abs::{LayerBase, LayerPropagateEnum},
        dense::{Dense, DenseImpl},
        input::{Input, InputImpl}, concat::{Concat, ConcatImpl},
    },
    utils::{injector::GenericInjector, json_wrap::JsonWrap},
};

pub struct ModelReader {
    activation_injector: GenericInjector<dyn Activation, JsonWrap, ModelReader>,
    layer_injector: GenericInjector<LayerPropagateEnum, JsonWrap, ModelReader>,
}

impl ModelReader {
    pub fn default() -> ModelReader {
        ModelReader {
            activation_injector: GenericInjector::default_activation(),
            layer_injector: GenericInjector::default_layer(),
        }
    }

    pub fn get_activation_di(&self) -> &GenericInjector<dyn Activation, JsonWrap, ModelReader> {
        return &self.activation_injector;
    }

    pub fn get_layer_di(&self) -> &GenericInjector<LayerPropagateEnum, JsonWrap, ModelReader> {
        return &self.layer_injector;
    }
}

impl GenericInjector<dyn Activation, JsonWrap, ModelReader> {
    pub fn default_activation() -> GenericInjector<dyn Activation, JsonWrap, ModelReader> {
        let mut injector: GenericInjector<dyn Activation, JsonWrap, ModelReader> =
            GenericInjector::new();

        injector.register(NoneAct::NAME, |_json, _reader| Box::new(NoneAct::default()));

        injector.register(ReLu::NAME, |json, _reader| ReLu::from_json(&json));

        injector.register(LeakyReLu::NAME, |json, _reader| LeakyReLu::from_json(&json));

        injector.register(SoftMax::NAME, |_, _reader| Box::new(SoftMax::default()));
        return injector;
    }
}

impl GenericInjector<LayerPropagateEnum, JsonWrap, ModelReader> {
    pub fn default_layer() -> GenericInjector<LayerPropagateEnum, JsonWrap, ModelReader> {
        let mut injector: GenericInjector<LayerPropagateEnum, JsonWrap, ModelReader> =
            GenericInjector::new();

        injector.register(Input::NAME, |json, reader| {
            Box::new(InputImpl::create_from_ser(json, reader))
        });

        injector.register(Dense::NAME, |json, reader| {
            Box::new(DenseImpl::create_from_ser(json, reader))
        });

        injector.register(Concat::NAME, |json, reader| {
            Box::new(ConcatImpl::create_from_ser(json, reader))
        });
        return injector;
    }
}
