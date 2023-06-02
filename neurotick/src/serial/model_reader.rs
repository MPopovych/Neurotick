use crate::activation::inject::ActivationInjector;

pub struct ModelReader {
    activation_injector: ActivationInjector,
}

impl ModelReader {
    pub fn get_activation_di(&self) -> &ActivationInjector {
        return &self.activation_injector
    }
}
