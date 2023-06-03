use std::any::Any;

pub trait AsAny: 'static {
    fn as_any(&self) -> &dyn Any;
}