use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
pub struct MatrixPack {
    pub width: usize,
    pub height: usize,
    pub data: String,
}

pub trait MatrixSerial<T: Sized> {
    fn pack(&self) -> MatrixPack;
    fn unpack(pack: &MatrixPack) -> T
    where
        Self: Sized;
}
