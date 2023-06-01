use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
pub struct MatrixPack {
    pub width: usize,
    pub height: usize,
    #[serde(with = "serde_bytes")]
    pub data: Vec<u8>,
}

pub trait MatrixSerial<T: Sized> {
    fn pack(&self) -> MatrixPack;
    fn unpack(pack: &MatrixPack) -> T
    where
        Self: Sized;
}
