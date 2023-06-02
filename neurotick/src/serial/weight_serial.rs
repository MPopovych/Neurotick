use super::matrix_serial::MatrixPack;

#[allow(dead_code)]

pub struct WeightSerialised {
    name: String,
    trainable: bool,
    pack: MatrixPack
}