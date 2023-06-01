use super::matrix_serial::MatrixPack;

pub struct WeightSerialised {
    name: String,
    trainable: bool,
    pack: MatrixPack
}