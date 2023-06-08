use std::fmt::Debug;
use std::ops::{Add, BitAnd, Mul};

use base64::Engine;
use ndarray::iter::{AxisIter, Iter};
use ndarray::{Array2, ArrayView, Axis, Ix1, Ix2};
use serde::{Deserialize, Deserializer, Serialize};

use crate::serial::matrix_serial::{MatrixPack, MatrixSerial};
use crate::suppliers::suppliers::Supplier;
use crate::utils::extensions::Distinct;

pub mod test;

#[derive(Clone)]
pub struct NDMatrix {
    pub width: usize,
    pub height: usize,
    pub values: Array2<f32>,
}

impl NDMatrix {
    pub fn from_raw_vec(width: usize, height: usize, raw_vec: Vec<f32>) -> NDMatrix {
        let data = Array2::from_shape_vec((height, width), raw_vec).unwrap();
        return NDMatrix {
            width,
            height,
            values: data,
        };
    }

    pub fn from_supply(width: usize, height: usize, supply: impl Supplier) -> NDMatrix {
        let mut own_mut = supply;
        return own_mut.supply_matrix(width, height);
    }

    pub fn with(width: usize, height: usize, with: Array2<f32>) -> NDMatrix {
        let shape = with.shape();
        if shape[0] != height || shape[1] != width {
            panic!(
                "Wrong array sizes as input {}:{} with {}:{}",
                width, height, shape[0], shape[1]
            )
        }
        return NDMatrix {
            width,
            height,
            values: with,
        };
    }

    pub fn new(width: usize, height: usize) -> NDMatrix {
        return NDMatrix {
            width,
            height,
            values: Array2::default((height, width)),
        };
    }

    pub fn constant(width: usize, height: usize, constant: f32) -> NDMatrix {
        return NDMatrix {
            width,
            height,
            values: Array2::from_elem((height, width), constant),
        };
    }

    pub fn mat_mul(a: &NDMatrix, b: &NDMatrix) -> NDMatrix {
        let r = (&a.values).dot(&b.values);
        return NDMatrix {
            width: b.width,
            height: a.height,
            values: r,
        };
    }

    pub fn hadamard(a: &NDMatrix, b: &NDMatrix) -> NDMatrix {
        a.check_same_shape(&b);
        let r = (&a.values).mul(&b.values);
        return NDMatrix {
            width: b.width,
            height: a.height,
            values: r,
        };
    }

    pub fn hadamard_row_wise(a: &NDMatrix, b: &NDMatrix) -> NDMatrix {
        a.check_same_width(&b);
        b.check_height_eq(1);

        let r = (&a.values).mul(&b.values);
        return NDMatrix {
            width: b.width,
            height: a.height,
            values: r,
        };
    }

    pub fn add(a: &NDMatrix, b: &NDMatrix) -> NDMatrix {
        a.check_same_width(&b);
        let r = (&a.values).add(&b.values);
        return NDMatrix {
            width: a.width,
            height: a.height,
            values: r,
        };
    }

    pub fn get(&self, y: usize, x: usize) -> f32 {
        self.check_args(y, x);
        return *self.values.get((y, x)).unwrap();
    }

    pub fn set(&mut self, y: usize, x: usize, value: f32) {
        self.check_args(y, x);
        *self.values.get_mut((y, x)).unwrap() = value;
    }

    pub fn iter_all(&self) -> Iter<'_, f32, Ix2> {
        return self.values.iter();
    }

    pub fn iter_rows(&self) -> AxisIter<'_, f32, Ix1> {
        return self.values.axis_iter(Axis(0));
    }

    pub fn iter_columns(&self) -> AxisIter<'_, f32, Ix1> {
        return self.values.axis_iter(Axis(1));
    }

    fn check_args(&self, y: usize, x: usize) {
        if y > self.height {
            panic!("Overflow of height: {} {}", y, self.height)
        }
        if x > self.width {
            panic!("Overflow of width: {} {}", x, self.width)
        }
    }

    fn check_same_width(&self, rhs: &Self) {
        if self.width != rhs.width {
            panic!("Different width: {} {}", self.width, rhs.width)
        }
    }

    #[allow(dead_code)]
    fn check_width_eq(&self, eq: usize) {
        if self.width != eq {
            panic!("Different width: {} {}", self.width, eq)
        }
    }

    fn check_same_height(&self, rhs: &Self) {
        if self.height != rhs.height {
            panic!("Different height: {} {}", self.height, rhs.height)
        }
    }

    fn check_height_eq(&self, eq: usize) {
        if self.height != eq {
            panic!("Different height: {} {}", self.height, eq)
        }
    }

    fn check_same_shape(&self, rhs: &Self) {
        self.check_same_width(rhs);
        self.check_same_height(rhs);
    }
}

impl NDMatrix {
    pub fn concat_horizontal(array: &[&NDMatrix]) -> NDMatrix {
        let width = array.iter().map(|m| m.width).sum();

        let height_set = array.iter().distinct_vec(|m| m.height);
        if height_set.len() != 1 {
            dbg!(&height_set);
            panic!("Concat not possible due to different heights");
        }
        let height = height_set[0];

        let views = array
            .iter()
            .map(|m| m.values.view())
            .collect::<Vec<ArrayView<'_, f32, Ix2>>>();
        let concat = ndarray::concatenate(Axis(1), &views[..]);
        return NDMatrix {
            width: width,
            height: height,
            values: concat.unwrap(),
        };
    }
}

impl<'a, 'b> Mul<&'b NDMatrix> for &'a NDMatrix {
    type Output = NDMatrix;

    fn mul(self, rhs: &'b NDMatrix) -> Self::Output {
        NDMatrix::mat_mul(self, rhs)
    }
}

impl<'a, 'b> BitAnd<&'b NDMatrix> for &'a NDMatrix {
    type Output = NDMatrix;

    fn bitand(self, rhs: &'b NDMatrix) -> Self::Output {
        NDMatrix::hadamard(self, rhs)
    }
}

impl<'a, 'b> Add<&'b NDMatrix> for &'a NDMatrix {
    type Output = NDMatrix;

    fn add(self, rhs: &'b NDMatrix) -> Self::Output {
        NDMatrix::add(self, rhs)
    }
}

impl Debug for NDMatrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "NDMatrix [{}:{}] {:?} -> \n{}",
            self.width,
            self.height,
            self.values.shape().clone(),
            &self.values,
        )
    }
}

impl MatrixSerial<NDMatrix> for NDMatrix {
    fn pack(&self) -> MatrixPack {
        let bytes: Vec<u8> = self
            .values
            .clone()
            .iter()
            .map(|f| f.to_be_bytes())
            .flatten()
            .collect();
        let encoded = base64::engine::general_purpose::STANDARD_NO_PAD.encode(bytes);
        MatrixPack {
            width: self.width,
            height: self.height,
            data: encoded,
        }
    }

    /**
     * This should panic if the byte packing is wrong
     */
    fn unpack(pack: &MatrixPack) -> NDMatrix {
        let decoded = base64::engine::general_purpose::STANDARD_NO_PAD
            .decode(&pack.data)
            .unwrap();
        let float_array = decoded
            .chunks_exact(4)
            .into_iter()
            .map(|be| f32::from_be_bytes(be.try_into().unwrap()))
            .collect();
        NDMatrix::from_raw_vec(pack.width, pack.height, float_array)
    }
}

impl Serialize for NDMatrix {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.pack().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for NDMatrix {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let pack: MatrixPack = MatrixPack::deserialize(deserializer)?;
        return Ok(NDMatrix::unpack(&pack));
    }
}
