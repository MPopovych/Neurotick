use std::fmt::Debug;
use std::ops::{Add, BitAnd, Mul};

use ndarray::Array2;

pub mod test;

#[derive(Clone)]
pub struct NDMatrix {
    pub width: usize,
    pub height: usize,
    values: Array2<f32>,
}

impl NDMatrix {
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

    pub fn mul(a: &NDMatrix, b: &NDMatrix) -> NDMatrix {
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

    fn check_same_height(&self, rhs: &Self) {
        if self.height != rhs.height {
            panic!("Different height: {} {}", self.height, rhs.height)
        }
    }

    fn check_same_shape(&self, rhs: &Self) {
        self.check_same_width(rhs);
        self.check_same_height(rhs);
    }
}

impl<'a, 'b> Mul<&'b NDMatrix> for &'a NDMatrix {
    type Output = NDMatrix;

    fn mul(self, rhs: &'b NDMatrix) -> Self::Output {
        NDMatrix::mul(self, rhs)
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
