use crate::matrix::nmatrix::NDMatrix;
use rand::{rngs::ThreadRng, thread_rng, Rng};
use rand_distr::StandardNormal;
use std::cell::RefCell;

pub enum Suppliers {
    Zero(RefCell<ZeroSupplier>),
    RandomNormal(RefCell<RandomNormalSupplier>),
    RandomUniform(RefCell<RandomUniformSupplier>),
    GlorothNormal(RefCell<GlorothNormalSupplier>),
    GlorothUniform(RefCell<GlorothUniformSupplier>),
}

pub trait Supplier {
    fn supply_single(&mut self, in_f: usize, out_f: usize) -> f32;
    fn supply_matrix(&mut self, width: usize, height: usize) -> NDMatrix;
    fn into_enum(self) -> Suppliers;
}

impl Suppliers {
    pub fn supply_single(&self, in_f: usize, out_f: usize) -> f32 {
        match self {
            Suppliers::Zero(rc) => rc.borrow_mut().supply_single(in_f, out_f),
            Suppliers::RandomNormal(rc) => rc.borrow_mut().supply_single(in_f, out_f),
            Suppliers::RandomUniform(rc) => rc.borrow_mut().supply_single(in_f, out_f),
            Suppliers::GlorothNormal(rc) => rc.borrow_mut().supply_single(in_f, out_f),
            Suppliers::GlorothUniform(rc) => rc.borrow_mut().supply_single(in_f, out_f),
        }
    }

    pub fn supply_matrix(&self, width: usize, height: usize) -> NDMatrix {
        match self {
            Suppliers::Zero(rc) => rc.borrow_mut().supply_matrix(width, height),
            Suppliers::RandomNormal(rc) => rc.borrow_mut().supply_matrix(width, height),
            Suppliers::RandomUniform(rc) => rc.borrow_mut().supply_matrix(width, height),
            Suppliers::GlorothNormal(rc) => rc.borrow_mut().supply_matrix(width, height),
            Suppliers::GlorothUniform(rc) => rc.borrow_mut().supply_matrix(width, height),
        }
    }
}

pub struct ZeroSupplier;

pub struct RandomNormalSupplier {
    pub mean: f32,
    pub std_dev: f32,
    rng: ThreadRng,
}

pub struct RandomUniformSupplier {
    pub max: f32,
    pub min: f32,
    rng: ThreadRng,
}

pub struct GlorothNormalSupplier {
    rng: ThreadRng,
}

pub struct GlorothUniformSupplier {
    rng: ThreadRng,
}

impl ZeroSupplier {
    #[allow(dead_code)]
    pub fn new() -> ZeroSupplier {
        return ZeroSupplier {};
    }
}

impl Supplier for ZeroSupplier {
    fn supply_single(&mut self, _in_f: usize, _out_f: usize) -> f32 {
        return 0.0;
    }

    fn supply_matrix(&mut self, width: usize, height: usize) -> NDMatrix {
        return NDMatrix::constant(width, height, 0.0);
    }

    fn into_enum(self) -> Suppliers {
        return Suppliers::Zero(RefCell::new(self));
    }
}

impl RandomNormalSupplier {
    #[allow(dead_code)]
    pub fn new(mean: f32, std_dev: f32) -> RandomNormalSupplier {
        return RandomNormalSupplier {
            mean,
            std_dev,
            rng: thread_rng(),
        };
    }
}

impl Supplier for RandomNormalSupplier {
    fn supply_single(&mut self, _in_f: usize, _out_f: usize) -> f32 {
        return self.rng.sample::<f32, _>(StandardNormal) * self.std_dev - self.mean;
    }

    fn supply_matrix(&mut self, width: usize, height: usize) -> NDMatrix {
        let vec: Vec<f32> = (0..(width * height))
            .map(|_| self.rng.sample::<f32, _>(StandardNormal) * self.std_dev - self.mean)
            .collect();
        return NDMatrix::from_raw_vec(width, height, vec);
    }

    fn into_enum(self) -> Suppliers {
        return Suppliers::RandomNormal(RefCell::new(self));
    }
}

impl RandomUniformSupplier {
    #[allow(dead_code)]
    pub fn new(max: f32, min: f32) -> RandomUniformSupplier {
        return RandomUniformSupplier {
            max,
            min,
            rng: thread_rng(),
        };
    }
}

impl Supplier for RandomUniformSupplier {
    fn supply_single(&mut self, _in_f: usize, _out_f: usize) -> f32 {
        let range = self.max - self.min;
        return self.rng.gen::<f32>() * range + self.min;
    }

    fn supply_matrix(&mut self, width: usize, height: usize) -> NDMatrix {
        let range = self.max - self.min;
        let vec: Vec<f32> = (0..(width * height))
            .map(|_| self.rng.gen::<f32>() * range + self.min)
            .collect();
        return NDMatrix::from_raw_vec(width, height, vec);
    }

    fn into_enum(self) -> Suppliers {
        return Suppliers::RandomUniform(RefCell::new(self));
    }
}

impl GlorothUniformSupplier {
    #[allow(dead_code)]
    pub fn new() -> GlorothUniformSupplier {
        return GlorothUniformSupplier { rng: thread_rng() };
    }
}

impl Supplier for GlorothUniformSupplier {
    fn supply_single(&mut self, in_f: usize, out_f: usize) -> f32 {
        let limit = (6.0 / (in_f + out_f) as f32).sqrt();
        return (self.rng.gen::<f32>() * 2.0 - 1.0) * limit;
    }

    fn supply_matrix(&mut self, width: usize, height: usize) -> NDMatrix {
        let limit = (6.0 / (width + height) as f32).sqrt();
        let vec: Vec<f32> = (0..(width * height))
            .map(|_| (self.rng.gen::<f32>() * 2.0 - 1.0) * limit)
            .collect();
        return NDMatrix::from_raw_vec(width, height, vec);
    }

    fn into_enum(self) -> Suppliers {
        return Suppliers::GlorothUniform(RefCell::new(self));
    }
}

impl GlorothNormalSupplier {
    #[allow(dead_code)]
    pub fn new() -> GlorothNormalSupplier {
        return GlorothNormalSupplier { rng: thread_rng() };
    }
}

impl Supplier for GlorothNormalSupplier {
    fn supply_single(&mut self, in_f: usize, out_f: usize) -> f32 {
        let std_dev = (2.0 / (in_f + out_f) as f32).sqrt();
        return self.rng.sample::<f32, _>(StandardNormal) * std_dev;
    }

    fn supply_matrix(&mut self, width: usize, height: usize) -> NDMatrix {
        let std_dev = (2.0 / (width + height) as f32).sqrt();
        let vec: Vec<f32> = (0..(width * height))
            .map(|_| self.rng.sample::<f32, _>(StandardNormal) * std_dev)
            .collect();
        return NDMatrix::from_raw_vec(width, height, vec);
    }

    fn into_enum(self) -> Suppliers {
        return Suppliers::GlorothNormal(RefCell::new(self));
    }
}
