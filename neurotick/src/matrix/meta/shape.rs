use serde::{Serialize, Deserialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Shape {
    /** Specified size, ex.: of features or timeseries size */
    Const(usize),
    /** Up to the layer impl to handle, should copy parent shape */
    Repeat,
    /** Up to the layer impl to handle, may be used in tokenising, variable timeseries, etc. */
    Variable,
}

impl Shape {
    /**
     * Will panic if the shape in not constant
     */
    pub fn unwrap_to_conts(&self) -> usize {
        return match self {
            Shape::Const(c) => *c,
            Shape::Repeat => panic!("No feature count"),
            Shape::Variable => panic!("No feature count"),
        };
    }
}
