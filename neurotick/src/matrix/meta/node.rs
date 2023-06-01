use crate::layer::abs::LBRef;

pub enum LBNode {
    DeadEnd,
    SingleParent(LBRef),
    MultipleParent(Vec<LBRef>),
}
