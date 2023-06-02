use crate::layer::abs::LBRef;

/**
 * Descriptor of graph structure. Represents and holds the references to parent nodes.
 */
pub enum LBNode {
    DeadEnd,
    SingleParent(LBRef),
    MultipleParent(Vec<LBRef>),
}
