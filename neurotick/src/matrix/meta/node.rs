use crate::layer::abs::LayerRef;

/**
 * Descriptor of graph structure. Represents and holds the references to parent nodes.
 */
pub enum LayerType {
    DeadEnd,
    SingleParent(LayerRef),
    MultipleParent(Vec<LayerRef>),
}
