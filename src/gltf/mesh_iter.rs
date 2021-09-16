use super::{Gltf, Mesh};
use ultraviolet::Mat4;

pub struct MeshIter<'a> {
    gltf: &'a Gltf<'a>,
    node_queue: Vec<usize>,
}

impl MeshIter<'_> {
    pub(crate) fn new<'a>(gltf: &'a Gltf<'a>, node_queue: Vec<usize>) -> MeshIter<'a> {
        MeshIter { gltf, node_queue }
    }
}

impl<'a> Iterator for MeshIter<'a> {
    type Item = (&'a [Mesh<'a>], Mat4);

    fn next(&mut self) -> Option<Self::Item> {
        let node_index = self.node_queue.pop()?;
        let node = self.gltf.nodes.get(node_index)?;
        if let Some(children) = &node.children {
            for child in children {
                self.node_queue.push(*child);
            }
        }
        let mesh = self.gltf.meshes.get(node.mesh?)?;
        Some((mesh, node.transform))
    }
}
