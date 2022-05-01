use crate::mesh::Mesh;
use crate::{Gltf, Material};
use glam::Mat4;

pub(crate) struct MeshIterItem<'a> {
    pub node_index: usize,
    pub mesh: &'a Mesh,
    pub material: &'a Material,
    pub skin: Option<usize>,
    pub transform: Mat4,
}

pub(crate) struct MeshIter<'a> {
    gltf: &'a Gltf,
    node_queue: Vec<usize>,
    current_sub_iter: Option<InnerMeshIter<'a>>,
}

impl MeshIter<'_> {
    pub(crate) fn new(gltf: &Gltf, node_queue: Vec<usize>) -> MeshIter {
        MeshIter {
            gltf,
            node_queue,
            current_sub_iter: None,
        }
    }
}

impl<'a> Iterator for MeshIter<'a> {
    type Item = MeshIterItem<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(next) = self.current_sub_iter.as_mut().and_then(Iterator::next) {
            Some(next)
        } else {
            let node_index = self.node_queue.pop()?;
            let node = self.gltf.nodes.get(node_index)?;
            if let Some(children) = &node.children {
                for child in children {
                    self.node_queue.push(*child);
                }
            }
            if let Some(mesh_index) = node.mesh {
                self.current_sub_iter
                    .insert(InnerMeshIter {
                        gltf: self.gltf,
                        skin: node.skin,
                        node_index,
                        mesh_index,
                        primitive_index: 0,
                        transform: node.transform,
                    })
                    .next()
            } else {
                self.next()
            }
        }
    }
}

struct InnerMeshIter<'a> {
    gltf: &'a Gltf,
    skin: Option<usize>,
    node_index: usize,
    mesh_index: usize,
    primitive_index: usize,
    transform: Mat4,
}

impl<'a> Iterator for InnerMeshIter<'a> {
    type Item = MeshIterItem<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let meshes = self.gltf.meshes.get(self.mesh_index)?;
        let (mesh, material_index) = meshes.get(self.primitive_index)?;
        let material = self.gltf.materials.get(*material_index).unwrap();
        self.primitive_index += 1;
        Some(MeshIterItem {
            node_index: self.node_index,
            mesh,
            material,
            skin: self.skin,
            transform: self.transform,
        })
    }
}
