use core::cmp::Ordering;

use crate::renderer::descriptors::material::Material;
use crate::renderer::pipeline_parameters::PipelineIndex;
use crate::renderer::scene::mesh::Mesh;
use crate::vertex_library::VertexLibrary;

/// Contains the various parts that make up a single "draw", so that ordering a slice of these will
/// result in optimal `VkDrawIndexedIndirectCommand` structures. The tags will group together when the
/// following match, prioritizing matches from from top to bottom:
///
/// - Pipeline (groups equal descriptor binds)
/// - Vertex library (groups equal vertex and index buffer binds)
/// - Mesh, material (groups equal `VkDrawIndexedIndirectCommand` params)
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct DrawCallTag<'a> {
    pub pipeline: PipelineIndex,
    pub vertex_library: &'a VertexLibrary,
    pub mesh: &'a Mesh,
    pub material: &'a Material,
}

impl Ord for DrawCallTag<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        let inequality = |draw_before_other: bool| {
            if draw_before_other { Ordering::Less } else { Ordering::Greater }
        };
        if (self.pipeline as u32) != (other.pipeline as u32) {
            inequality((self.pipeline as u32) < (other.pipeline as u32))
        } else if self.vertex_library != other.vertex_library {
            inequality(self.vertex_library < other.vertex_library)
        } else if self.mesh != other.mesh {
            inequality(self.mesh < other.mesh)
        } else if self.material != other.material {
            inequality(self.material < other.material)
        } else {
            Ordering::Equal
        }
    }
}

impl PartialOrd for DrawCallTag<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
