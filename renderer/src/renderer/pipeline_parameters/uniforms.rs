use bytemuck::{Pod, Zeroable};
use glam::{Mat4, UVec4, Vec4};

use crate::renderer::pipeline_parameters::constants::{MAX_DRAW_CALLS, MAX_JOINT_COUNT, MAX_MATERIALS};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ProjViewTransforms {
    pub projection: Mat4,
    pub view: Mat4,
}

/// The per-frame uniform buffer.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct RenderSettings {
    pub debug_value: u32,
}

pub trait StructureOfArraysUniform {
    fn soa_size(&self) -> usize;
    fn soa_write(&self, dst: &mut [u8]);
    /// Sets the length and zeroes out the empty area.
    fn soa_resize(&mut self, new_len: usize, fill_with_zeroes: bool);
}

macro_rules! declare_soa_uniform_struct {
    {
        pub struct $struct_name:ident[$max_count:expr] {
            $(pub $field_name:ident: $field_type:ty),*,
        }
    } => {
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        #[repr(C)]
        pub struct $struct_name {
            $(pub $field_name: $field_type),*,
        }
        #[allow(dead_code)]
        impl $struct_name {
            pub const MAX_COUNT: usize = $max_count;
            pub const DESCRIPTOR_SIZE: u64 = ($max_count * core::mem::size_of::<$struct_name>()) as u64;
        }
        impl crate::renderer::pipeline_parameters::uniforms::StructureOfArraysUniform
        for arrayvec::ArrayVec<$struct_name, { $max_count }> {
            fn soa_size(&self) -> usize {
                core::mem::size_of::<$struct_name>() * $struct_name::MAX_COUNT
            }
            #[allow(unused_assignments)]
            fn soa_write(&self, dst: &mut [u8]) {
                assert_eq!(self.soa_size(), dst.len());
                for (i, s) in self.iter().enumerate() {
                    let mut array_offset = 0;
                    $({
                        let element_size = core::mem::size_of::<$field_type>();
                        let offset = array_offset + i * element_size;
                        let from: &[$field_type] = &[s.$field_name];
                        let from: &[u8] = bytemuck::cast_slice(from);
                        dst[offset..offset + element_size].copy_from_slice(from);
                        array_offset += $max_count * element_size;
                    })*
                }
            }
            fn soa_resize(&mut self, new_len: usize, fill_with_zeroes: bool) {
                if new_len < self.len() {
                    self.truncate(new_len);
                } else {
                    assert!(new_len <= $max_count);
                    let old_len = self.len();
                    unsafe { self.set_len(new_len) };
                    if fill_with_zeroes {
                        let expanded_part = &mut self[old_len..new_len];
                        let expanded_part = bytemuck::cast_slice_mut::<$struct_name, u8>(expanded_part);
                        expanded_part.fill(0);
                    }
                }
            }
        }
    };
}

declare_soa_uniform_struct! {
    pub struct MaterialIds[MAX_DRAW_CALLS as usize] {
        pub material_id: u32,
    }
}

declare_soa_uniform_struct! {
    pub struct JointsOffsets[MAX_JOINT_COUNT as usize] {
        pub joints_offset: u32,
    }
}

declare_soa_uniform_struct! {
    pub struct PbrFactors[MAX_MATERIALS as usize] {
        pub base_color: Vec4,
        pub emissive_and_occlusion: Vec4,
        pub alpha_rgh_mtl_normal: Vec4,
        pub textures: UVec4,
    }
}

declare_soa_uniform_struct! {
    pub struct ImGuiDrawCmd[MAX_MATERIALS as usize] {
        pub clip_rect: [f32; 4],
        pub texture_index: u32,
    }
}

#[cfg(test)]
mod soa_uniforms_tests {
    use arrayvec::ArrayVec;
    use bytemuck::{Pod, Zeroable};
    use glam::Vec2;

    use super::StructureOfArraysUniform;

    declare_soa_uniform_struct! {
        pub struct Foo[3] {
            pub a: Vec2,
            pub b: i32,
        }
    }

    #[derive(Clone, Copy, Zeroable, Pod)]
    #[repr(C)]
    pub struct FooManualSoa {
        pub a: [Vec2; 3],
        pub b: [i32; 3],
    }

    #[test]
    fn test_soa_uniform() {
        let mut array = ArrayVec::<Foo, { Foo::MAX_COUNT }>::new();
        array.push(Foo { a: Vec2::new(1.0, 2.0), b: 3 });
        array.push(Foo { a: Vec2::new(4.0, 5.0), b: 6 });
        array.push(Foo { a: Vec2::new(7.0, 8.0), b: 9 });
        let expected = &[FooManualSoa { a: [Vec2::new(1.0, 2.0), Vec2::new(4.0, 5.0), Vec2::new(7.0, 8.0)], b: [3, 6, 9] }];
        let expected = bytemuck::cast_slice::<FooManualSoa, u8>(expected);
        let size = array.soa_size();
        let mut soa_buffer = vec![0; size];
        array.soa_write(&mut soa_buffer);
        assert_eq!(expected, &soa_buffer);
    }
}
