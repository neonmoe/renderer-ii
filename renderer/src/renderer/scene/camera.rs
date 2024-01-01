use glam::{Mat4, Quat, Vec3};

use crate::renderer::pipeline_parameters::ProjViewTransforms;
use crate::renderer::scene::coordinate_system::CoordinateSystem;

/// Creates the projection matrix, with an infinite far plane if `far` is
/// [None]. Z is always reversed (1 = near, 0 = far).
fn create_proj(width: f32, height: f32, near: f32, far: Option<f32>) -> Mat4 {
    let fov = 74f32.to_radians();
    let aspect_ratio = width / height;
    if let Some(far) = far {
        projection_reverse_z(fov, aspect_ratio, near, far)
    } else {
        projection_reverse_z_with_inf_far(fov, aspect_ratio, near)
    }
}

// Weird terminology collision: the coordinate space we're working in is
// right-handed (Vulkan clip space: +X right, +Y down, +Z forward), yet the
// projection matrix is called left-handed (negative determinant, I guess?).

/// `M_I` * `P_LH` from: <https://iolite-engine.com/blog_posts/reverse_z_cheatsheet>
fn projection_reverse_z(fov: f32, aspect_ratio: f32, near: f32, far: f32) -> Mat4 {
    let flip_z = Mat4::from_scale_rotation_translation(Vec3::new(1.0, 1.0, -1.0), Quat::IDENTITY, Vec3::new(0.0, 0.0, 1.0));
    flip_z * proj(fov, aspect_ratio, far / (far - near), 1.0, -(far * near) / (far - near))
}

/// `P_RevLH` as `z_f` approaches inf from: <https://iolite-engine.com/blog_posts/reverse_z_cheatsheet>
fn projection_reverse_z_with_inf_far(fov: f32, aspect_ratio: f32, near: f32) -> Mat4 {
    proj(fov, aspect_ratio, 0.0, 1.0, near)
}

fn proj(fov: f32, aspect_ratio: f32, m22: f32, m23: f32, m32: f32) -> Mat4 {
    let sy = 1.0 / (fov / 2.0).tan();
    let sx = sy / aspect_ratio;
    Mat4::from_cols_array(&[sx, 0.0, 0.0, 0.0, 0.0, sy, 0.0, 0.0, 0.0, 0.0, m22, m23, 0.0, 0.0, m32, 0.0])
}

pub struct Camera {
    pub position: Vec3,
    pub orientation: Quat,
    pub near: f32,
    /// If None, an infinite projection is used.
    pub far: Option<f32>,
}

impl Default for Camera {
    fn default() -> Self {
        Camera { position: Vec3::ZERO, orientation: Quat::IDENTITY, near: 0.1, far: Some(100.0) }
    }
}

impl Camera {
    #[profiling::function]
    pub(crate) fn create_proj_view_transforms(&self, width: f32, height: f32, world_space: CoordinateSystem) -> ProjViewTransforms {
        let view = Mat4::from_rotation_translation(self.orientation, self.position).inverse();
        let vk_space_from_world_space = world_space.create_transform_to(&CoordinateSystem::VULKAN);
        let projection = create_proj(width, height, self.near, self.far) * vk_space_from_world_space;
        ProjViewTransforms { projection, view }
    }
}
