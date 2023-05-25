use crate::renderer::pipelines::pipeline_parameters::ProjViewTransforms;
use glam::{Mat4, Quat, Vec3};

fn create_proj_view(camera_transform: Mat4, width: f32, height: f32, near: f32, far: f32) -> ProjViewTransforms {
    let fov = 74f32.to_radians();
    let aspect_ratio = width / height;
    ProjViewTransforms {
        projection: reverse_z_rh_projection(fov, aspect_ratio, near, far),
        view: camera_transform.inverse(),
    }
}

fn reverse_z_rh_projection(fov: f32, aspect_ratio: f32, near: f32, far: f32) -> Mat4 {
    let sy = 1.0 / (fov / 2.0).tan();
    let sx = sy / aspect_ratio;
    let m22 = -(far / (near - far)) - 1.0;
    let m32 = -(near * far) / (near - far);
    Mat4::from_cols_array(&[sx, 0.0, 0.0, 0.0, 0.0, -sy, 0.0, 0.0, 0.0, 0.0, m22, -1.0, 0.0, 0.0, m32, 0.0])
}

#[allow(dead_code)]
fn reverse_z_rh_infinite_projection(fov: f32, aspect_ratio: f32, near: f32) -> Mat4 {
    let sy = 1.0 / (fov / 2.0).tan();
    let sx = sy / aspect_ratio;
    Mat4::from_cols_array(&[sx, 0.0, 0.0, 0.0, 0.0, -sy, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, near, 0.0])
}

#[allow(dead_code)]
fn z_rh_infinite_projection(fov: f32, aspect_ratio: f32, near: f32) -> Mat4 {
    let sy = 1.0 / (fov / 2.0).tan();
    let sx = sy / aspect_ratio;
    let near = -2.0 * near;
    Mat4::from_cols_array(&[sx, 0.0, 0.0, 0.0, 0.0, -sy, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0, near, 0.0])
}

pub struct Camera {
    pub position: Vec3,
    pub orientation: Quat,
    pub near: f32,
    pub far: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Camera {
            position: Vec3::ZERO,
            orientation: Quat::IDENTITY,
            near: 0.1,
            far: 100.0,
        }
    }
}

impl Camera {
    #[profiling::function]
    pub(crate) fn create_global_transforms(&self, width: f32, height: f32) -> ProjViewTransforms {
        create_proj_view(
            Mat4::from_rotation_translation(self.orientation, self.position),
            width,
            height,
            self.near,
            self.far,
        )
    }
}
