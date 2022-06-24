use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Quat, Vec3};

#[repr(C)]
#[derive(Clone, Copy)]
pub struct GlobalTransforms {
    _projection: Mat4,
    _view: Mat4,
}

// Mat4's are Pods, therefore they are Zeroable, therefore this is too.
unsafe impl Zeroable for GlobalTransforms {}

// repr(c) + Mat4's are Pods since glam has the bytemuck feature enabled.
unsafe impl Pod for GlobalTransforms {}

impl GlobalTransforms {
    fn new(camera_transform: Mat4, width: f32, height: f32, near: f32, far: f32) -> GlobalTransforms {
        let fov = 74f32.to_radians();
        let aspect_ratio = width / height;
        GlobalTransforms {
            _projection: reverse_z_rh_projection(fov, aspect_ratio, near, far),
            _view: camera_transform.inverse(),
        }
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
    pub(crate) fn create_global_transforms(&self, width: f32, height: f32) -> GlobalTransforms {
        // Input coordinates (from arrow-game, which uses Trenchbroom for levels) are in "Quake" coordinates.
        // Post-camera-transform coordinates should be in Vulkan coordinates.
        // Quake: +X: forward, +Y: left, +Z: up
        // Vulkan: +X: right, +Y: up, +Z: backward
        let quake_camera_transform = Mat4::from_rotation_translation(self.orientation, self.position);

        //  0  0 -1  0
        // -1  0  0  0
        //  0  1  0  0
        //  0  0  0  1
        let mut quake_to_y_up_rh = Mat4::ZERO;
        quake_to_y_up_rh.col_mut(0).y = -1.0;
        quake_to_y_up_rh.col_mut(1).z = 1.0;
        quake_to_y_up_rh.col_mut(2).x = -1.0;
        quake_to_y_up_rh.col_mut(3).w = 1.0;

        let camera_transform = quake_camera_transform * quake_to_y_up_rh;
        GlobalTransforms::new(camera_transform, width, height, self.near, self.far)
    }
}
