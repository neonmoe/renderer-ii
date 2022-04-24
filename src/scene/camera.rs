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
    fn new(camera_transform: Mat4, width: f32, height: f32) -> GlobalTransforms {
        let fov = 74f32.to_radians();
        let aspect_ratio = width / height;
        // Lower values seem to cause Z-fighting in the sponza scene.
        // Might be better to use two projection matrixes for e.g. 0.1->5, 5->inf.
        let near = 0.5;
        GlobalTransforms {
            _projection: reverse_z_rh_infinite_projection(fov, aspect_ratio, near),
            _view: camera_transform.inverse(),
        }
    }
}

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

#[derive(Default)]
pub struct Camera {
    pub position: Vec3,
    pub orientation: Quat,
}

impl Camera {
    #[profiling::function]
    pub(crate) fn create_global_transforms(&self, width: f32, height: f32) -> GlobalTransforms {
        GlobalTransforms::new(Mat4::from_rotation_translation(self.orientation, self.position), width, height)
    }
}
