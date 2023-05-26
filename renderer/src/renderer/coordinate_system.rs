use glam::{Mat4, Vec3, Vec4};

pub struct CoordinateSystem {
    pub up: Vec3,
    pub right: Vec3,
    pub forward: Vec3,
}

impl CoordinateSystem {
    pub const VULKAN: CoordinateSystem = CoordinateSystem {
        up: Vec3::NEG_Y,
        right: Vec3::X,
        forward: Vec3::Z,
    };

    pub const QUAKE: CoordinateSystem = CoordinateSystem {
        up: Vec3::Z,
        right: Vec3::NEG_Y,
        forward: Vec3::X,
    };

    pub const GLTF: CoordinateSystem = CoordinateSystem {
        up: Vec3::Y,
        right: Vec3::NEG_X,
        forward: Vec3::Z,
    };

    /// Creates a transformation matrix which will transform vectors from this
    /// coordinate system to the target_space coordinate system.
    pub fn create_transform_to(&self, target_space: &CoordinateSystem) -> Mat4 {
        let self_from_ruf = Mat4::from_cols(
            Vec4::from((self.right, 0.0)),
            Vec4::from((self.up, 0.0)),
            Vec4::from((self.forward, 0.0)),
            Vec4::W,
        );
        let ruf_from_self = self_from_ruf.inverse();
        let target_from_ruf = Mat4::from_cols(
            Vec4::from((target_space.right, 0.0)),
            Vec4::from((target_space.up, 0.0)),
            Vec4::from((target_space.forward, 0.0)),
            Vec4::W,
        );
        target_from_ruf * ruf_from_self
    }
}
