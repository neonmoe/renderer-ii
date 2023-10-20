use std::fs;
use std::path::Path;

use renderer::image_loading::{self, ntex, ImageLoadingError, TextureKind};
use renderer::{vk, ForBuffers, ForImages, VulkanArenaMeasurementError, VulkanArenaMeasurer};

use crate::{gltf_json, GltfLoadingError};

#[derive(thiserror::Error, Debug)]
pub enum GltfMemoryMeasurementError {
    #[error("failed to read gltf file")]
    GltfLoading(#[source] GltfLoadingError),
    #[error("failed to measure memory requirements")]
    VulkanArenaMeasurement(#[source] VulkanArenaMeasurementError),
    #[error("failed to load image for memory requirement query")]
    ImageLoading(#[source] ImageLoadingError),
}

pub fn measure_gltf_memory_usage(
    measurement_arenas: (&mut VulkanArenaMeasurer<ForBuffers>, &mut VulkanArenaMeasurer<ForImages>),
    gltf_path: &Path,
    resource_path: &Path,
) -> Result<(), GltfMemoryMeasurementError> {
    profiling::scope!("measure gltf memory requirements");
    let gltf: String = {
        profiling::scope!("load gltf file");
        fs::read_to_string(gltf_path)
            .map_err(GltfLoadingError::MissingFile)
            .map_err(GltfMemoryMeasurementError::GltfLoading)?
    };
    let gltf: gltf_json::GltfJson = {
        profiling::scope!("parse gltf");
        serde_json::from_str(&gltf)
            .map_err(GltfLoadingError::JsonDeserialization)
            .map_err(GltfMemoryMeasurementError::GltfLoading)?
    };
    measure(measurement_arenas, gltf, None, resource_path)
}

pub fn measure_glb_memory_usage(
    measurement_arenas: (&mut VulkanArenaMeasurer<ForBuffers>, &mut VulkanArenaMeasurer<ForImages>),
    glb_path: &Path,
    resource_path: &Path,
) -> Result<(), GltfMemoryMeasurementError> {
    profiling::scope!("measure glb memory requirements");
    let glb = {
        profiling::scope!("load glb file");
        fs::read(glb_path)
            .map_err(GltfLoadingError::MissingFile)
            .map_err(GltfMemoryMeasurementError::GltfLoading)?
    };
    let (json, buffer) = {
        profiling::scope!("parse glb chunks");
        crate::read_glb_json_and_buffer(&glb).map_err(GltfMemoryMeasurementError::GltfLoading)?
    };
    let gltf: gltf_json::GltfJson = {
        profiling::scope!("parse gltf");
        serde_json::from_str(json)
            .map_err(GltfLoadingError::JsonDeserialization)
            .map_err(GltfMemoryMeasurementError::GltfLoading)?
    };
    measure(measurement_arenas, gltf, Some(buffer), resource_path)
}

fn measure(
    (buffer_measurer, image_measurer): (&mut VulkanArenaMeasurer<ForBuffers>, &mut VulkanArenaMeasurer<ForImages>),
    gltf: gltf_json::GltfJson,
    bin_buffer: Option<&[u8]>,
    resource_path: &Path,
) -> Result<(), GltfMemoryMeasurementError> {
    for buffer in &gltf.buffers {
        profiling::scope!("measure buffer mem reqs");
        buffer_measurer
            .add_buffer(crate::get_mesh_buffer_create_info(buffer.byte_length as vk::DeviceSize))
            .map_err(GltfMemoryMeasurementError::VulkanArenaMeasurement)?;
    }

    let mut memmap_holder = None;
    let image_texture_kinds = crate::get_gltf_texture_kinds(&gltf).map_err(GltfMemoryMeasurementError::GltfLoading)?;
    for (i, image) in gltf.images.iter().enumerate() {
        profiling::scope!("measure image mem reqs");
        let bytes = crate::load_image_bytes(&mut memmap_holder, resource_path, bin_buffer, image, &gltf)
            .map_err(GltfMemoryMeasurementError::GltfLoading)?;
        let kind = image_texture_kinds.get(&i).copied().unwrap_or(TextureKind::LinearColor);
        let name = image.uri.as_deref().unwrap_or("glb binary buffer");
        let image_data = ntex::decode(bytes)
            .map_err(|err| GltfLoadingError::NtexDecoding(err, name.to_string()))
            .map_err(GltfMemoryMeasurementError::GltfLoading)?;
        let image_create_info =
            image_loading::get_image_data_create_info(&image_data, kind).map_err(GltfMemoryMeasurementError::ImageLoading)?;
        image_measurer
            .add_image(image_create_info)
            .map_err(GltfMemoryMeasurementError::VulkanArenaMeasurement)?;
    }

    Ok(())
}
