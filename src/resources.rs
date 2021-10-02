use crate::Error;
use ash::Device;
use ash::{version::DeviceV1_0, vk};
use std::sync::Mutex; // TODO: Make Resources mutable

// TODO: Add resource removal (via refcounting?)

pub(crate) struct AllocatedImage(pub(crate) vk::Image, pub(crate) vk::ImageView, pub(crate) vk_mem::Allocation);
pub(crate) struct AllocatedBuffer(pub(crate) vk::Buffer, pub(crate) vk_mem::Allocation);

struct LoadingImage {
    upload_fence: vk::Fence,
    staging_buffer: AllocatedBuffer,
    target_image: AllocatedImage,
}

struct LoadingBuffer {
    upload_fence: vk::Fence,
    staging_buffer: AllocatedBuffer,
    target_buffer: AllocatedBuffer,
}

/// Holds and cleans up buffers and images for [Gpu].
pub struct Resources {
    loading_images: Mutex<Vec<LoadingImage>>,
    loading_buffers: Mutex<Vec<LoadingBuffer>>,
    images: Mutex<Vec<AllocatedImage>>,
    buffers: Mutex<Vec<AllocatedBuffer>>,
}

impl Resources {
    #[profiling::function]
    pub(crate) fn new() -> Resources {
        Resources {
            loading_images: Mutex::new(Vec::new()),
            loading_buffers: Mutex::new(Vec::new()),
            images: Mutex::new(Vec::new()),
            buffers: Mutex::new(Vec::new()),
        }
    }

    /// Cleans up the staging memory for buffers and images which have
    /// been uploaded. Should be called ~every frame.
    #[profiling::function]
    pub(crate) fn clean_up_staging_memory(&self, device: &Device, allocator: &vk_mem::Allocator) -> Result<(), Error> {
        let mut loading_images = self.loading_images.lock().unwrap();
        let mut images = self.images.lock().unwrap();
        let loaded_images = loading_images
            .iter()
            .enumerate()
            .filter_map(|(i, loading_image)| {
                match unsafe {
                    device
                        .get_fence_status(loading_image.upload_fence)
                        .map_err(Error::VulkanFenceStatus)
                } {
                    Ok(true) => Some(Ok(i)),
                    Ok(false) => None,
                    Err(err) => Some(Err(err)),
                }
            })
            .collect::<Result<Vec<usize>, Error>>()?;
        for loaded_index in loaded_images.into_iter().rev() {
            let loading_image = loading_images.remove(loaded_index);
            unsafe { device.destroy_fence(loading_image.upload_fence, None) };
            destroy_buffer(allocator, loading_image.staging_buffer)?;
            images.push(loading_image.target_image);
        }

        let mut loading_buffers = self.loading_buffers.lock().unwrap();
        let mut buffers = self.buffers.lock().unwrap();
        let loaded_buffers = loading_buffers
            .iter()
            .enumerate()
            .filter_map(|(i, loading_buffer)| {
                match unsafe {
                    device
                        .get_fence_status(loading_buffer.upload_fence)
                        .map_err(Error::VulkanFenceStatus)
                } {
                    Ok(true) => Some(Ok(i)),
                    Ok(false) => None,
                    Err(err) => Some(Err(err)),
                }
            })
            .collect::<Result<Vec<usize>, Error>>()?;
        for loaded_index in loaded_buffers.into_iter().rev() {
            let loading_buffer = loading_buffers.remove(loaded_index);
            unsafe { device.destroy_fence(loading_buffer.upload_fence, None) };
            destroy_buffer(allocator, loading_buffer.staging_buffer)?;
            buffers.push(loading_buffer.target_buffer);
        }

        Ok(())
    }

    #[profiling::function]
    pub(crate) fn add_buffer(&self, upload_fence: vk::Fence, staging_buffer: Option<AllocatedBuffer>, target_buffer: AllocatedBuffer) {
        if let Some(staging_buffer) = staging_buffer {
            let mut buffers = self.loading_buffers.lock().unwrap();
            buffers.push(LoadingBuffer {
                upload_fence,
                staging_buffer,
                target_buffer,
            });
        } else {
            let mut buffers = self.buffers.lock().unwrap();
            buffers.push(target_buffer);
        }
    }

    #[profiling::function]
    pub(crate) fn add_image(&self, upload_fence: vk::Fence, staging_buffer: Option<AllocatedBuffer>, target_image: AllocatedImage) {
        if let Some(staging_buffer) = staging_buffer {
            let mut images = self.loading_images.lock().unwrap();
            images.push(LoadingImage {
                upload_fence,
                staging_buffer,
                target_image,
            });
        } else {
            let mut images = self.images.lock().unwrap();
            images.push(target_image);
        }
    }

    /// Cleans up all the Vulkan resources. Cannot be implemented in
    /// Drop, because [Gpu] needs to explicitly clean up the resources
    /// in its Drop.
    #[profiling::function]
    pub(crate) fn clean_up(&self, device: &Device, allocator: &vk_mem::Allocator) -> Result<(), Error> {
        let mut guard = self.loading_buffers.lock().unwrap();
        while let Some(LoadingBuffer {
            upload_fence,
            staging_buffer,
            target_buffer,
        }) = guard.pop()
        {
            unsafe { device.destroy_fence(upload_fence, None) };
            destroy_buffer(allocator, staging_buffer)?;
            destroy_buffer(allocator, target_buffer)?;
        }

        let mut guard = self.loading_images.lock().unwrap();
        while let Some(LoadingImage {
            upload_fence,
            staging_buffer,
            target_image,
        }) = guard.pop()
        {
            unsafe { device.destroy_fence(upload_fence, None) };
            destroy_buffer(allocator, staging_buffer)?;
            destroy_image(device, allocator, target_image)?;
        }

        let mut guard = self.buffers.lock().unwrap();
        while let Some(buffer) = guard.pop() {
            destroy_buffer(allocator, buffer)?;
        }

        let mut guard = self.images.lock().unwrap();
        while let Some(image) = guard.pop() {
            destroy_image(device, allocator, image)?;
        }

        Ok(())
    }
}

#[profiling::function]
fn destroy_buffer(allocator: &vk_mem::Allocator, buffer: AllocatedBuffer) -> Result<(), Error> {
    allocator.destroy_buffer(buffer.0, &buffer.1).map_err(Error::ResourceCleanup)
}

#[profiling::function]
fn destroy_image(device: &Device, allocator: &vk_mem::Allocator, image: AllocatedImage) -> Result<(), Error> {
    unsafe { device.destroy_image_view(image.1, None) };
    allocator.destroy_image(image.0, &image.2).map_err(Error::ResourceCleanup)
}
