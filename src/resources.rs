use crate::{Error, Gpu, Texture};
use ash::Device;
use ash::{version::DeviceV1_0, vk};
use std::cell::Cell;
use std::rc::{Rc, Weak};
use std::sync::Mutex; // TODO: Make Resources mutable

// TODO: Add resource removal (via refcounting?)

pub(crate) struct AllocatedImage(pub(crate) vk::Image, pub(crate) vk::ImageView, pub(crate) vk_mem::Allocation);
pub(crate) struct AllocatedBuffer(pub(crate) vk::Buffer, pub(crate) vk_mem::Allocation);

pub(crate) enum UploadStatus {
    Loading,
    Loaded,
    Destroyed,
}

pub(crate) type RefCountedStatus = Rc<Cell<UploadStatus>>;
pub(crate) type WeakRefCountedStatus = Weak<Cell<UploadStatus>>;

struct LoadingImage {
    upload_fence: vk::Fence,
    staging_buffer: AllocatedBuffer,
    target_image: AllocatedImage,
    refcounted_status: RefCountedStatus,
}

struct LoadingBuffer {
    upload_fence: vk::Fence,
    staging_buffer: AllocatedBuffer,
    target_buffer: AllocatedBuffer,
    refcounted_status: RefCountedStatus,
}

/// Holds and cleans up buffers and images for [Gpu].
pub struct Resources {
    loading_images: Mutex<Vec<LoadingImage>>,
    loading_buffers: Mutex<Vec<LoadingBuffer>>,
    images: Mutex<Vec<(RefCountedStatus, AllocatedImage)>>,
    buffers: Mutex<Vec<(RefCountedStatus, AllocatedBuffer)>>,
    texture_index_refs: Mutex<Vec<(WeakRefCountedStatus, u32)>>,
}

impl Resources {
    #[profiling::function]
    pub(crate) fn new() -> Resources {
        Resources {
            loading_images: Mutex::new(Vec::new()),
            loading_buffers: Mutex::new(Vec::new()),
            images: Mutex::new(Vec::new()),
            buffers: Mutex::new(Vec::new()),
            texture_index_refs: Mutex::new(Vec::new()),
        }
    }

    /// Cleans up the staging memory for buffers and images which have
    /// been uploaded. Should be called ~every frame.
    #[profiling::function]
    pub(crate) fn clean_up_unused_memory(&self, gpu: &Gpu) -> Result<(), Error> {
        let mut loading_images = self.loading_images.lock().unwrap();
        let mut images = self.images.lock().unwrap();
        let loaded_images = loading_images
            .iter()
            .enumerate()
            .filter_map(|(i, loading_image)| {
                match unsafe { gpu.device.get_fence_status(loading_image.upload_fence) }.map_err(Error::VulkanFenceStatus) {
                    Ok(true) => Some(Ok(i)),
                    Ok(false) => None,
                    Err(err) => Some(Err(err)),
                }
            })
            .collect::<Result<Vec<usize>, Error>>()?;
        for loaded_index in loaded_images.into_iter().rev() {
            let loading_image = loading_images.remove(loaded_index);
            unsafe { gpu.device.destroy_fence(loading_image.upload_fence, None) };
            destroy_buffer(&gpu.allocator, loading_image.staging_buffer)?;
            loading_image.refcounted_status.set(UploadStatus::Loaded);
            images.push((loading_image.refcounted_status, loading_image.target_image));
        }

        // This lint is wrong: images needs to be mutably borrowed
        // when iterating through the indices, so it can't be iterated
        // on at the same time.
        #[allow(clippy::needless_collect)]
        let dropped_images = images
            .iter()
            .enumerate()
            .filter_map(|(i, image)| if Rc::strong_count(&image.0) == 1 { Some(i) } else { None })
            .collect::<Vec<usize>>();
        for dropped_index in dropped_images.into_iter().rev() {
            let dropped_image = images.remove(dropped_index);
            dropped_image.0.set(UploadStatus::Destroyed);
            destroy_image(&gpu.device, &gpu.allocator, dropped_image.1)?;
        }

        let mut texture_index_refs = self.texture_index_refs.lock().unwrap();
        let mut dropped_indices = texture_index_refs
            .iter()
            .filter_map(|(texture_ref, i)| if Weak::strong_count(texture_ref) == 0 { Some(*i) } else { None })
            .collect::<Vec<u32>>();
        dropped_indices.sort_unstable();
        dropped_indices.dedup();
        texture_index_refs.retain(|(_, i)| !dropped_indices.contains(i));
        for dropped_index in dropped_indices {
            gpu.release_texture_index(dropped_index);
        }

        let mut loading_buffers = self.loading_buffers.lock().unwrap();
        let mut buffers = self.buffers.lock().unwrap();
        let loaded_buffers = loading_buffers
            .iter()
            .enumerate()
            .filter_map(|(i, loading_buffer)| {
                match unsafe { gpu.device.get_fence_status(loading_buffer.upload_fence) }.map_err(Error::VulkanFenceStatus) {
                    Ok(true) => Some(Ok(i)),
                    Ok(false) => None,
                    Err(err) => Some(Err(err)),
                }
            })
            .collect::<Result<Vec<usize>, Error>>()?;
        for loaded_index in loaded_buffers.into_iter().rev() {
            let loading_buffer = loading_buffers.remove(loaded_index);
            unsafe { gpu.device.destroy_fence(loading_buffer.upload_fence, None) };
            destroy_buffer(&gpu.allocator, loading_buffer.staging_buffer)?;
            loading_buffer.refcounted_status.set(UploadStatus::Loaded);
            buffers.push((loading_buffer.refcounted_status, loading_buffer.target_buffer));
        }

        // This lint is wrong: images needs to be mutably borrowed
        // when iterating through the indices, so it can't be iterated
        // on at the same time.
        #[allow(clippy::needless_collect)]
        let dropped_buffers = buffers
            .iter()
            .enumerate()
            .filter_map(|(i, buffer)| if Rc::strong_count(&buffer.0) == 1 { Some(i) } else { None })
            .collect::<Vec<usize>>();
        for dropped_index in dropped_buffers.into_iter().rev() {
            let dropped_buffer = buffers.remove(dropped_index);
            dropped_buffer.0.set(UploadStatus::Destroyed);
            destroy_buffer(&gpu.allocator, dropped_buffer.1)?;
        }

        Ok(())
    }

    #[profiling::function]
    pub(crate) fn add_buffer(
        &self,
        upload_fence: vk::Fence,
        staging_buffer: Option<AllocatedBuffer>,
        target_buffer: AllocatedBuffer,
    ) -> RefCountedStatus {
        let refcounted_status = Rc::new(Cell::new(UploadStatus::Loading));
        if let Some(staging_buffer) = staging_buffer {
            let mut buffers = self.loading_buffers.lock().unwrap();
            buffers.push(LoadingBuffer {
                upload_fence,
                staging_buffer,
                target_buffer,
                refcounted_status: refcounted_status.clone(),
            });
        } else {
            let mut buffers = self.buffers.lock().unwrap();
            buffers.push((refcounted_status.clone(), target_buffer));
        }
        refcounted_status
    }

    #[profiling::function]
    pub(crate) fn add_image(
        &self,
        upload_fence: vk::Fence,
        staging_buffer: Option<AllocatedBuffer>,
        target_image: AllocatedImage,
    ) -> RefCountedStatus {
        let refcounted_status = Rc::new(Cell::new(UploadStatus::Loading));
        if let Some(staging_buffer) = staging_buffer {
            let mut images = self.loading_images.lock().unwrap();
            images.push(LoadingImage {
                upload_fence,
                staging_buffer,
                target_image,
                refcounted_status: refcounted_status.clone(),
            });
        } else {
            let mut images = self.images.lock().unwrap();
            images.push((refcounted_status.clone(), target_image));
        }
        refcounted_status
    }

    pub(crate) fn add_texture_index_ref(&self, texture: &Texture, index: u32) {
        let tex_ref = texture.create_weak_ref();
        let mut guard = self.texture_index_refs.lock().unwrap();
        guard.push((tex_ref, index));
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
            refcounted_status,
        }) = guard.pop()
        {
            refcounted_status.set(UploadStatus::Destroyed);
            unsafe { device.destroy_fence(upload_fence, None) };
            destroy_buffer(allocator, staging_buffer)?;
            destroy_buffer(allocator, target_buffer)?;
        }

        let mut guard = self.loading_images.lock().unwrap();
        while let Some(LoadingImage {
            upload_fence,
            staging_buffer,
            target_image,
            refcounted_status,
        }) = guard.pop()
        {
            refcounted_status.set(UploadStatus::Destroyed);
            unsafe { device.destroy_fence(upload_fence, None) };
            destroy_buffer(allocator, staging_buffer)?;
            destroy_image(device, allocator, target_image)?;
        }

        let mut guard = self.buffers.lock().unwrap();
        while let Some((refcounted_status, buffer)) = guard.pop() {
            refcounted_status.set(UploadStatus::Destroyed);
            destroy_buffer(allocator, buffer)?;
        }

        let mut guard = self.images.lock().unwrap();
        while let Some((refcounted_status, image)) = guard.pop() {
            refcounted_status.set(UploadStatus::Destroyed);
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
