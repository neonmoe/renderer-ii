use crate::physical_device::PhysicalDevice;
use crate::vulkan_raii::{self, AnyImage, Device, Surface};
use alloc::rc::Rc;
use arrayvec::ArrayVec;
use ash::extensions::khr;
use ash::vk;

#[derive(thiserror::Error, Debug)]
pub enum SwapchainError {
    #[error("the old swapchain is still in use")]
    OldSwapchainInUse,
    #[error("failed to get swapchain images")]
    GetSwapchainImages(#[source] vk::Result),
    #[error("failed to get present modes supported by the physical device")]
    GetPhysicalDevicePresentModes(#[source] vk::Result),
    #[error("failed to get surface capabilities (for resolution) from the physical device")]
    GetPhysicalDeviceSurfaceCapabilities(#[source] vk::Result),
    #[error("failed to create swapchain (window issues?)")]
    SwapchainCreation(#[source] vk::Result),
}

pub struct SwapchainSettings {
    pub extent: vk::Extent2D,
    pub immediate_present: bool,
}

pub enum SwapchainBase {
    Surface(Surface),
    OldSwapchain(Swapchain),
}

pub struct Swapchain {
    pub extent: vk::Extent2D,
    pub(crate) images: ArrayVec<Rc<AnyImage>, 8>,
    swapchain: Rc<vulkan_raii::Swapchain>,
}

impl Swapchain {
    pub fn new(
        device: &Device,
        physical_device: &PhysicalDevice,
        surface: Surface,
        settings: &SwapchainSettings,
    ) -> Result<Swapchain, SwapchainError> {
        profiling::scope!("swapchain creation");

        let queue_family_indices = [
            physical_device.graphics_queue_family.index,
            physical_device.surface_queue_family.index,
        ];
        let (swapchain, extent) = create_swapchain(
            &device.surface,
            &device.swapchain,
            surface.inner,
            None,
            physical_device,
            &queue_family_indices,
            settings,
        )?;
        let swapchain = Rc::new(vulkan_raii::Swapchain {
            inner: swapchain,
            device: device.swapchain.clone(),
            surface,
        });

        let images = unsafe { device.swapchain.get_swapchain_images(swapchain.inner) }
            .map_err(SwapchainError::GetSwapchainImages)?
            .into_iter()
            .map(|image| Rc::new(AnyImage::Swapchain(image, swapchain.clone())))
            .collect::<ArrayVec<_, 8>>();

        let vk::Extent2D { width, height } = extent;
        let swapchain_format = physical_device.swapchain_format;
        let frame_count = images.len() as u32;
        crate::name_vulkan_object(
            device,
            swapchain.inner,
            format_args!("{width}x{height}, {swapchain_format:?}, {frame_count} frames"),
        );

        Ok(Swapchain { extent, images, swapchain })
    }

    pub fn recreate(
        &mut self,
        device: &Device,
        physical_device: &PhysicalDevice,
        settings: &SwapchainSettings,
    ) -> Result<(), SwapchainError> {
        profiling::scope!("swapchain re-creation");

        self.images.clear();
        let swapchain_holder = if let Some(swapchain) = Rc::get_mut(&mut self.swapchain) {
            swapchain
        } else {
            return Err(SwapchainError::OldSwapchainInUse);
        };
        let queue_family_indices = [
            physical_device.graphics_queue_family.index,
            physical_device.surface_queue_family.index,
        ];
        let (new_swapchain, extent) = create_swapchain(
            &device.surface,
            &device.swapchain,
            swapchain_holder.surface.inner,
            Some(swapchain_holder.inner),
            physical_device,
            &queue_family_indices,
            settings,
        )?;
        // The mutable borrow of self.inner ensures that this won't leave any dangling swapchains.
        unsafe { device.swapchain.destroy_swapchain(swapchain_holder.inner, None) };
        // And now swapchain.inner is a valid swapchain again.
        swapchain_holder.inner = new_swapchain;
        self.extent = extent;

        self.images.extend(
            unsafe { device.swapchain.get_swapchain_images(self.swapchain.inner) }
                .map_err(SwapchainError::GetSwapchainImages)?
                .into_iter()
                .map(|image| Rc::new(AnyImage::Swapchain(image, self.swapchain.clone()))),
        );

        let vk::Extent2D { width, height } = extent;
        let swapchain_format = physical_device.swapchain_format;
        let frame_count = self.images.len() as u32;
        crate::name_vulkan_object(
            device,
            self.swapchain.inner,
            format_args!("{width}x{height}, {swapchain_format:?}, {frame_count} frames"),
        );
        Ok(())
    }

    pub fn frame_count(&self) -> u32 {
        self.images.len() as u32
    }

    /// Returns true if all the Rc's owned by this Swapchain would be dropped if
    /// this Swapchain was dropped. This should be asserted before dropping when
    /// the swapchain is supposed to be properly disposed of.
    ///
    /// Generally, this will return false if the Framebuffers created from this
    /// Swapchain are not dropped yet.
    pub fn no_external_refs(&self) -> bool {
        Rc::strong_count(&self.swapchain) == self.images.len() + 1 && self.images.iter().all(|image| Rc::strong_count(image) == 1)
    }

    pub(crate) fn inner(&self) -> vk::SwapchainKHR {
        self.swapchain.inner
    }

    pub(crate) fn device(&self) -> &khr::Swapchain {
        &self.swapchain.device
    }
}

#[profiling::function]
fn create_swapchain(
    surface_ext: &khr::Surface,
    swapchain_ext: &khr::Swapchain,
    surface: vk::SurfaceKHR,
    old_swapchain: Option<vk::SwapchainKHR>,
    physical_device: &PhysicalDevice,
    queue_family_indices: &[u32],
    settings: &SwapchainSettings,
) -> Result<(vk::SwapchainKHR, vk::Extent2D), SwapchainError> {
    let present_modes = unsafe { surface_ext.get_physical_device_surface_present_modes(physical_device.inner, surface) }
        .map_err(SwapchainError::GetPhysicalDevicePresentModes)?;
    let mut present_mode = vk::PresentModeKHR::FIFO;
    if settings.immediate_present {
        if present_modes.contains(&vk::PresentModeKHR::MAILBOX) {
            present_mode = vk::PresentModeKHR::MAILBOX;
        } else if present_modes.contains(&vk::PresentModeKHR::IMMEDIATE) {
            present_mode = vk::PresentModeKHR::IMMEDIATE;
        }
    }

    let surface_capabilities = unsafe {
        surface_ext
            .get_physical_device_surface_capabilities(physical_device.inner, surface)
            .map_err(SwapchainError::GetPhysicalDeviceSurfaceCapabilities)
    }?;
    let unset_extent = vk::Extent2D {
        width: u32::MAX,
        height: u32::MAX,
    };
    let image_extent = if surface_capabilities.current_extent != unset_extent {
        surface_capabilities.current_extent
    } else {
        settings.extent
    };
    let mut min_image_count = 2.max(surface_capabilities.min_image_count);
    if surface_capabilities.max_image_count > 0 {
        min_image_count = min_image_count.min(surface_capabilities.max_image_count)
    }

    let mut swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
        .surface(surface)
        .min_image_count(min_image_count)
        .image_format(physical_device.swapchain_format)
        .image_color_space(physical_device.swapchain_color_space)
        .image_array_layers(1)
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
        .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
        .present_mode(present_mode)
        .clipped(true)
        .image_extent(image_extent);
    if queue_family_indices[0] != queue_family_indices[1] {
        swapchain_create_info = swapchain_create_info
            .image_sharing_mode(vk::SharingMode::CONCURRENT)
            .queue_family_indices(queue_family_indices);
    } else {
        swapchain_create_info = swapchain_create_info.image_sharing_mode(vk::SharingMode::EXCLUSIVE);
    }
    if let Some(old_swapchain) = old_swapchain {
        swapchain_create_info = swapchain_create_info.old_swapchain(old_swapchain);
    }
    let swapchain = unsafe { swapchain_ext.create_swapchain(&swapchain_create_info, None) }.map_err(SwapchainError::SwapchainCreation)?;

    Ok((swapchain, swapchain_create_info.image_extent))
}
