use crate::{Error, Gpu};
use ash::extensions::khr;
use ash::version::DeviceV1_0;
use ash::vk;

pub const SWAPCHAIN_FORMAT: vk::Format = vk::Format::B8G8R8A8_SRGB;

/// The shorter-lived half of the rendering pair, along with [Gpu].
///
/// This struct has the concrete rendering objects, like the render
/// passes, framebuffers, command buffers and so on.
pub struct Canvas<'a> {
    /// Held by [Canvas] to ensure that the swapchain and command
    /// buffers are dropped before the device.
    pub gpu: &'a Gpu<'a>,

    pub extent: vk::Extent2D,

    pub(crate) swapchain: vk::SwapchainKHR,
    pub(crate) swapchain_image_views: Vec<vk::ImageView>,
    pub(crate) swapchain_framebuffers: Vec<vk::Framebuffer>,
}

impl Drop for Canvas<'_> {
    fn drop(&mut self) {
        let device = &self.gpu.device;

        for &framebuffer in &self.swapchain_framebuffers {
            unsafe { device.destroy_framebuffer(framebuffer, None) };
        }

        for &image_view in &self.swapchain_image_views {
            unsafe { device.destroy_image_view(image_view, None) };
        }

        unsafe {
            self.gpu
                .swapchain_ext
                .destroy_swapchain(self.swapchain, None)
        };
    }
}

impl Canvas<'_> {
    /// Creates a new Canvas. Should be recreated when the window size
    /// changes.
    ///
    /// The fallback width and height parameters are used when Vulkan
    /// can't get the window size, e.g. when creating a new window in
    /// Wayland (when the window size is specified by the size of the
    /// initial framebuffer).
    pub fn new<'a>(
        gpu: &'a Gpu,
        old_canvas: Option<&Canvas>,
        fallback_width: u32,
        fallback_height: u32,
    ) -> Result<Canvas<'a>, Error> {
        let device = &gpu.device;
        let swapchain_ext = &gpu.swapchain_ext;
        let queue_family_indices = [gpu.graphics_family_index, gpu.surface_family_index];
        let (swapchain, swapchain_format, extent) = create_swapchain(
            &gpu.surface_ext,
            &swapchain_ext,
            gpu.driver.surface,
            vk::Extent2D {
                width: fallback_width,
                height: fallback_height,
            },
            old_canvas.map(|r| r.swapchain),
            gpu.physical_device,
            &queue_family_indices,
        )?;

        let swapchain_images = unsafe { swapchain_ext.get_swapchain_images(swapchain) }
            .map_err(Error::VulkanGetSwapchainImages)?;
        let swapchain_image_views = swapchain_images
            .into_iter()
            .map(|image| {
                let subresource_range = vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                };
                let image_view_create_info = vk::ImageViewCreateInfo {
                    image,
                    view_type: vk::ImageViewType::TYPE_2D,
                    format: swapchain_format,
                    subresource_range,
                    ..Default::default()
                };
                unsafe { device.create_image_view(&image_view_create_info, None) }
                    .map_err(Error::VulkanSwapchainImageViewCreation)
            })
            .collect::<Result<Vec<_>, _>>()?;

        let swapchain_framebuffers = swapchain_image_views
            .iter()
            .map(|image_view| {
                let attachments = [*image_view];
                let framebuffer_create_info = vk::FramebufferCreateInfo::builder()
                    .render_pass(gpu.final_render_pass)
                    .attachments(&attachments)
                    .width(extent.width)
                    .height(extent.height)
                    .layers(1);
                unsafe { device.create_framebuffer(&framebuffer_create_info, None) }
                    .map_err(Error::VulkanFramebufferCreation)
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Canvas {
            gpu,
            extent,
            swapchain,
            swapchain_image_views,
            swapchain_framebuffers,
        })
    }
}

fn create_swapchain(
    surface_ext: &khr::Surface,
    swapchain_ext: &khr::Swapchain,
    surface: vk::SurfaceKHR,
    extent: vk::Extent2D,
    old_swapchain: Option<vk::SwapchainKHR>,
    physical_device: vk::PhysicalDevice,
    queue_family_indices: &[u32],
) -> Result<(vk::SwapchainKHR, vk::Format, vk::Extent2D), Error> {
    let present_mode = vk::PresentModeKHR::FIFO;
    let min_image_count = 2;

    let (image_format, image_color_space) = {
        let surface_formats = unsafe {
            surface_ext
                .get_physical_device_surface_formats(physical_device, surface)
                .map_err(Error::VulkanPhysicalDeviceSurfaceQuery)
        }?;
        let color_space = if let Some(format) = surface_formats
            .iter()
            .find(|format| format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR)
        {
            format.color_space
        } else {
            surface_formats[0].color_space
        };
        (SWAPCHAIN_FORMAT, color_space)
    };

    let get_image_extent = || -> Result<vk::Extent2D, Error> {
        let surface_capabilities = unsafe {
            surface_ext
                .get_physical_device_surface_capabilities(physical_device, surface)
                .map_err(Error::VulkanPhysicalDeviceSurfaceQuery)
        }?;
        let unset_extent = vk::Extent2D {
            width: u32::MAX,
            height: u32::MAX,
        };
        let image_extent = if surface_capabilities.current_extent != unset_extent {
            surface_capabilities.current_extent
        } else {
            extent
        };
        Ok(image_extent)
    };

    let mut swapchain_create_info = vk::SwapchainCreateInfoKHR {
        surface,
        min_image_count,
        image_format,
        image_color_space,
        image_array_layers: 1,
        image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
        pre_transform: vk::SurfaceTransformFlagsKHR::IDENTITY,
        composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
        present_mode,
        clipped: vk::TRUE,
        ..Default::default()
    };
    if queue_family_indices.windows(2).any(|indices| {
        if let [a, b] = *indices {
            a != b
        } else {
            unreachable!()
        }
    }) {
        swapchain_create_info.image_sharing_mode = vk::SharingMode::CONCURRENT;
        swapchain_create_info.queue_family_index_count = queue_family_indices.len() as u32;
        swapchain_create_info.p_queue_family_indices = queue_family_indices.as_ptr();
    } else {
        swapchain_create_info.image_sharing_mode = vk::SharingMode::EXCLUSIVE;
    }
    if let Some(old_swapchain) = old_swapchain {
        swapchain_create_info.old_swapchain = old_swapchain;
    }
    // Get image extent at the latest possible time to avoid getting
    // an outdated extent. Bummer that this is an issue, but I haven't
    // found a good way to avoid it.
    swapchain_create_info.image_extent = get_image_extent()?;
    let swapchain = unsafe { swapchain_ext.create_swapchain(&swapchain_create_info, None) }
        .map_err(Error::VulkanSwapchainCreation)?;

    Ok((swapchain, image_format, swapchain_create_info.image_extent))
}
