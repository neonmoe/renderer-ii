use crate::Error;
use ash::extensions::khr;
use ash::version::DeviceV1_0;
use ash::vk;
use ash::{Device, Entry, Instance};

pub const SWAPCHAIN_FORMAT: vk::Format = vk::Format::B8G8R8A8_SRGB;

pub struct SwapchainSettings<'a> {
    pub old_swapchain: Option<&'a Swapchain<'a>>,
    pub queue_family_indices: &'a [u32],
    pub extent: vk::Extent2D,
    pub immediate_present: bool,
}

/// Just contains the Vulkan swapchain and imageviews to its images.
///
/// On resize:
/// 1. Drop [RenderPass](crate::RenderPass),
/// 2. Re-create [Swapchain],
/// 3. Reset the framebuffers' [VulkanArena](crate::VulkanArena),
/// 4. Re-create [RenderPass](crate::RenderPass).
pub struct Swapchain<'a> {
    pub extent: vk::Extent2D,
    pub format: vk::Format,
    /// Amount of swapchain images. 2 for double-buffering, 3 for
    /// triple-buffering.
    pub queued_images: usize,

    // TODO: Make simple new-drop wrappers for the Vulkan objects used
    // in Swapchain and RenderPass, then remove their Drop
    // impls. Seems that the lifetime issues are caused by the manual
    // Drop impl.
    pub swapchain: vk::SwapchainKHR,
    pub swapchain_image_views: Vec<vk::ImageView>,

    device: &'a Device,
    swapchain_ext: khr::Swapchain,
}

impl Drop for Swapchain<'_> {
    fn drop(&mut self) {
        for &image_view in &self.swapchain_image_views {
            profiling::scope!("destroy swapchain image view");
            unsafe { self.device.destroy_image_view(image_view, None) };
        }
        {
            profiling::scope!("destroy swapchain");
            unsafe { self.swapchain_ext.destroy_swapchain(self.swapchain, None) };
        }
    }
}

impl Swapchain<'_> {
    pub fn new<'a>(
        entry: &'a Entry,
        instance: &'a Instance,
        device: &'a Device,
        // TODO: PhysicalDevice with lifetime
        physical_device: vk::PhysicalDevice,
        // TODO: Surface with lifetime
        surface: vk::SurfaceKHR,
        settings: SwapchainSettings,
    ) -> Result<Swapchain<'a>, Error> {
        profiling::scope!("new_swapchain");

        let surface_ext = khr::Surface::new(entry, instance);
        let swapchain_ext = khr::Swapchain::new(instance, device);

        // NOTE: The following combinations should be presented as a config option:
        // - FIFO + 2 (traditional double-buffered vsync)
        //   - no tearing, good latency, bad for perf when running under refresh rate
        // - FIFO + 3 (like double-buffering, but longer queue)
        //   - no tearing, bad latency, no perf issues when running under refresh rate
        // - MAILBOX + 3 (render-constantly, discard frames when waiting for vsync)
        //   - no tearing, great latency, optimal choice when available
        // - IMMEDIATE + 2 (render-constantly, ignore vsync (probably causes tearing))
        //   - possible tearing, best latency
        // With the non-available ones grayed out, of course.
        let present_mode = if settings.immediate_present {
            vk::PresentModeKHR::IMMEDIATE
        } else {
            vk::PresentModeKHR::FIFO
        };
        let min_swapchain_images = 3;

        let (format, image_color_space) = {
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

        // NOTE: There's a bit of a race condition here, because the
        // surface extents are queried here, but the swapchain is
        // created a bit later. We expect that the window is not
        // resized between here and vkCreateSwapchainKHR. If it is
        // resized during that time, behaviour is
        // platform-dependent. If the platform decides to create the
        // swapchain with a different extent than this, issues may
        // arise, as the rest of the code expects this extent to be
        // the correct one.
        //
        // See imageExtent here: https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VkSwapchainCreateInfoKHR.html
        let extent = {
            let surface_capabilities = unsafe {
                surface_ext
                    .get_physical_device_surface_capabilities(physical_device, surface)
                    .map_err(Error::VulkanPhysicalDeviceSurfaceQuery)
            }?;
            let unset_extent = vk::Extent2D {
                width: u32::MAX,
                height: u32::MAX,
            };
            if surface_capabilities.current_extent != unset_extent {
                surface_capabilities.current_extent
            } else {
                settings.extent
            }
        };

        let mut swapchain_create_info = vk::SwapchainCreateInfoKHR::builder()
            .surface(surface)
            .min_image_count(min_swapchain_images)
            .image_format(format)
            .image_color_space(image_color_space)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true)
            .image_extent(extent);
        if settings
            .queue_family_indices
            .windows(2)
            .any(|indices| if let [a, b] = *indices { a != b } else { unreachable!() })
        {
            swapchain_create_info = swapchain_create_info
                .image_sharing_mode(vk::SharingMode::CONCURRENT)
                .queue_family_indices(settings.queue_family_indices);
        } else {
            swapchain_create_info = swapchain_create_info.image_sharing_mode(vk::SharingMode::EXCLUSIVE);
        }
        if let Some(old_swapchain) = settings.old_swapchain {
            swapchain_create_info = swapchain_create_info.old_swapchain(old_swapchain.swapchain);
        }
        let swapchain = unsafe { swapchain_ext.create_swapchain(&swapchain_create_info, None) }.map_err(Error::VulkanSwapchainCreation)?;

        // TODO: Add another set of images to render to, to allow for post processing
        // Also, consider: render to a linear/higher depth image, then map to SRGB for the swapchain?
        let swapchain_images = unsafe { swapchain_ext.get_swapchain_images(swapchain) }.map_err(Error::VulkanGetSwapchainImages)?;
        let queued_images = swapchain_images.len();
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
                    format,
                    subresource_range,
                    ..Default::default()
                };
                unsafe { device.create_image_view(&image_view_create_info, None) }.map_err(Error::VulkanSwapchainImageViewCreation)
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Swapchain {
            extent,
            format,
            queued_images,
            swapchain,
            swapchain_image_views,
            device,
            swapchain_ext,
        })
    }
}
