use crate::arena::VulkanArena;
use crate::debug_utils;
use crate::vulkan_raii::{AnyImage, Device, Framebuffer, ImageView, Surface, Swapchain};
use crate::{Error, PhysicalDevice, Pipelines};
use ash::extensions::khr;
use ash::{vk, Entry, Instance};
use std::rc::Rc;

struct SwapchainSettings {
    extent: vk::Extent2D,
    immediate_present: bool,
}

/// The shorter-lived half of the rendering pair, along with [Gpu].
///
/// This struct has the concrete rendering objects, like the render
/// passes, framebuffers, command buffers and so on.
pub struct Canvas {
    pub extent: vk::Extent2D,
    pub frame_count: u32,

    pub(crate) swapchain: Rc<Swapchain>,
    pub(crate) framebuffers: Vec<Framebuffer>,
}

impl Canvas {
    /// Creates a new Canvas. Should be recreated when the window size
    /// changes.
    ///
    /// The fallback width and height parameters are used when Vulkan
    /// can't get the window size, e.g. when creating a new window in
    /// Wayland (when the window size is specified by the size of the
    /// initial framebuffer).
    ///
    /// If `immediate_present` is true, the immediate present mode is
    /// used. Otherwise, FIFO. FIFO only releases frames after they've
    /// been displayed on screen, so it caps the fps to the screen's
    /// refresh rate.
    pub fn new(
        entry: &Entry,
        instance: &Instance,
        surface: &Rc<Surface>,
        device: &Rc<Device>,
        physical_device: &PhysicalDevice,
        pipelines: &Pipelines,
        old_canvas: Option<&Canvas>,
        fallback_width: u32,
        fallback_height: u32,
        immediate_present: bool,
    ) -> Result<Canvas, Error> {
        profiling::scope!("new_canvas");
        let surface_ext = khr::Surface::new(entry, instance);
        let swapchain_ext = khr::Swapchain::new(instance, &device.inner);
        let queue_family_indices = [
            physical_device.graphics_queue_family.index,
            physical_device.surface_queue_family.index,
        ];
        let (swapchain, extent) = create_swapchain(
            &surface_ext,
            &swapchain_ext,
            surface.inner,
            old_canvas.map(|r| r.swapchain.inner),
            physical_device,
            &queue_family_indices,
            &SwapchainSettings {
                extent: vk::Extent2D {
                    width: fallback_width,
                    height: fallback_height,
                },
                immediate_present,
            },
        )?;
        let swapchain_format = physical_device.swapchain_format;
        let vk::Extent2D { width, height } = extent;
        let swapchain_images = unsafe { swapchain_ext.get_swapchain_images(swapchain) }.map_err(Error::VulkanGetSwapchainImages)?;
        let frame_count = swapchain_images.len() as u32;
        debug_utils::name_vulkan_object(
            device,
            swapchain,
            format_args!("{width}x{height}, {swapchain_format:?}, {frame_count} frames"),
        );
        let swapchain = Rc::new(Swapchain {
            inner: swapchain,
            device: swapchain_ext,
            surface: surface.clone(),
        });

        // TODO(high): Split Canvas:
        // - [done] Into Pipelines+RenderPass. Pipelines' viewport needs to be made dynamic.
        // - Into Swapchain. (create_swapchain is probably enough.)
        // - Into Framebuffers.
        // Only the last two need to be recreated per resize, and framebuffers
        // should be destroyed before swapchains, and created after.
        let mut framebuffer_arena = VulkanArena::new(
            instance,
            device,
            physical_device.inner,
            1_000_000_000, // FIXME: query framebuffer memory requirements for arena
            vk::MemoryPropertyFlags::DEVICE_LOCAL | vk::MemoryPropertyFlags::LAZILY_ALLOCATED,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            format_args!("framebuffer arena ({width}x{height}, {swapchain_format:?}, {frame_count} frames)"),
        )?;

        let create_image_view = |aspect_mask: vk::ImageAspectFlags, format: vk::Format| {
            move |image: AnyImage| -> Result<ImageView, Error> {
                let subresource_range = vk::ImageSubresourceRange {
                    aspect_mask,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                };
                let image_view_create_info = vk::ImageViewCreateInfo {
                    image: image.inner(),
                    view_type: vk::ImageViewType::TYPE_2D,
                    format,
                    subresource_range,
                    ..Default::default()
                };
                let image_view =
                    unsafe { device.create_image_view(&image_view_create_info, None) }.map_err(Error::VulkanSwapchainImageViewCreation)?;
                Ok(ImageView {
                    inner: image_view,
                    device: device.clone(),
                    image: Rc::new(image),
                })
            }
        };

        let mut create_image = |format: vk::Format, usage: vk::ImageUsageFlags| {
            let image_create_info = vk::ImageCreateInfo::builder()
                .image_type(vk::ImageType::TYPE_2D)
                .format(format)
                .extent(vk::Extent3D { width, height, depth: 1 })
                .mip_levels(1)
                .array_layers(1)
                .samples(pipelines.attachment_sample_count)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(usage);
            framebuffer_arena.create_image(*image_create_info, format_args!("tbd"))
        };

        // TODO(med): Add another set of images to render to, to allow for post processing
        // Also, consider: render to a linear/higher depth image, then map to SRGB for the swapchain?
        let swapchain_image_views = swapchain_images
            .into_iter()
            .map(|image| AnyImage::Swapchain(image, swapchain.clone()))
            .map(create_image_view(vk::ImageAspectFlags::COLOR, swapchain_format))
            .collect::<Result<Vec<_>, _>>()?;

        let color_images = (0..swapchain_image_views.len())
            .map(|_| create_image(swapchain_format, vk::ImageUsageFlags::COLOR_ATTACHMENT))
            .collect::<Result<Vec<_>, Error>>()?;
        let color_image_views = color_images
            .into_iter()
            .map(AnyImage::from)
            .map(create_image_view(vk::ImageAspectFlags::COLOR, swapchain_format))
            .collect::<Result<Vec<_>, _>>()?;

        let depth_images = (0..swapchain_image_views.len())
            .map(|_| create_image(physical_device.depth_format, vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT))
            .collect::<Result<Vec<_>, Error>>()?;
        let depth_image_views = depth_images
            .into_iter()
            .map(AnyImage::from)
            .map(create_image_view(vk::ImageAspectFlags::DEPTH, physical_device.depth_format))
            .collect::<Result<Vec<_>, _>>()?;

        for (((i, sc), color), depth) in swapchain_image_views
            .iter()
            .enumerate()
            .zip(color_image_views.iter())
            .zip(depth_image_views.iter())
        {
            let nth = i + 1;
            debug_utils::name_vulkan_object(device, sc.inner, format_args!("swapchain (frame {nth}/{frame_count})"));
            debug_utils::name_vulkan_object(device, sc.image.inner(), format_args!("swapchain (frame {nth}/{frame_count})"));
            debug_utils::name_vulkan_object(device, color.inner, format_args!("color fb (frame {nth}/{frame_count})"));
            debug_utils::name_vulkan_object(device, color.image.inner(), format_args!("color fb (frame {nth}/{frame_count})"));
            debug_utils::name_vulkan_object(device, depth.inner, format_args!("depth fb (frame {nth}/{frame_count})"));
            debug_utils::name_vulkan_object(device, depth.image.inner(), format_args!("depth fb (frame {nth}/{frame_count})"));
        }

        let framebuffers = color_image_views
            .into_iter()
            .enumerate()
            .zip(depth_image_views.into_iter())
            .zip(swapchain_image_views.into_iter())
            .map(|(((i, color_image_view), depth_image_view), swapchain_image_view)| {
                let attachments = [color_image_view.inner, depth_image_view.inner, swapchain_image_view.inner];
                let framebuffer_create_info = vk::FramebufferCreateInfo::builder()
                    .render_pass(pipelines.render_pass.inner)
                    .attachments(&attachments)
                    .width(extent.width)
                    .height(extent.height)
                    .layers(1);
                let framebuffer =
                    unsafe { device.create_framebuffer(&framebuffer_create_info, None) }.map_err(Error::VulkanFramebufferCreation)?;
                debug_utils::name_vulkan_object(device, framebuffer, format_args!("main fb {}/{frame_count}", i + 1));
                Ok(Framebuffer {
                    inner: framebuffer,
                    device: device.clone(),
                    render_pass: pipelines.render_pass.clone(),
                    attachments: vec![color_image_view, depth_image_view, swapchain_image_view],
                })
            })
            .collect::<Result<Vec<Framebuffer>, Error>>()?;

        Ok(Canvas {
            extent,
            frame_count,
            swapchain,
            framebuffers,
        })
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
) -> Result<(vk::SwapchainKHR, vk::Extent2D), Error> {
    let present_modes = unsafe { surface_ext.get_physical_device_surface_present_modes(physical_device.inner, surface) }
        .map_err(Error::VulkanPhysicalDeviceSurfaceQuery)?;
    let mut present_mode = vk::PresentModeKHR::FIFO;
    if settings.immediate_present {
        // TODO(med): Remove immediate present, use proper gpu profiling instead.
        if present_modes.contains(&vk::PresentModeKHR::MAILBOX) {
            present_mode = vk::PresentModeKHR::MAILBOX;
        } else if present_modes.contains(&vk::PresentModeKHR::IMMEDIATE) {
            present_mode = vk::PresentModeKHR::IMMEDIATE;
        }
    }

    let surface_capabilities = unsafe {
        surface_ext
            .get_physical_device_surface_capabilities(physical_device.inner, surface)
            .map_err(Error::VulkanPhysicalDeviceSurfaceQuery)
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

    let mut swapchain_create_info = vk::SwapchainCreateInfoKHR::builder()
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
    let swapchain = unsafe { swapchain_ext.create_swapchain(&swapchain_create_info, None) }.map_err(Error::VulkanSwapchainCreation)?;

    Ok((swapchain, swapchain_create_info.image_extent))
}
