//! Wrappers for Vulkan objects, for ensuring proper lifetimes and
//! destroying the objects in the wrappers' Drop impl.

use crate::arena::ImageAllocation;
use crate::Error;
use ash::version::DeviceV1_0;
use ash::vk;
use ash::Device;
use std::ops::Deref;

pub use framebuffer::VkFramebuffer;
pub use image_view::VkImageView;
pub use pipelines::VkPipelines;
pub use render_pass::VkRenderPass;
pub use swapchain::VkSwapchain;

mod image_view {
    use super::*;

    pub struct VkImageView<'a> {
        image_view: vk::ImageView,
        device: &'a Device,
    }

    impl VkImageView<'_> {
        /// Wrapper for [Device::create_image_view]. The ImageAllocation
        /// must be the same one as the create info is using, since the
        /// borrow lifetime ensures the image view's validity throughout
        /// its existence.
        pub unsafe fn new<'a>(
            device: &'a Device,
            _image: Option<&'a ImageAllocation>,
            create_info: &vk::ImageViewCreateInfo,
        ) -> Result<VkImageView<'a>, Error> {
            let image_view = device
                .create_image_view(create_info, None)
                .map_err(Error::VulkanImageViewCreation)?;
            Ok(VkImageView { device, image_view })
        }
    }

    impl Deref for VkImageView<'_> {
        type Target = vk::ImageView;
        fn deref(&self) -> &vk::ImageView {
            &self.image_view
        }
    }

    impl Drop for VkImageView<'_> {
        fn drop(&mut self) {
            unsafe { self.device.destroy_image_view(self.image_view, None) };
        }
    }
}

mod swapchain {
    use super::*;
    use ash::extensions::khr;
    use ash::Instance;

    pub struct VkSwapchain<'a> {
        #[allow(dead_code)]
        /// Held by the struct just to ensure the validity of the
        /// swapchain.
        device: &'a Device,
        swapchain_ext: khr::Swapchain,
        swapchain: vk::SwapchainKHR,
    }

    impl VkSwapchain<'_> {
        /// Wrapper for [khr::Swapchain::create_swapchain].
        pub unsafe fn new<'a>(
            instance: &'a Instance,
            device: &'a Device,
            create_info: &vk::SwapchainCreateInfoKHR,
        ) -> Result<VkSwapchain<'a>, Error> {
            let swapchain_ext = khr::Swapchain::new(instance, device);
            let swapchain = swapchain_ext
                .create_swapchain(create_info, None)
                .map_err(Error::VulkanSwapchainCreation)?;
            Ok(VkSwapchain {
                device,
                swapchain_ext,
                swapchain,
            })
        }
    }

    impl Deref for VkSwapchain<'_> {
        type Target = vk::SwapchainKHR;
        fn deref(&self) -> &vk::SwapchainKHR {
            &self.swapchain
        }
    }

    impl Drop for VkSwapchain<'_> {
        fn drop(&mut self) {
            unsafe { self.swapchain_ext.destroy_swapchain(self.swapchain, None) };
        }
    }
}

mod framebuffer {
    use super::*;

    pub struct VkFramebuffer<'a> {
        device: &'a Device,
        framebuffer: vk::Framebuffer,
    }

    impl VkFramebuffer<'_> {
        /// Wrapper for [Device::create_framebuffer].
        pub unsafe fn new<'a>(device: &'a Device, create_info: &vk::FramebufferCreateInfo) -> Result<VkFramebuffer<'a>, Error> {
            let framebuffer = device
                .create_framebuffer(create_info, None)
                .map_err(Error::VulkanFramebufferCreation)?;
            Ok(VkFramebuffer { device, framebuffer })
        }
    }

    impl Deref for VkFramebuffer<'_> {
        type Target = vk::Framebuffer;
        fn deref(&self) -> &vk::Framebuffer {
            &self.framebuffer
        }
    }

    impl Drop for VkFramebuffer<'_> {
        fn drop(&mut self) {
            unsafe { self.device.destroy_framebuffer(self.framebuffer, None) };
        }
    }
}

mod render_pass {
    use super::*;

    pub struct VkRenderPass<'a> {
        device: &'a Device,
        render_pass: vk::RenderPass,
    }

    impl VkRenderPass<'_> {
        /// Wrapper for [Device::create_render_pass].
        pub unsafe fn new<'a>(device: &'a Device, create_info: &vk::RenderPassCreateInfo) -> Result<VkRenderPass<'a>, Error> {
            let render_pass = device
                .create_render_pass(create_info, None)
                .map_err(Error::VulkanRenderPassCreation)?;
            Ok(VkRenderPass { device, render_pass })
        }
    }

    impl Deref for VkRenderPass<'_> {
        type Target = vk::RenderPass;
        fn deref(&self) -> &vk::RenderPass {
            &self.render_pass
        }
    }

    impl Drop for VkRenderPass<'_> {
        fn drop(&mut self) {
            unsafe { self.device.destroy_render_pass(self.render_pass, None) };
        }
    }
}

mod pipelines {
    use super::*;

    pub struct VkPipelines<'a> {
        device: &'a Device,
        pipelines: Vec<vk::Pipeline>,
    }

    impl VkPipelines<'_> {
        /// Wrapper for [Device::create_pipeline].
        pub unsafe fn new<'a>(device: &'a Device, create_infos: &[vk::GraphicsPipelineCreateInfo]) -> Result<VkPipelines<'a>, Error> {
            let pipelines = device
                .create_graphics_pipelines(vk::PipelineCache::null(), create_infos, None)
                .map_err(|(_, err)| Error::VulkanGraphicsPipelineCreation(err))?;
            Ok(VkPipelines { device, pipelines })
        }
    }

    impl Deref for VkPipelines<'_> {
        type Target = [vk::Pipeline];
        fn deref(&self) -> &[vk::Pipeline] {
            &self.pipelines
        }
    }

    impl Drop for VkPipelines<'_> {
        fn drop(&mut self) {
            for pipeline in &self.pipelines {
                unsafe { self.device.destroy_pipeline(*pipeline, None) };
            }
        }
    }
}
