//! Wrappers for Vulkan objects, for ensuring proper lifetimes and
//! destroying the objects in the wrappers' Drop impl.

use crate::arena::ImageAllocation;
use crate::Error;
use ash::version::DeviceV1_0;
use ash::vk;
use ash::Device;
use std::ops::Deref;

pub use image_view::VkImageView;

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
            _image: &'a ImageAllocation,
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
