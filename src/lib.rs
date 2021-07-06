//! A Vulkan renderer for 3D games. The mission statement will
//! probably narrow down over time.

use ash::version::EntryV1_0;
use ash::{vk, Entry};
use raw_window_handle::HasRawWindowHandle;
use std::ffi::CStr;
use std::os::raw::c_char;

mod error;
use error::Error;

macro_rules! cstr {
    ($string:literal) => {
        unsafe { CStr::from_bytes_with_nul_unchecked(concat!($string, "\0").as_bytes()) }.as_ptr()
    };
}

pub struct Renderer;

impl Renderer {
    pub fn create(window: &dyn HasRawWindowHandle) -> Result<Renderer, Error> {
        let entry = unsafe { Entry::new().unwrap() };
        let app_info = vk::ApplicationInfo {
            p_application_name: cstr!("neonvk-sandbox"),
            application_version: vk::make_version(0, 1, 0),
            api_version: vk::make_version(1, 0, 0),
            ..Default::default()
        };
        let layers = &[cstr!("VK_LAYER_KHRONOS_validation")];
        let extensions = ash_window::enumerate_required_extensions(window)
            .map_err(|err| Error::WindowRequiredExtensions(err))?;
        log::debug!("Requested extensions: {:#?}", extensions);
        let extensions: Vec<*const c_char> = extensions.into_iter().map(|cs| cs.as_ptr()).collect();
        let create_info = vk::InstanceCreateInfo {
            p_application_info: &app_info,
            enabled_layer_count: layers.len() as u32,
            pp_enabled_layer_names: layers.as_ptr(),
            enabled_extension_count: extensions.len() as u32,
            pp_enabled_extension_names: extensions.as_ptr(),
            ..Default::default()
        };
        let instance = unsafe { entry.create_instance(&create_info, None) }
            .map_err(|err| Error::VulkanInstanceCreation(err))?;
        let _surface = unsafe { ash_window::create_surface(&entry, &instance, window, None) }
            .map_err(|err| Error::VulkanSurfaceCreation(err))?;
        // TODO: Next up: physical device -> device -> queue -> swapchain?
        // TODO: Wire up VK_EXT_debug_utils so that vulkan can log errors to `log`

        Ok(Renderer)
    }
}
