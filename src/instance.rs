use crate::debug_utils;
use ash::{vk, Entry};
use raw_window_handle::HasRawWindowHandle;
use std::ffi::CStr;
use std::os::raw::c_char;

pub static REQUIRED_VULKAN_VERSION: u32 = vk::API_VERSION_1_2;

#[derive(thiserror::Error, Debug)]
pub enum InstanceCreationError {
    #[error("failed to enumerate vulkan extensions required to create a surface from a window")]
    WindowExtensionEnumeration(#[source] vk::Result),
    #[error("failed to create vulkan window")]
    InstanceCreation(#[source] vk::Result),
}

pub struct Instance {
    pub entry: Entry,
    pub inner: ash::Instance,
    pub debug_utils_available: bool,
    debug_utils_messenger: Option<vk::DebugUtilsMessengerEXT>,
}

impl Drop for Instance {
    fn drop(&mut self) {
        if let Some(debug_utils_messenger) = self.debug_utils_messenger.take() {
            debug_utils::destroy_debug_utils_messenger(&self.entry, &self.inner, debug_utils_messenger);
        }

        {
            profiling::scope!("destroy vulkan instance");
            unsafe { self.inner.destroy_instance(None) };
        }
    }
}

impl Instance {
    pub fn new(window: &dyn HasRawWindowHandle) -> Result<Instance, InstanceCreationError> {
        profiling::scope!("vulkan instance creation");
        let entry = Entry::linked();
        let app_info = vk::ApplicationInfo::builder()
            .application_name(cstr!("neonvk-sandbox"))
            .application_version(make_api_version(0, 0, 1, 0))
            //.api_version(REQUIRED_VULKAN_VERSION);
            // FIXME: This makes renderdoc work, change out later.
            .api_version(vk::API_VERSION_1_3);

        let mut layers = Vec::with_capacity(1);
        let mut validation_layer_enabled = false;
        if cfg!(feature = "vulkan-validation") {
            if is_validation_layer_supported(&entry, "VK_LAYER_KHRONOS_validation") {
                layers.push(cstr!("VK_LAYER_KHRONOS_validation").as_ptr());
                validation_layer_enabled = true;
            } else {
                log::error!("vulkan-validation feature is enabled, but VK_LAYER_KHRONOS_validation is not available on this system");
            }
        }

        let mut extensions = ash_window::enumerate_required_extensions(window)
            .map_err(InstanceCreationError::WindowExtensionEnumeration)?
            .iter()
            .map(|&cs| {
                if let Ok(s) = unsafe { CStr::from_ptr(cs) }.to_str() {
                    log::debug!("Instance extension: {}", s);
                }
                cs
            })
            .collect::<Vec<*const c_char>>();
        let debug_utils_available = is_extension_supported(&entry, "VK_EXT_debug_utils");
        if debug_utils_available {
            extensions.push(cstr!("VK_EXT_debug_utils").as_ptr());
            log::debug!("Instance extension (optional): VK_EXT_debug_utils");
        }

        let create_info = vk::InstanceCreateInfo::builder()
            .application_info(&app_info)
            .enabled_layer_names(&layers)
            .enabled_extension_names(&extensions);

        let enabled_validation_features = [
            vk::ValidationFeatureEnableEXT::BEST_PRACTICES,
            vk::ValidationFeatureEnableEXT::GPU_ASSISTED,
            vk::ValidationFeatureEnableEXT::GPU_ASSISTED_RESERVE_BINDING_SLOT,
            vk::ValidationFeatureEnableEXT::SYNCHRONIZATION_VALIDATION,
        ];
        let mut validation_features = vk::ValidationFeaturesEXT::builder().enabled_validation_features(&enabled_validation_features);

        let create_info = if validation_layer_enabled {
            create_info.push_next(&mut validation_features)
        } else {
            create_info
        };

        let instance = {
            profiling::scope!("vk::create_instance");
            unsafe { entry.create_instance(&create_info, None) }.map_err(InstanceCreationError::InstanceCreation)?
        };

        let debug_utils_messenger = if debug_utils_available {
            debug_utils::init_debug_utils(&entry, &instance);
            debug_utils::create_debug_utils_messenger(&entry, &instance).ok()
        } else {
            None
        };

        Ok(Instance {
            entry,
            inner: instance,
            debug_utils_available,
            debug_utils_messenger,
        })
    }
}

fn make_api_version(variant: u32, major: u32, minor: u32, patch: u32) -> u32 {
    (variant << 29) | (major << 22) | (minor << 12) | patch
}

#[profiling::function]
fn is_validation_layer_supported(entry: &Entry, target_layer_name: &str) -> bool {
    match {
        profiling::scope!("vk::enumerate_instance_layer_properties");
        entry.enumerate_instance_layer_properties()
    } {
        Err(_) => false,
        Ok(layers) => layers.iter().any(|layer_properties| {
            let layer_name_slice = &layer_properties.layer_name[..];
            let layer_name = unsafe { CStr::from_ptr(layer_name_slice.as_ptr()) }.to_string_lossy();
            layer_name == target_layer_name
        }),
    }
}

#[profiling::function]
fn is_extension_supported(entry: &Entry, target_extension_name: &str) -> bool {
    match {
        profiling::scope!("vk::enumerate_instance_extension_properties");
        entry.enumerate_instance_extension_properties(None)
    } {
        Err(_) => false,
        Ok(extensions) => extensions.iter().any(|extension_properties| {
            let extension_name_slice = &extension_properties.extension_name[..];
            let extension_name = unsafe { CStr::from_ptr(extension_name_slice.as_ptr()) }.to_string_lossy();
            extension_name == target_extension_name
        }),
    }
}
