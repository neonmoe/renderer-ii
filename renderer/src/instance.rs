use core::ffi::{c_char, CStr};

use arrayvec::ArrayVec;
#[cfg(feature = "vulkan-debug-utils")]
use ash::extensions::ext::DebugUtils;
use ash::{vk, Entry};
use raw_window_handle::HasRawDisplayHandle;

/// Max 2 from [`ash_window::enumerate_required_extensions`] + debug utils.
const MAX_INSTANCE_EXTENSIONS: usize = 2 + 1;

#[derive(thiserror::Error, Debug)]
pub enum InstanceCreationError {
    #[error("failed to enumerate vulkan extensions required to create a surface from a window")]
    WindowExtensionEnumeration(#[source] vk::Result),
    #[error("failed to create vulkan instance")]
    InstanceCreation(#[source] vk::Result),
}

pub struct Instance {
    pub entry: Entry,
    pub inner: ash::Instance,
}

impl Instance {
    pub fn new(
        display: &dyn HasRawDisplayHandle,
        app_name: &CStr,
        major_version: u32,
        minor_version: u32,
        patch_version: u32,
    ) -> Result<Instance, InstanceCreationError> {
        profiling::scope!("vulkan instance creation");
        let version = make_api_version(0, major_version, minor_version, patch_version);
        let entry = Entry::linked();
        // NOTE: The API version given to VkApplicationInfo is a maximum, not a minimum.
        // UNASSIGNED-BestPractices-vkCreateDevice-API-version-mismatch is expected.
        let app_info =
            vk::ApplicationInfo::default().application_name(app_name).application_version(version).api_version(vk::API_VERSION_1_3);
        let app_name = app_name.to_str().unwrap_or("<invalid utf-8>");
        log::debug!(
            "Creating Vulkan instance with application name: \"{app_name}\", version: {major_version}.{minor_version}.{patch_version} (0x{version:X})"
        );

        let mut layers: ArrayVec<*const c_char, 2> = ArrayVec::new();
        let mut validation_layer_enabled = false;
        let mut profiles_layer_enabled = false;
        if cfg!(feature = "vulkan-validation") {
            if is_layer_supported(&entry, "VK_LAYER_KHRONOS_validation") {
                layers.push(cstr!("VK_LAYER_KHRONOS_validation").as_ptr());
                validation_layer_enabled = true;
            } else {
                log::error!("vulkan-validation feature is enabled, but VK_LAYER_KHRONOS_validation is not available on this system");
            }
            if is_layer_supported(&entry, "VK_LAYER_KHRONOS_profiles") {
                layers.push(cstr!("VK_LAYER_KHRONOS_profiles").as_ptr());
                profiles_layer_enabled = true;
            } else {
                log::error!("vulkan-validation feature is enabled, but VK_LAYER_KHRONOS_profiles is not available on this system");
            }
        }

        #[allow(unused_mut)]
        let mut extensions = ash_window::enumerate_required_extensions(display.raw_display_handle())
            .map_err(InstanceCreationError::WindowExtensionEnumeration)?
            .iter()
            .map(|&cs| {
                if let Ok(s) = unsafe { CStr::from_ptr(cs) }.to_str() {
                    log::debug!("Instance extension: {}", s);
                }
                cs
            })
            .collect::<ArrayVec<*const c_char, MAX_INSTANCE_EXTENSIONS>>();
        #[cfg(feature = "vulkan-debug-utils")]
        let mut debug_utils_enabled = false;
        #[cfg(feature = "vulkan-debug-utils")]
        if is_extension_supported(&entry, DebugUtils::NAME) {
            extensions.push(DebugUtils::NAME.as_ptr());
            debug_utils_enabled = true;
            log::debug!("Instance extension (optional): {}", DebugUtils::NAME.to_str().unwrap());
        }

        let mut create_info =
            vk::InstanceCreateInfo::default().application_info(&app_info).enabled_layer_names(&layers).enabled_extension_names(&extensions);

        let enabled_validation_features = [
            vk::ValidationFeatureEnableEXT::BEST_PRACTICES,
            vk::ValidationFeatureEnableEXT::GPU_ASSISTED,
            vk::ValidationFeatureEnableEXT::GPU_ASSISTED_RESERVE_BINDING_SLOT,
            vk::ValidationFeatureEnableEXT::SYNCHRONIZATION_VALIDATION,
        ];
        let mut validation_features = vk::ValidationFeaturesEXT::default().enabled_validation_features(&enabled_validation_features);
        if validation_layer_enabled {
            create_info = create_info.push_next(&mut validation_features);
        }

        if profiles_layer_enabled {
            // ash doesn't have VkProfileLayerSettingsEXT, if/when that's
            // available, prefer include_str!ing the JSON here.
            // TODO: Generate profile JSON based on the renderer's requirements
            // TODO: Assert on the limits in the profile JSON instead of hard coding all the checks (maybe use the Vulkan Profiles library)
            std::env::set_var("VK_KHRONOS_PROFILES_PROFILE_FILE", "renderer/src/vk-profiles/VP_CUSTOM_profile_based_on_2022_baseline.json");
        }

        #[cfg(feature = "vulkan-debug-utils")]
        let mut debug_utils_messenger_create_info = crate::create_debug_utils_messenger_info();
        #[cfg(feature = "vulkan-debug-utils")]
        if debug_utils_enabled {
            create_info = create_info.push_next(&mut debug_utils_messenger_create_info);
        }

        let instance = {
            profiling::scope!("vk::create_instance");
            unsafe { entry.create_instance(&create_info, None) }.map_err(InstanceCreationError::InstanceCreation)?
        };

        #[cfg(feature = "vulkan-debug-utils")]
        if debug_utils_enabled {
            crate::init_debug_utils(&entry, &instance);
            let debug_utils = DebugUtils::new(&entry, &instance);
            let _ = unsafe { debug_utils.create_debug_utils_messenger(&debug_utils_messenger_create_info, None) };
        }

        Ok(Instance { entry, inner: instance })
    }
}

fn make_api_version(variant: u32, major: u32, minor: u32, patch: u32) -> u32 {
    (variant << 29) | (major << 22) | (minor << 12) | patch
}

#[profiling::function]
fn is_layer_supported(entry: &Entry, target_layer_name: &str) -> bool {
    match unsafe {
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
fn is_extension_supported(entry: &Entry, target_extension_name: &CStr) -> bool {
    match unsafe {
        profiling::scope!("vk::enumerate_instance_extension_properties");
        entry.enumerate_instance_extension_properties(None)
    } {
        Err(_) => false,
        Ok(extensions) => extensions.iter().any(|extension_properties| {
            let extension_name_slice = &extension_properties.extension_name[..];
            let extension_name = unsafe { CStr::from_ptr(extension_name_slice.as_ptr()) };
            extension_name == target_extension_name
        }),
    }
}
