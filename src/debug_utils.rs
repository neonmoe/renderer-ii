use alloc::ffi::CString;
use ash::extensions::ext;
use ash::extensions::ext::DebugUtils;
use ash::{vk, Device, Entry, Instance};
use core::ffi::CStr;
use core::ffi::{c_char, c_void};
use core::fmt::Arguments;

static mut DEBUG_UTILS: Option<ext::DebugUtils> = None;

#[profiling::function]
pub(crate) fn init_debug_utils(entry: &Entry, instance: &Instance) {
    unsafe { DEBUG_UTILS = Some(ext::DebugUtils::new(entry, instance)) };
}

#[profiling::function]
pub(crate) fn name_vulkan_object<H: vk::Handle>(device: &Device, object: H, name: Arguments) {
    name_vulkan_object_impl(device, H::TYPE, object.as_raw(), name)
}

fn name_vulkan_object_impl(device: &Device, object_type: vk::ObjectType, object_handle: u64, name: Arguments) {
    if let Some(debug_utils) = unsafe { DEBUG_UTILS.as_ref() } {
        let object_name = format_object_name(format!("{:?}", object_type));
        let name = CString::from_vec_with_nul(format!("{}: {}\0", object_name, name).into_bytes()).unwrap();
        let name_info = vk::DebugUtilsObjectNameInfoEXT {
            object_type,
            object_handle,
            p_object_name: name.as_ptr(),
            ..Default::default()
        };
        let _ = unsafe { debug_utils.set_debug_utils_object_name(device.handle(), &name_info) };
    }
}

fn format_object_name(mut object_name: String) -> String {
    // The object_name string is the debug form of an object type,
    // e.g. IMAGE_VIEW. This transforms it into PascalCase.

    // The first char is already correctly cased.
    for i in 1.. {
        // The length changes over the loop, hence the manual check.
        if i >= object_name.len() {
            break;
        }

        for char_len in 1..=4 {
            if let Some(char_str) = object_name.get(i..i + char_len) {
                if char_str == "_" {
                    // Remove the _. The uppercase character after it will be
                    // left in its place.
                    object_name.remove(i);
                } else {
                    // Everything else gets lowercased.
                    object_name[i..i + char_len].make_ascii_lowercase();
                }
                break;
            }
        }
    }
    object_name
}

#[profiling::function]
pub(crate) fn create_debug_utils_messenger_info() -> vk::DebugUtilsMessengerCreateInfoEXT {
    let message_severity = vk::DebugUtilsMessageSeverityFlagsEXT::INFO
        | vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
        | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
        | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING;
    let message_type = vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION;
    *vk::DebugUtilsMessengerCreateInfoEXT::builder()
        .message_severity(message_severity)
        .message_type(message_type)
        .pfn_user_callback(Some(debug_utils_messenger_callback))
}

#[profiling::function]
pub(crate) fn destroy_debug_utils_messenger(entry: &Entry, instance: &Instance, debug_utils_messenger: vk::DebugUtilsMessengerEXT) {
    let debug_utils = DebugUtils::new(entry, instance);
    unsafe { debug_utils.destroy_debug_utils_messenger(debug_utils_messenger, None) }
}

unsafe extern "system" fn debug_utils_messenger_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_types: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut c_void,
) -> vk::Bool32 {
    if p_callback_data.is_null() {
        return vk::FALSE;
    }
    let ptr_to_str = |ptr: *const c_char| {
        if ptr.is_null() {
            String::from("-")
        } else {
            CStr::from_ptr(ptr).to_string_lossy().to_string()
        }
    };
    let message_id = ptr_to_str((*p_callback_data).p_message_id_name);
    let message = ptr_to_str((*p_callback_data).p_message);
    let message = if message.contains("MessageID = ") && message.contains("] Object ") {
        // Looks like a message with a lot of meta info, pick out the
        // last part containing the actual human-language message.
        message.split(" | ").last().unwrap_or(&message)
    } else {
        &message
    };
    vulkan_debug(message_severity, message_types, &message_id, message, None, None, None);
    vk::FALSE
}

fn vulkan_debug(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_types: vk::DebugUtilsMessageTypeFlagsEXT,
    message_id: &str,
    message: &str,
    queue_label: Option<&str>,
    cmdbuf_label: Option<&str>,
    object_name: Option<&str>,
) {
    let formatted_message = {
        use core::fmt::Write;
        let mut msg = String::with_capacity(256);
        let _ = write!(&mut msg, "[{}] {}", message_id, message);
        if let Some(object_name) = object_name {
            let _ = write!(&mut msg, "\n  Object: {}", object_name);
        }
        if let Some(cmdbuf_label) = cmdbuf_label {
            let _ = write!(&mut msg, "\n  Cmdbuf: {}", cmdbuf_label);
        }
        if let Some(queue_label) = queue_label {
            let _ = write!(&mut msg, "\n  Queue: {}", queue_label);
        }
        msg
    };

    use vk::DebugUtilsMessageSeverityFlagsEXT as Severity;
    use vk::DebugUtilsMessageTypeFlagsEXT as Type;
    match (message_severity, message_types) {
        (Severity::ERROR, _) => {
            log::error!("{}", formatted_message);
            core::hint::black_box(0); // Place a breakpoint here to debug validation errors
        }
        (Severity::WARNING, _) | (Severity::INFO, _) | (_, Type::VALIDATION) | (_, Type::PERFORMANCE) => {
            log::debug!("{}", formatted_message)
        }
        (Severity::VERBOSE, _) => log::trace!("{}", formatted_message),
        (severity, _) => log::error!("[unknown severity: {:?}] {}", severity, formatted_message),
    }
}
