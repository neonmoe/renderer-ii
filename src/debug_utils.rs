use ash::extensions::ext::DebugUtils;
use ash::{vk, Entry, Instance};
use std::ffi::CStr;
use std::os::raw::{c_char, c_void};

pub(crate) fn create_debug_utils_messenger(
    entry: &Entry,
    instance: &Instance,
) -> Result<vk::DebugUtilsMessengerEXT, vk::Result> {
    let debug_utils = DebugUtils::new(entry, instance);
    let create_info = vk::DebugUtilsMessengerCreateInfoEXT {
        message_severity: vk::DebugUtilsMessageSeverityFlagsEXT::all(),
        message_type: vk::DebugUtilsMessageTypeFlagsEXT::all(),
        pfn_user_callback: Some(debug_utils_messenger_callback),
        ..Default::default()
    };
    unsafe { debug_utils.create_debug_utils_messenger(&create_info, None) }
}

pub(crate) fn destroy_debug_utils_messenger(
    entry: &Entry,
    instance: &Instance,
    debug_utils_messenger: vk::DebugUtilsMessengerEXT,
) {
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
            unsafe { CStr::from_ptr(ptr) }.to_string_lossy().to_string()
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
    vulkan_debug(
        message_severity,
        message_types,
        &message_id,
        &message,
        None,
        None,
        None,
    );
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
    // Explicitly silenced warnings, with explanations for why:
    #[allow(clippy::single_match)]
    match message_id {
        "VUID-VkSwapchainCreateInfoKHR-imageExtent-01274" => {
            // This is caused by a the validation layer getting the
            // surface size during the create_swapchain, but the info
            // is filled in beforehand. So if the size of the window
            // changes between these two, the validation layer will
            // complain that the requested extent is too small or too
            // big.
            // TODO: Fix extents for swapchains recreated during a continuous resize
            return;
        }
        _ => {}
    }

    let formatted_message = {
        use std::fmt::Write;
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
        (Severity::ERROR, _) => log::error!("{}", formatted_message),
        (Severity::WARNING, _)
        | (Severity::INFO, _)
        | (_, Type::VALIDATION)
        | (_, Type::PERFORMANCE) => {
            log::debug!("{}", formatted_message)
        }
        (Severity::VERBOSE, _) => log::trace!("{}", formatted_message),
        (severity, _) => log::error!("[unknown severity: {:?}] {}", severity, formatted_message),
    }
}
