extern crate proc_macro;
use std::collections::HashMap;
use std::fmt::Write;

use heck::ToLowerCamelCase;
use once_cell::sync::Lazy;
use proc_macro::{TokenStream, TokenTree};
use tinyjson::JsonValue;

cfg_if::cfg_if! {
    if #[cfg(feature = "android_baseline_2021")] {
        static JSON: &str = include_str!("VP_ANDROID_baseline_2021.json");
    } else if #[cfg(feature = "android_baseline_2022")] {
        static JSON: &str = include_str!("VP_ANDROID_baseline_2022.json");
    } else if #[cfg(feature = "desktop_baseline_2022")] {
        static JSON: &str = include_str!("VP_LUNARG_desktop_baseline_2022.json");
    } else if #[cfg(feature = "desktop_baseline_2023")] {
        static JSON: &str = include_str!("VP_LUNARG_desktop_baseline_2023.json");
    } else if #[cfg(feature = "desktop_portability_2022")] {
        static JSON: &str = include_str!("VP_LUNARG_desktop_portability_2022.json");
    } else {
        compile_error!("at least one vk-profile feature needs to be enabled");
    }
}

static PROFILE: Lazy<JsonValue> = Lazy::new(|| JSON.parse().unwrap());

/// Expands to the given limit from VkPhysicalDeviceProperties.limits as an
/// integer literal.
#[proc_macro]
pub fn pd_limit(item: TokenStream) -> TokenStream {
    let limit_name = read_string_literal(item, "limit name").to_lower_camel_case();
    let limits = PROFILE["capabilities"]["baseline"]["properties"]["VkPhysicalDeviceProperties"]["limits"]
        .get::<HashMap<_, _>>()
        .unwrap();
    if let Some(limit) = limits.get(&limit_name) {
        let limit = *limit.get::<f64>().unwrap() as i64;
        format!("{limit}").parse().unwrap()
    } else {
        panic!("limit \"{limit_name}\" was not found in the vulkan profile");
    }
}

/// Expands to an array containing (vk::Format, vk::FormatFeatureFlags), where
/// the feature flags are the optimalTilingFeatures of each format listed in the
/// profile.
#[proc_macro]
pub fn all_optimal_tiling_features(_item: TokenStream) -> TokenStream {
    let formats = PROFILE["capabilities"]["baseline"]["formats"].get::<HashMap<_, _>>().unwrap();
    let mut output_code = String::with_capacity(4096);
    output_code.push('[');
    for (format_name, format_props) in formats {
        let format_name = format_name.replace("VK_FORMAT_", "vk::Format::");
        let props = format_props["VkFormatProperties"].get::<HashMap<_, _>>().unwrap();
        if let Some(optimal_features) = props.get("optimalTilingFeatures") {
            let optimal_features = optimal_features.get::<Vec<_>>().unwrap();
            if optimal_features.is_empty() {
                continue;
            }
            write!(&mut output_code, "({format_name}, ").unwrap();
            for (i, feature) in optimal_features.iter().map(|s| s.get::<String>().unwrap()).enumerate() {
                if i > 0 {
                    output_code.push('|');
                }
                let feature = feature.strip_suffix("_BIT").unwrap();
                let feature = feature.replace("VK_FORMAT_FEATURE_", "vk::FormatFeatureFlags::");
                write!(&mut output_code, "{feature}").unwrap();
            }
            output_code.push_str("),\n");
        }
    }
    output_code.push(']');
    output_code.parse().unwrap()
}

fn read_string_literal(item: TokenStream, name: &str) -> String {
    let Some(arg) = item.into_iter().next() else {
        panic!("expected {name}");
    };
    let TokenTree::Literal(arg) = arg else {
        panic!("{name} must be a string literal");
    };
    let mut arg = arg.to_string();
    if arg.len() < 2 || &arg[0..1] != "\"" || &arg[arg.len() - 1..arg.len()] != "\"" {
        panic!("{name} must be a string literal");
    }
    arg.truncate(arg.len() - 1);
    arg.remove(0);
    arg
}
