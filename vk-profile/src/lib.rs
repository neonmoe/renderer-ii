extern crate proc_macro;
use std::collections::HashMap;

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
