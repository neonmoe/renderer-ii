use proc_macro2::{Delimiter, Group, Literal, Punct, Spacing};
use proc_macro_error::proc_macro_error;
use quote::{ToTokens, TokenStreamExt};
use shaderc::{
    CompileOptions, Compiler, EnvVersion, IncludeType, OptimizationLevel, ResolvedInclude, ShaderKind, SourceLanguage, SpirvVersion,
    TargetEnv,
};
use std::ffi::OsStr;
use std::fs;
use std::path::PathBuf;
use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
use syn::{Lit, Token};

struct Shader {
    bytes: Vec<u32>,
    source_path: PathBuf,
}

impl ToTokens for Shader {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        let mut byte_stream = proc_macro2::TokenStream::new();
        for byte in &self.bytes {
            byte_stream.append(Literal::u32_suffixed(*byte));
            byte_stream.append(Punct::new(',', Spacing::Alone));
        }
        tokens.append(Group::new(Delimiter::Bracket, byte_stream));
    }
}

struct MacroParams {
    params: Punctuated<Lit, Token![,]>,
}

impl Parse for MacroParams {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        Ok(MacroParams {
            params: input.parse_terminated(Lit::parse)?,
        })
    }
}

#[proc_macro_error]
#[proc_macro]
pub fn include_spirv(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let params = syn::parse_macro_input!(input as MacroParams);
    let mut param_iter = params.params.iter();
    let relative_shader_path = if let Some(path) = param_iter.next() {
        if let Lit::Str(path) = path {
            path.value()
        } else {
            proc_macro_error::abort!(path.span(), "expected shader path to be a string literal");
        }
    } else {
        panic!("expected at least the shader path");
    };
    let defines = param_iter
        .map(|param| {
            if let Lit::Str(define) = param {
                define.value()
            } else {
                proc_macro_error::abort!(param.span(), "expected shader define to be a string literal");
            }
        })
        .collect::<Vec<String>>();
    let shader = compile_shader(&relative_shader_path, &defines);
    let shader_path_string = shader.source_path.to_string_lossy();
    let result = quote::quote!({
        const FILE_DEPENDENCY_MARKER: &str = include_str!(#shader_path_string);
        static BYTES: &[u32] = &#shader;
        (#relative_shader_path, BYTES)
    });
    proc_macro::TokenStream::from(result)
}

fn compile_shader(shader_path: &str, defines: &[String]) -> Shader {
    let shader_dir = PathBuf::from("src").canonicalize().unwrap();
    let mut compiler = Compiler::new().unwrap();
    let mut options = CompileOptions::new().unwrap();
    options.set_optimization_level(OptimizationLevel::Performance);
    if cfg!(feature = "generate-debug-info") {
        options.set_generate_debug_info();
    }
    options.set_target_env(TargetEnv::Vulkan, EnvVersion::Vulkan1_0 as u32);
    options.set_target_spirv(SpirvVersion::V1_0);
    options.set_source_language(SourceLanguage::GLSL);
    options.set_warnings_as_errors();
    options.set_include_callback(
        |requested_name: &str, include_type: IncludeType, requesting_name: &str, include_depth: usize| {
            if include_depth > 100 {
                return Err(String::from("include depth is over 100, there's probably an include loop"));
            }
            match include_type {
                IncludeType::Relative => {
                    let mut path = PathBuf::from(&shader_dir);
                    path.push(requesting_name);
                    path.pop();
                    path.push(requested_name);
                    let relative_path = path.strip_prefix(&shader_dir).unwrap();
                    let resolved_name = relative_path.to_string_lossy().to_string();
                    let content = fs::read_to_string(path).map_err(|err| format!("{}", err))?;
                    Ok(ResolvedInclude { resolved_name, content })
                }
                IncludeType::Standard => Err(String::from("<>-style includes are not implemented")),
            }
        },
    );
    for define in defines {
        options.add_macro_definition(&define, None);
    }

    let mut full_shader_path = shader_dir.clone();
    full_shader_path.push(shader_path);

    let source = fs::read_to_string(&full_shader_path).unwrap();
    let shader_kind = match full_shader_path.extension().and_then(OsStr::to_str) {
        Some("vert") => ShaderKind::Vertex,
        Some("frag") => ShaderKind::Fragment,
        _ => panic!("extension must be 'vert' or 'frag'."),
    };

    match compiler.compile_into_spirv(&source, shader_kind, shader_path, "main", Some(&options)) {
        Ok(compilation_artifact) => Shader {
            bytes: compilation_artifact.as_binary().to_vec(),
            source_path: full_shader_path,
        },
        Err(err) => panic!("{}", err.to_string().trim()),
    }
}
