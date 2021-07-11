This crate contains a procedural macro named `include_spirv!` that
compiles GLSL from the calling crate's src-directory. The compiler
settings are tuned to the renderer this is a part of. The shader is
compiled into a `&'static [u8]` of SPIR-V.
