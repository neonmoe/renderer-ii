TODO(cleanup): write a proper readme

## Credits

The sandbox binary uses some external assets created by other people.

The Sponza scene is *the* Sponza scene, specifically the version from the
[glTF-Sample-Models][sponza] repository.

The animated computer-terrarium is [Smol Ame in an Upcycled Terrarium][smol-ame]
by [Seafoam][seafoam], with shape keys removed and meshes triangulated by me,
for compatibility with the renderer (Blender glTF export does not export
tangents for non-triangulated meshes, and meshes with shape keys can't be
triangulated). The original model is licensed CC-BY 4.0.

## Building

### Prerequisites

- Rust and Cargo
- SDL2 development libraries
- Vulkan SDK (or `libvulkan-dev` or similar on linux)

### Building

```sh
cargo build --release
```

This will result in an executable named "sandbox" in
[target/release/](target/release/).

### Shader compilation

Note that compiled SPIR-V versions of the shaders are included in the
repository, so you don't need a glsl-to-spirv toolchain just to build the
library. If you edit the GLSL shader files in `shaders/glsl`, the build script
will try to build them into SPIR-V using `glslc`, in which case you'll need
[shaderc][shaderc] to be installed to build the library.

[sponza]: https://github.com/KhronosGroup/glTF-Sample-Models/tree/master/2.0/Sponza
[smol-ame]: https://sketchfab.com/3d-models/smol-ame-in-an-upcycled-terrarium-hololiveen-490cecc249d242188fda5ad3160a4b24
[seafoam]: https://sketchfab.com/seafoam
[shaderc]: https://github.com/google/shaderc
