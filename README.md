TODO: write a proper readme

## Building

### Prerequisites

- Rust and Cargo
- CMake
- C compiler

And if you don't have [shaderc](https://github.com/google/shaderc)
installed where the shaderc rust crate can find it, the following:

- Git
- Python 3
- C++11 compiler
- Ninja (optional except for windows)

See the documentation for the [shaderc](https://docs.rs/shaderc) and
[sdl2](https://github.com/Rust-SDL2/rust-sdl2) crates for more info on
the requirements.

### Building

```sh
cargo build --release
```

This will result in an executable named "sandbox" in
[target/release/](target/release/).
