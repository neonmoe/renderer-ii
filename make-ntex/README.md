CLI tool to encode conventional image files into a simple image container
format, using the `image` crate to read images and `intel_tex` for image
compression. All credit for the hard parts to the image crate developers, Intel
for the ISPC texture compressor, and Graham Wihlidal for the rust bindings to
said compressor.

## Motivation

Pngs and jpegs require a heavy decompression step, KTX requires some arcane
information in the form of a Data Format Descriptor, and DDS is some DirectX
stuff I don't want to get involved with. Hence, this:
```
The GPU decodable image container format this file follows:

the first 992 bytes: this null-terminated header including the null
u32: width
u32: height
u32: depth
u32: mip level count
u32: format from the vulkan 1.3 spec
u32: block width
u32: block height
u32: size of one block in bytes
the rest of the bytes: the raw images for each mip level with no padding

A u32 is a 32-bit little-endian unsigned integer.

The first mip level is this many bytes:

  ceil(width / block width) * ceil(height / block height) * (size of one block in bytes)

Each mip level's size after that is simply the previous mip level's size
divided by two, until it would go under the size of one block.

Files in this format should not be considered ground truth.
Handle your source images in a sane format such as PNG.
Convert them into this format for bundling with applications.

This header should be used to distinguish between versions of this format.

The header is 1024 bytes, hopefully it aligns well.
```
This is how every file in this format starts. After this header (and some
newlines, and a null terminator), at byte offset 1024, starts tightly packed
images which can be directly uploaded into a Vulkan buffer, and then to an
image.
