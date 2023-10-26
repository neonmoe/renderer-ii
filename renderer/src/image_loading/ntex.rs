use core::convert::TryInto;

use arrayvec::ArrayVec;
use ash::vk;

use crate::image_loading::ImageData;

const MAGIC_STRING: &[u8] = b"The GPU decodable image container format";

#[derive(thiserror::Error, Debug)]
pub enum NtexDecodeError {
    #[error("invalid ntex header (the image file is probably not an ntex file)")]
    InvalidHeader,
    #[error("ntex header contained depth: {0}. Non-1 values are not supported yet.")]
    DepthUnsupported(u32),
    #[error("ntex image data ended early: file contains {actual} of data, expected at least {expected}")]
    NotEnoughPixels { expected: crate::Bytes, actual: crate::Bytes },
    #[error("ntex file length does not match header, expected: {expected}, actual: {actual}")]
    FileLength { expected: usize, actual: usize },
}

#[profiling::function]
pub fn decode(bytes: &[u8]) -> Result<ImageData, NtexDecodeError> {
    let ImageData {
        width,
        height,
        format,
        pixels: _,
        mip_ranges,
    } = decode_header(bytes)?;

    let pixels_len = mip_ranges[mip_ranges.len() - 1].end;
    let bytes_len = pixels_len + 1024;
    if bytes_len > bytes.len() {
        return Err(NtexDecodeError::NotEnoughPixels {
            expected: crate::Bytes(bytes_len as u64),
            actual: crate::Bytes(bytes.len() as u64),
        });
    }
    if bytes.len() != bytes_len {
        return Err(NtexDecodeError::FileLength {
            expected: bytes_len,
            actual: bytes.len(),
        });
    }

    Ok(ImageData {
        width,
        height,
        format,
        pixels: &bytes[1024..1024 + pixels_len],
        mip_ranges,
    })
}

/// Like [`decode`], but [`ImageData::pixels`] is empty.
#[profiling::function]
pub fn decode_header(bytes: &[u8]) -> Result<ImageData<'static>, NtexDecodeError> {
    if &bytes[0..40] != MAGIC_STRING || bytes.len() < 1024 {
        return Err(NtexDecodeError::InvalidHeader);
    }
    let width = u32::from_le_bytes(bytes[992..996].try_into().unwrap());
    let height = u32::from_le_bytes(bytes[996..1000].try_into().unwrap());
    let depth = u32::from_le_bytes(bytes[1000..1004].try_into().unwrap());
    if depth != 1 {
        return Err(NtexDecodeError::DepthUnsupported(depth));
    }
    let mip_levels = u32::from_le_bytes(bytes[1004..1008].try_into().unwrap());
    let format = u32::from_le_bytes(bytes[1008..1012].try_into().unwrap());
    let block_width = u32::from_le_bytes(bytes[1012..1016].try_into().unwrap());
    let block_height = u32::from_le_bytes(bytes[1016..1020].try_into().unwrap());
    let block_size = u32::from_le_bytes(bytes[1020..1024].try_into().unwrap());

    let mut mip_ranges = ArrayVec::new();
    let mut prev_mip_end = 0;
    for mip in 0..mip_levels {
        let mip_width = width / (1 << mip);
        let mip_height = height / (1 << mip);
        let mip_size = (mip_width as f32 / block_width as f32).ceil() as usize
            * (mip_height as f32 / block_height as f32).ceil() as usize
            * block_size as usize;
        mip_ranges.push(prev_mip_end..prev_mip_end + mip_size);
        prev_mip_end += mip_size;
    }

    Ok(ImageData {
        width,
        height,
        format: vk::Format::from_raw(format as i32),
        pixels: &[],
        mip_ranges,
    })
}
