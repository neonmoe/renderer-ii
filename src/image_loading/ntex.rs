use crate::Error;
use ash::vk;
use std::convert::TryInto;
use std::ops::Range;

pub struct NtexData<'a> {
    pub width: u32,
    pub height: u32,
    pub format: vk::Format,
    /// Note: not necessarily actual pixels, just the pixel data.
    pub pixels: &'a [u8],
    /// The ranges from `pixels` that represent different mip levels.
    pub mip_ranges: Vec<Range<usize>>,
}

#[profiling::function]
pub fn decode(bytes: &[u8]) -> Result<NtexData, Error> {
    if &bytes[0..40] != b"The GPU decodable image container format" || bytes.len() < 1024 {
        return Err(Error::BadNtex);
    }

    let width = u32::from_le_bytes(bytes[992..996].try_into().unwrap());
    let height = u32::from_le_bytes(bytes[996..1000].try_into().unwrap());
    // let depth = u32::from_le_bytes(bytes[1000..1004].try_into().unwrap());
    let mip_levels = u32::from_le_bytes(bytes[1004..1008].try_into().unwrap());
    let format = u32::from_le_bytes(bytes[1008..1012].try_into().unwrap());
    let block_width = u32::from_le_bytes(bytes[1012..1016].try_into().unwrap());
    let block_height = u32::from_le_bytes(bytes[1016..1020].try_into().unwrap());
    let block_size = u32::from_le_bytes(bytes[1020..1024].try_into().unwrap());
    let mut mip_ranges = Vec::with_capacity(mip_levels as usize);
    let mut prev_mip_end = 0;
    for mip in 0..mip_levels {
        let mip_width = width / (1 << mip);
        let mip_height = height / (1 << mip);
        let mip_size = (mip_width as f32 / block_width as f32).ceil() as usize
            * (mip_height as f32 / block_height as f32).ceil() as usize
            * block_size as usize;
        if prev_mip_end + mip_size + 1024 > bytes.len() {
            return Err(Error::BadNtex);
        }
        mip_ranges.push(prev_mip_end..prev_mip_end + mip_size);
        prev_mip_end += mip_size;
    }
    debug_assert_eq!(bytes.len(), prev_mip_end + 1024);

    Ok(NtexData {
        width,
        height,
        format: vk::Format::from_raw(format as i32),
        pixels: &bytes[1024..],
        mip_ranges,
    })
}
