/// Container for a u8 slice to align it to 4 bytes.
#[repr(C, align(4))]
pub(crate) struct U32AlignedBytes<const SIZE: usize>(pub [u8; SIZE]);

/// Transmutes the slice to be u32s and divides the length by 4. `bytes` must be
/// aligned to 4 bytes.
pub(crate) const unsafe fn include_bytes_as_u32s(bytes: &[u8]) -> &[u32] {
    // Safety: U32AlignedBytes has align(4) and the ptr is at offset 0.
    let u32_ptr: *const u32 = std::mem::transmute(bytes.as_ptr());
    // There are 4x as many elements in the u8 slice as the u32 slice.
    let u32_slice_len = bytes.len() / 4;
    // Safety: ptr and len are based on the safe slice above, len is modified ot match the u8 -> u32 transmute.
    core::slice::from_raw_parts(u32_ptr, u32_slice_len)
}

/// Like `include_bytes`, but it transmutes the u8s to a slice of one-fourth as
/// many u32s, properly aligned. Does not transpose the words based on
/// endianness.
#[macro_export]
macro_rules! include_words {
    ($path:expr) => {{
        // The actual SPIR-V bytes as a static slice, aligned at 4 byes with the above struct.
        static BYTES: &[u8] = &$crate::include_words::U32AlignedBytes(*include_bytes!($path)).0;
        unsafe { $crate::include_words::include_bytes_as_u32s(BYTES) }
    }};
}
