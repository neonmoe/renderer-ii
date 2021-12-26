pub enum HeapType {
    /// For staging buffers, to be moved to GpuLocal later.
    Staging,
    /// For per-frame data, written by the CPU and read by the GPU.
    Temp,
    /// Fastest memory for reading on the GPU. Eventual destination
    /// for long-lived resources.
    GpuLocal,
}

pub struct Allocator;
