use arrayvec::ArrayVec;

pub const MAX_COLOR_ATTACHMENTS: usize = 3;
pub const MAX_ATTACHMENTS: usize = MAX_COLOR_ATTACHMENTS + 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttachmentName {
    Hdr,
    Depth,
    TonemappedMultisampled,
    TonemappedSwapchain,
}

pub enum Attachments<T> {
    /// Attachments:
    /// - hdr color,
    /// - depth,
    /// - tonemapped color (presented),
    SingleSampled { hdr: T, depth: T, present_tonemapped: T },
    /// Attachments:
    /// - hdr color (multisampled),
    /// - depth (multisampled),
    /// - tonemapped color (multisampled, resolve source),
    MultiSampled { hdr: T, depth: T, resolve_src_tonemapped: T },
}

impl<T> Attachments<T> {
    /// Converts back to [`Attachments`] from a
    /// [`Attachments::named_attachments`]-shaped vec.
    pub fn create_similar<U>(to: &Attachments<U>, mut arr: ArrayVec<T, MAX_ATTACHMENTS>) -> Self {
        match to {
            Attachments::SingleSampled { .. } => {
                let present_tonemapped = arr.remove(2);
                let depth = arr.remove(1);
                let hdr = arr.remove(0);
                Attachments::SingleSampled {
                    hdr,
                    depth,
                    present_tonemapped,
                }
            }
            Attachments::MultiSampled { .. } => {
                let resolve_src_tonemapped = arr.remove(2);
                let depth = arr.remove(1);
                let hdr = arr.remove(0);
                Attachments::MultiSampled {
                    hdr,
                    depth,
                    resolve_src_tonemapped,
                }
            }
        }
    }

    pub fn all_attachments(&self) -> ArrayVec<(AttachmentName, &T), MAX_ATTACHMENTS> {
        let mut attachments = ArrayVec::new();
        match self {
            Attachments::SingleSampled {
                hdr,
                depth,
                present_tonemapped,
            } => {
                attachments.push((AttachmentName::Hdr, hdr));
                attachments.push((AttachmentName::Depth, depth));
                attachments.push((AttachmentName::TonemappedSwapchain, present_tonemapped));
            }
            Attachments::MultiSampled {
                hdr,
                depth,
                resolve_src_tonemapped,
            } => {
                attachments.push((AttachmentName::Hdr, hdr));
                attachments.push((AttachmentName::Depth, depth));
                attachments.push((AttachmentName::TonemappedMultisampled, resolve_src_tonemapped));
            }
        }
        attachments
    }

    pub fn color_attachments(&self) -> ArrayVec<(AttachmentName, &T), MAX_COLOR_ATTACHMENTS> {
        let mut attachments = ArrayVec::new();
        match self {
            Attachments::SingleSampled {
                hdr,
                depth: _,
                present_tonemapped,
            } => {
                attachments.push((AttachmentName::Hdr, hdr));
                attachments.push((AttachmentName::TonemappedSwapchain, present_tonemapped));
            }
            Attachments::MultiSampled {
                hdr,
                depth: _,
                resolve_src_tonemapped,
            } => {
                attachments.push((AttachmentName::Hdr, hdr));
                attachments.push((AttachmentName::TonemappedMultisampled, resolve_src_tonemapped));
            }
        }
        attachments
    }

    pub fn depth_attachment(&self) -> &T {
        match self {
            Attachments::SingleSampled { depth, .. } | Attachments::MultiSampled { depth, .. } => depth,
        }
    }
}
