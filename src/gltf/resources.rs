use crate::Error;
use std::borrow::Cow;
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::rc::Rc;

/// A library of external resources used when loading glTF files.
///
/// Can be shared between multiple glTFs. If they refer to the same
/// resources, this will even help with memory consumption, as they
/// will share the resource cache.
#[derive(Default)]
pub struct GltfResources {
    /// A cache of loaded resources. The keys map to glTF URIs.
    ///
    /// Tip: to bundle resources in the executable, you can populate
    /// this with [include_str] and [include_bytes].
    pub resource_cache: HashMap<PathBuf, Rc<Cow<'static, [u8]>>>,
    /// The base path from which missing resources are loaded.
    pub loading_path: Option<PathBuf>,
}

impl GltfResources {
    pub fn with_path(path: PathBuf) -> GltfResources {
        GltfResources {
            loading_path: Some(path),
            ..Default::default()
        }
    }

    #[profiling::function]
    pub fn insert<U: Into<PathBuf>, B: Into<Cow<'static, [u8]>>>(&mut self, uri: U, bytes: B) {
        self.resource_cache.insert(uri.into(), Rc::new(bytes.into()));
    }

    /// Load the file from `uri` relative to
    /// [GltfResources::loading_path] if needed, and return a clone of
    /// the value in the cache.
    ///
    /// The buffer has to be cloned from the HashMap, since borrowing
    /// would also borrow the entire map, since otherwise you could
    /// remove the borrowed element from the map. An append-only
    /// HashMap would be nice, but I don't really want that much more
    /// code just for this.
    #[profiling::function]
    pub fn get_or_load<U: Into<PathBuf>>(&mut self, uri: U) -> Result<Rc<Cow<'static, [u8]>>, Error> {
        let uri: PathBuf = uri.into();
        match self.resource_cache.entry(uri.clone()) {
            Entry::Occupied(occupied) => Ok(occupied.get().clone()),
            Entry::Vacant(vacant) => match &self.loading_path {
                Some(base_path) => {
                    let path = base_path.join(&uri);
                    let buffer = fs::read(path).map_err(|err| Error::GltfBufferLoading(uri.to_string_lossy().to_string(), err))?;
                    let buffer = Rc::new(Cow::Owned(buffer));
                    Ok(vacant.insert(buffer).clone())
                }
                None => Err(Error::GltfMissingDirectory(uri.to_string_lossy().to_string())),
            },
        }
    }
}
