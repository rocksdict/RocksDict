use crate::{
    db_reference::DbReference,
    ffi_try, ffi_try_impl,
    util::{error_message, to_cpath},
    Rdict, RocksDictConfig,
};
use pyo3::{exceptions::PyException, prelude::*};

/// Database's checkpoint object.
/// Used to create checkpoints of the specified DB from time to time.
#[pyclass(name = "Checkpoint")]
pub(crate) struct CheckpointPy {
    /// must keep a reference count of DB to keep DB alive.
    pub(crate) _db: DbReference,

    /// db config
    db_config: RocksDictConfig,

    /// inner checkpoint
    pub(crate) inner: *mut librocksdb_sys::rocksdb_checkpoint_t,
}

#[pymethods]
impl CheckpointPy {
    /// Creates new checkpoint object for specific DB.
    ///
    /// Does not actually produce checkpoints, call `.create_checkpoint()` method to produce
    /// a DB checkpoint.
    #[new]
    #[pyo3(signature = (db))]
    pub fn new(db: &Rdict) -> PyResult<Self> {
        let db_ref = db.get_db()?.clone();

        let checkpoint: *mut librocksdb_sys::rocksdb_checkpoint_t;

        unsafe {
            checkpoint = ffi_try!(librocksdb_sys::rocksdb_checkpoint_object_create(
                db_ref.inner()
            ));
        }

        if checkpoint.is_null() {
            return Err(PyException::new_err("Could not create checkpoint object."));
        }

        Ok(Self {
            inner: checkpoint,
            db_config: db.config(),
            _db: db_ref,
        })
    }

    /// Creates new physical DB checkpoint in directory specified by `path`.
    #[pyo3(signature = (path))]
    pub fn create_checkpoint(&self, path: &str) -> PyResult<()> {
        let cpath = to_cpath(path)?;

        /// Undocumented parameter for `ffi::rocksdb_checkpoint_create` function. Zero by default.
        const LOG_SIZE_FOR_FLUSH: u64 = 0_u64;

        unsafe {
            ffi_try!(librocksdb_sys::rocksdb_checkpoint_create(
                self.inner,
                cpath.as_ptr(),
                LOG_SIZE_FOR_FLUSH,
            ));
        }

        self.db_config.save_to_dir(path)?;
        Ok(())
    }
}

impl Drop for CheckpointPy {
    fn drop(&mut self) {
        unsafe {
            librocksdb_sys::rocksdb_checkpoint_object_destroy(self.inner);
        }
    }
}

unsafe impl Send for CheckpointPy {}
unsafe impl Sync for CheckpointPy {}
