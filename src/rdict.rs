use crate::encoder::{decode_value, encode_key, encode_raw, encode_value};
use crate::iter::{RdictItems, RdictKeys, RdictValues};
use crate::options::{CachePy, EnvPy, SliceTransformType};
use crate::{
    CompactOptionsPy, FlushOptionsPy, IngestExternalFileOptionsPy, OptionsPy, RdictIter,
    ReadOptionsPy, Snapshot, WriteBatchPy, WriteOptionsPy,
};
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rocksdb::{
    ColumnFamily, ColumnFamilyDescriptor, Direction, FlushOptions, IteratorMode, LiveFile,
    ReadOptions, WriteOptions, DB, DEFAULT_COLUMN_FAMILY_NAME,
};
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::HashMap;
use std::fs;
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::Duration;

pub const ROCKSDICT_CONFIG_FILE: &str = "rocksdict-config.json";
/// 8MB default LRU cache size
pub const DEFAULT_LRU_CACHE_SIZE: usize = 8 * 1024 * 1024;

pub fn config_file(path: &str) -> PathBuf {
    let mut config_path = PathBuf::from(path);
    config_path.push(ROCKSDICT_CONFIG_FILE);
    config_path
}

///
/// A persistent on-disk dictionary. Supports string, int, float, bytes as key, values.
///
/// Example:
///     ::
///
///         from rocksdict import Rdict
///
///         db = Rdict("./test_dir")
///         db[0] = 1
///
///         db = None
///         db = Rdict("./test_dir")
///         assert(db[0] == 1)
///
/// Args:
///     path (str): path to the database
///     options (Options): Options object
///     column_families (dict): (name, options) pairs, these `Options`
///         must have the same `raw_mode` argument as the main `Options`.
///     access_type (AccessType): there are four access types:
///         ReadWrite, ReadOnly, WithTTL, and Secondary, use
///         AccessType class to create.
#[pyclass(name = "Rdict")]
pub(crate) struct Rdict {
    pub(crate) write_opt: WriteOptions,
    pub(crate) flush_opt: FlushOptionsPy,
    pub(crate) read_opt: ReadOptions,
    pub(crate) pickle_loads: PyObject,
    pub(crate) pickle_dumps: PyObject,
    pub(crate) write_opt_py: WriteOptionsPy,
    pub(crate) read_opt_py: ReadOptionsPy,
    pub(crate) column_family: Option<Arc<ColumnFamily>>,
    pub(crate) opt_py: OptionsPy,
    pub(crate) slice_transforms: Arc<RwLock<HashMap<String, SliceTransformType>>>,
    // drop DB last
    pub(crate) db: Option<Arc<RefCell<DB>>>,
}

/// Define DB Access Types.
///
/// Notes:
///     There are four access types:
///      - ReadWrite: default value
///      - ReadOnly
///      - WithTTL
///      - Secondary
///
/// Examples:
///     ::
///
///         from rocksdict import Rdict, AccessType
///
///         # open with 24 hours ttl
///         db = Rdict("./main_path", access_type = AccessType.with_ttl(24 * 3600))
///
///         # open as read_only
///         db = Rdict("./main_path", access_type = AccessType.read_only())
///
///         # open as secondary
///         db = Rdict("./main_path", access_type = AccessType.secondary("./secondary_path"))
///
#[derive(Clone)]
#[pyclass(name = "AccessType")]
pub(crate) struct AccessType(AccessTypeInner);

#[derive(Serialize, Deserialize)]
pub struct RocksDictConfig {
    pub raw_mode: bool,
    // mapping from column families to SliceTransformType
    pub prefix_extractors: HashMap<String, SliceTransformType>,
}

impl RocksDictConfig {
    pub fn load<P: AsRef<Path>>(path: P) -> PyResult<Self> {
        let config_file = fs::File::options().read(true).open(path)?;
        match serde_json::from_reader(config_file) {
            Ok(c) => Ok(c),
            Err(e) => Err(PyException::new_err(e.to_string())),
        }
    }

    pub fn save<P: AsRef<Path>>(&self, path: P) -> PyResult<()> {
        let config_file = fs::File::options().create(true).write(true).open(path)?;
        match serde_json::to_writer(config_file, self) {
            Ok(_) => Ok(()),
            Err(e) => Err(PyException::new_err(e.to_string())),
        }
    }
}

impl Rdict {
    fn dump_config(&self) -> PyResult<()> {
        let config_path = config_file(&self.path()?);
        RocksDictConfig {
            raw_mode: self.opt_py.raw_mode,
            prefix_extractors: self.slice_transforms.read().unwrap().clone(),
        }
        .save(config_path)
    }
}

#[pymethods]
impl Rdict {
    /// Create a new database or open an existing one.
    ///
    /// If Options are not provided:
    /// - first, attempt to read from the path
    /// - if failed to read from the path, use default
    #[new]
    #[pyo3(signature = (
        path,
        options = None,
        column_families = None,
        access_type = AccessType::read_write()
    ))]
    fn new(
        path: &str,
        options: Option<OptionsPy>,
        column_families: Option<HashMap<String, OptionsPy>>,
        access_type: AccessType,
        py: Python,
    ) -> PyResult<Self> {
        let pickle = PyModule::import(py, "pickle")?.to_object(py);
        let options_loaded = OptionsPy::load_latest_inner(
            path,
            EnvPy::default()?,
            false,
            CachePy::new_lru_cache(DEFAULT_LRU_CACHE_SIZE)?,
        );
        let (options, column_families) = match (options_loaded, options, column_families) {
            (Ok((opt_loaded, cols_loaded)), opt, cols) => match (opt, cols) {
                (Some(opt), Some(cols)) => (opt, Some(cols)),
                (Some(opt), None) => (opt, Some(cols_loaded)),
                (None, Some(cols)) => (opt_loaded, Some(cols)),
                (None, None) => (opt_loaded, Some(cols_loaded)),
            },
            (Err(_), Some(opt), cols) => (opt, cols),
            (Err(_), None, cols) => {
                log::info!("using default configuration");
                (OptionsPy::new(false), cols)
            }
        };
        // save slice transforms types in rocksdict config
        let config_path = config_file(path);
        let mut prefix_extractors = HashMap::new();
        if let Some(slice_transform) = &options.prefix_extractor {
            prefix_extractors.insert(
                DEFAULT_COLUMN_FAMILY_NAME.to_string(),
                slice_transform.clone(),
            );
        }
        if let Some(cf) = &column_families {
            for (name, opt) in cf.iter() {
                if let Some(slice_transform) = &opt.prefix_extractor {
                    prefix_extractors.insert(name.clone(), slice_transform.clone());
                }
            }
        }
        let rocksdict_config = RocksDictConfig {
            raw_mode: options.raw_mode,
            prefix_extractors: prefix_extractors.clone(),
        };
        let opt_inner = &options.inner_opt;
        match fs::create_dir_all(path) {
            Ok(_) => match {
                if let Some(cf) = column_families {
                    let mut has_default_cf = false;
                    // check options_raw_mode for column families
                    for (cf_name, cf_opt) in cf.iter() {
                        if cf_opt.raw_mode != options.raw_mode {
                            return Err(PyException::new_err(format!(
                                "Options should have raw_mode={}",
                                options.raw_mode
                            )));
                        }
                        if cf_name.as_str() == DEFAULT_COLUMN_FAMILY_NAME {
                            has_default_cf = true;
                        }
                    }
                    let mut cfs = cf
                        .into_iter()
                        .map(|(name, opt)| ColumnFamilyDescriptor::new(name, opt.inner_opt))
                        .collect::<Vec<_>>();
                    // automatically add default column families
                    if !has_default_cf {
                        cfs.push(ColumnFamilyDescriptor::new(
                            DEFAULT_COLUMN_FAMILY_NAME,
                            opt_inner.clone(),
                        ));
                    }
                    match access_type.0 {
                        AccessTypeInner::ReadWrite => DB::open_cf_descriptors(opt_inner, path, cfs),
                        AccessTypeInner::ReadOnly {
                            error_if_log_file_exist,
                        } => DB::open_cf_descriptors_read_only(
                            opt_inner,
                            path,
                            cfs,
                            error_if_log_file_exist,
                        ),
                        AccessTypeInner::Secondary { secondary_path } => {
                            DB::open_cf_descriptors_as_secondary(
                                opt_inner,
                                path,
                                &secondary_path,
                                cfs,
                            )
                        }
                        AccessTypeInner::WithTTL { ttl } => {
                            DB::open_cf_descriptors_with_ttl(opt_inner, path, cfs, ttl)
                        }
                    }
                } else {
                    match access_type.0 {
                        AccessTypeInner::ReadWrite => DB::open(opt_inner, path),
                        AccessTypeInner::ReadOnly {
                            error_if_log_file_exist,
                        } => DB::open_for_read_only(opt_inner, path, error_if_log_file_exist),
                        AccessTypeInner::Secondary { secondary_path } => {
                            DB::open_as_secondary(opt_inner, path, &secondary_path)
                        }
                        AccessTypeInner::WithTTL { ttl } => DB::open_with_ttl(opt_inner, path, ttl),
                    }
                }
            } {
                Ok(db) => {
                    let r_opt = ReadOptionsPy::default(options.raw_mode, py)?;
                    let w_opt = WriteOptionsPy::new();
                    // save rocksdict config
                    rocksdict_config.save(config_path)?;
                    Ok(Rdict {
                        db: Some(Arc::new(RefCell::new(db))),
                        write_opt: (&w_opt).into(),
                        flush_opt: FlushOptionsPy::new(),
                        read_opt: (&r_opt).into(),
                        pickle_loads: pickle.getattr(py, "loads")?,
                        pickle_dumps: pickle.getattr(py, "dumps")?,
                        write_opt_py: w_opt,
                        read_opt_py: r_opt,
                        column_family: None,
                        opt_py: options.clone(),
                        slice_transforms: Arc::new(RwLock::new(prefix_extractors)),
                    })
                }
                Err(e) => Err(PyException::new_err(e.to_string())),
            },
            Err(e) => Err(PyException::new_err(e.to_string())),
        }
    }

    /// Optionally disable WAL or sync for this write.
    ///
    /// Example:
    ///     ::
    ///
    ///         from rocksdict import Rdict, Options, WriteBatch, WriteOptions
    ///
    ///         path = "_path_for_rocksdb_storageY1"
    ///         db = Rdict(path)
    ///
    ///         # set write options
    ///         write_options = WriteOptions()
    ///         write_options.set_sync(False)
    ///         write_options.disable_wal(True)
    ///         db.set_write_options(write_options)
    ///
    ///         # write to db
    ///         db["my key"] = "my value"
    ///         db["key2"] = "value2"
    ///         db["key3"] = "value3"
    ///
    ///         # remove db
    ///         del db
    ///         Rdict.destroy(path)
    fn set_write_options(&mut self, write_opt: &WriteOptionsPy) {
        self.write_opt = write_opt.into();
        self.write_opt_py = write_opt.clone();
    }

    /// Configure Read Options for all the get operations.
    fn set_read_options(&mut self, read_opt: &ReadOptionsPy) -> PyResult<()> {
        if self.opt_py.raw_mode != read_opt.raw_mode {
            return Err(PyException::new_err(format!(
                "ReadOptions raw_mode should be set to {}",
                read_opt.raw_mode
            )));
        }
        self.read_opt = read_opt.into();
        self.read_opt_py = read_opt.clone();
        Ok(())
    }

    /// Parse list for batch get.
    fn __getitem__(&self, key: &PyAny, py: Python) -> PyResult<PyObject> {
        if let Some(db) = &self.db {
            // batch_get
            if let Ok(keys) = PyTryFrom::try_from(key) {
                return Ok(get_batch_inner(
                    db,
                    keys,
                    py,
                    &self.read_opt,
                    &self.pickle_loads,
                    &self.column_family,
                    self.opt_py.raw_mode,
                )?
                .to_object(py));
            }
            let db = db.borrow();
            let value_result = if self.opt_py.raw_mode {
                let key = encode_raw(key)?;
                if let Some(cf) = &self.column_family {
                    db.get_pinned_cf_opt(cf.deref(), key, &self.read_opt)
                } else {
                    db.get_pinned_opt(key, &self.read_opt)
                }
            } else {
                let key = encode_key(key, self.opt_py.raw_mode)?;
                if let Some(cf) = &self.column_family {
                    db.get_pinned_cf_opt(cf.deref(), key, &self.read_opt)
                } else {
                    db.get_pinned_opt(key, &self.read_opt)
                }
            };
            match value_result {
                Ok(value) => match value {
                    None => Err(PyException::new_err("key not found")),
                    Some(slice) => {
                        decode_value(py, slice.as_ref(), &self.pickle_loads, self.opt_py.raw_mode)
                    }
                },
                Err(e) => Err(PyException::new_err(e.to_string())),
            }
        } else {
            Err(PyException::new_err("DB already closed"))
        }
    }

    fn __setitem__(&self, key: &PyAny, value: &PyAny, py: Python) -> PyResult<()> {
        if let Some(db) = &self.db {
            let db = db.borrow();
            if self.opt_py.raw_mode {
                let key = encode_raw(key)?;
                let value = encode_raw(value)?;
                let put_result = if let Some(cf) = &self.column_family {
                    db.put_cf_opt(cf.deref(), key, value, &self.write_opt)
                } else {
                    db.put_opt(key, value, &self.write_opt)
                };
                match put_result {
                    Ok(_) => Ok(()),
                    Err(e) => Err(PyException::new_err(e.to_string())),
                }
            } else {
                let key = encode_key(key, self.opt_py.raw_mode)?;
                let value = encode_value(value, &self.pickle_dumps, self.opt_py.raw_mode, py)?;
                let put_result = if let Some(cf) = &self.column_family {
                    db.put_cf_opt(cf.deref(), key, value, &self.write_opt)
                } else {
                    db.put_opt(key, value, &self.write_opt)
                };
                match put_result {
                    Ok(_) => Ok(()),
                    Err(e) => Err(PyException::new_err(e.to_string())),
                }
            }
        } else {
            Err(PyException::new_err("DB already closed"))
        }
    }

    fn __contains__(&self, key: &PyAny) -> PyResult<bool> {
        if let Some(db) = &self.db {
            let db = db.borrow();
            let may_exist = if self.opt_py.raw_mode {
                let key = encode_raw(key)?;
                if let Some(cf) = &self.column_family {
                    db.key_may_exist_cf_opt(cf.deref(), key, &self.read_opt)
                } else {
                    db.key_may_exist_opt(key, &self.read_opt)
                }
            } else {
                let key = encode_key(key, self.opt_py.raw_mode)?;
                if let Some(cf) = &self.column_family {
                    db.key_may_exist_cf_opt(cf.deref(), &key[..], &self.read_opt)
                } else {
                    db.key_may_exist_opt(&key[..], &self.read_opt)
                }
            };
            if may_exist {
                let value_result = if self.opt_py.raw_mode {
                    let key = encode_raw(key)?;
                    if let Some(cf) = &self.column_family {
                        db.get_pinned_cf_opt(cf.deref(), key, &self.read_opt)
                    } else {
                        db.get_pinned_opt(key, &self.read_opt)
                    }
                } else {
                    let key = encode_key(key, self.opt_py.raw_mode)?;
                    if let Some(cf) = &self.column_family {
                        db.get_pinned_cf_opt(cf.deref(), &key[..], &self.read_opt)
                    } else {
                        db.get_pinned_opt(&key[..], &self.read_opt)
                    }
                };
                match value_result {
                    Ok(value) => match value {
                        None => Ok(false),
                        Some(_) => Ok(true),
                    },
                    Err(e) => Err(PyException::new_err(e.to_string())),
                }
            } else {
                Ok(false)
            }
        } else {
            Err(PyException::new_err("DB already closed"))
        }
    }

    fn __delitem__(&self, key: &PyAny) -> PyResult<()> {
        if let Some(db) = &self.db {
            let db = db.borrow();
            let del_result = if self.opt_py.raw_mode {
                let key = encode_raw(key)?;
                if let Some(cf) = &self.column_family {
                    db.delete_cf_opt(cf.deref(), key, &self.write_opt)
                } else {
                    db.delete_opt(key, &self.write_opt)
                }
            } else {
                let key = encode_key(key, self.opt_py.raw_mode)?;
                if let Some(cf) = &self.column_family {
                    db.delete_cf_opt(cf.deref(), &key[..], &self.write_opt)
                } else {
                    db.delete_opt(&key[..], &self.write_opt)
                }
            };
            match del_result {
                Ok(_) => Ok(()),
                Err(e) => Err(PyException::new_err(e.to_string())),
            }
        } else {
            Err(PyException::new_err("DB already closed"))
        }
    }

    /// Reversible for iterating over keys and values.
    ///
    /// Examples:
    ///     ::
    ///
    ///         from rocksdict import Rdict, Options, ReadOptions
    ///
    ///         path = "_path_for_rocksdb_storage5"
    ///         db = Rdict(path)
    ///
    ///         for i in range(50):
    ///             db[i] = i ** 2
    ///
    ///         iter = db.iter()
    ///
    ///         iter.seek_to_first()
    ///
    ///         j = 0
    ///         while iter.valid():
    ///             assert iter.key() == j
    ///             assert iter.value() == j ** 2
    ///             print(f"{iter.key()} {iter.value()}")
    ///             iter.next()
    ///             j += 1
    ///
    ///         iter.seek_to_first();
    ///         assert iter.key() == 0
    ///         assert iter.value() == 0
    ///         print(f"{iter.key()} {iter.value()}")
    ///
    ///         iter.seek(25)
    ///         assert iter.key() == 25
    ///         assert iter.value() == 625
    ///         print(f"{iter.key()} {iter.value()}")
    ///
    ///         del iter, db
    ///         Rdict.destroy(path)
    ///
    /// Args:
    ///     read_opt: ReadOptions, must have the same `raw_mode` argument.
    ///
    /// Returns: Reversible
    #[pyo3(signature = (read_opt = None))]
    fn iter(&self, read_opt: Option<&ReadOptionsPy>, py: Python) -> PyResult<RdictIter> {
        let read_opt: ReadOptionsPy = match read_opt {
            None => ReadOptionsPy::default(self.opt_py.raw_mode, py)?,
            Some(opt) => opt.clone(),
        };
        if let Some(db) = &self.db {
            Ok(RdictIter::new(
                db,
                &self.column_family,
                read_opt,
                &self.pickle_loads,
                self.opt_py.raw_mode,
            )?)
        } else {
            Err(PyException::new_err("DB already closed"))
        }
    }

    /// Iterate through all keys and values pairs.
    ///
    /// Examples:
    ///     ::
    ///
    ///         for k, v in db.items():
    ///             print(f"{k} -> {v}")
    ///
    /// Args:
    ///     backwards: iteration direction, forward if `False`.
    ///     from_key: iterate from key, first seek to this key
    ///         or the nearest next key for iteration
    ///         (depending on iteration direction).
    ///     read_opt: ReadOptions, must have the same `raw_mode` argument.
    #[pyo3(signature = (backwards = false, from_key = None, read_opt = None))]
    fn items(
        &self,
        backwards: bool,
        from_key: Option<&PyAny>,
        read_opt: Option<&ReadOptionsPy>,
        py: Python,
    ) -> PyResult<RdictItems> {
        RdictItems::new(self.iter(read_opt, py)?, backwards, from_key)
    }

    /// Iterate through all keys
    ///
    /// Examples:
    ///     ::
    ///
    ///         all_keys = [k for k in db.keys()]
    ///
    /// Args:
    ///     backwards: iteration direction, forward if `False`.
    ///     from_key: iterate from key, first seek to this key
    ///         or the nearest next key for iteration
    ///         (depending on iteration direction).
    ///     read_opt: ReadOptions, must have the same `raw_mode` argument.
    #[pyo3(signature = (backwards = false, from_key = None, read_opt = None))]
    fn keys(
        &self,
        backwards: bool,
        from_key: Option<&PyAny>,
        read_opt: Option<&ReadOptionsPy>,
        py: Python,
    ) -> PyResult<RdictKeys> {
        RdictKeys::new(self.iter(read_opt, py)?, backwards, from_key)
    }

    /// Iterate through all values.
    ///
    /// Examples:
    ///     ::
    ///
    ///         all_keys = [v for v in db.values()]
    ///
    /// Args:
    ///     backwards: iteration direction, forward if `False`.
    ///     from_key: iterate from key, first seek to this key
    ///         or the nearest next key for iteration
    ///         (depending on iteration direction).
    ///     read_opt: ReadOptions, must have the same `raw_mode` argument.
    #[pyo3(signature = (backwards = false, from_key = None, read_opt = None))]
    fn values(
        &self,
        backwards: bool,
        from_key: Option<&PyAny>,
        read_opt: Option<&ReadOptionsPy>,
        py: Python,
    ) -> PyResult<RdictValues> {
        RdictValues::new(self.iter(read_opt, py)?, backwards, from_key)
    }

    /// Manually flush the current column family.
    ///
    /// Notes:
    ///     Manually call mem-table flush.
    ///     It is recommended to call flush() or close() before
    ///     stopping the python program, to ensure that all written
    ///     key-value pairs have been flushed to the disk.
    ///
    /// Args:
    ///     wait (bool): whether to wait for the flush to finish.
    #[pyo3(signature = (wait = true))]
    fn flush(&self, wait: bool) -> PyResult<()> {
        if let Some(db) = &self.db {
            let mut f_opt = FlushOptions::new();
            f_opt.set_wait(wait);
            let db = db.borrow();
            let flush_result = if let Some(cf) = &self.column_family {
                db.flush_cf_opt(cf.deref(), &f_opt)
            } else {
                db.flush_opt(&f_opt)
            };
            match flush_result {
                Ok(_) => Ok(()),
                Err(e) => Err(PyException::new_err(e.into_string())),
            }
        } else {
            Err(PyException::new_err("DB already closed"))
        }
    }

    /// Flushes the WAL buffer. If `sync` is set to `true`, also syncs
    /// the data to disk.
    #[pyo3(signature = (sync = true))]
    fn flush_wal(&self, sync: bool) -> PyResult<()> {
        if let Some(db) = &self.db {
            let db = db.borrow();
            let flush_result = db.flush_wal(sync);
            match flush_result {
                Ok(_) => Ok(()),
                Err(e) => Err(PyException::new_err(e.into_string())),
            }
        } else {
            Err(PyException::new_err("DB already closed"))
        }
    }

    /// Creates column family with given name and options.
    ///
    /// Args:
    ///     name: name of this column family
    ///     options: Rdict Options for this column family
    ///
    /// Return:
    ///     the newly created column family
    #[pyo3(signature = (name, options = OptionsPy::new(false)))]
    fn create_column_family(&self, name: &str, options: OptionsPy) -> PyResult<Rdict> {
        if options.raw_mode != self.opt_py.raw_mode {
            return Err(PyException::new_err(format!(
                "Options should have raw_mode={}",
                self.opt_py.raw_mode
            )));
        }
        // write slice_transform info into config file
        if let Some(slice_transform) = options.prefix_extractor {
            self.slice_transforms
                .write()
                .unwrap()
                .insert(name.to_string(), slice_transform);
        }
        self.dump_config()?;
        if let Some(db) = &self.db {
            let create_result = db.borrow_mut().create_cf(name, &options.inner_opt);
            match create_result {
                Ok(_) => Ok(self.get_column_family(name)?),
                Err(e) => Err(PyException::new_err(e.to_string())),
            }
        } else {
            Err(PyException::new_err("DB already closed"))
        }
    }

    /// Drops the column family with the given name
    fn drop_column_family(&self, name: &str) -> PyResult<()> {
        if let Some(db) = &self.db {
            match db.borrow_mut().drop_cf(name) {
                Ok(_) => Ok(()),
                Err(e) => Err(PyException::new_err(e.to_string())),
            }
        } else {
            Err(PyException::new_err("DB already closed"))
        }
    }

    /// Get a column family Rdict
    ///
    /// Args:
    ///     name: name of this column family
    ///     options: Rdict Options for this column family
    ///
    /// Return:
    ///     the column family Rdict of this name
    pub fn get_column_family(&self, name: &str) -> PyResult<Self> {
        if let Some(db) = &self.db {
            match db.borrow().cf_handle(name) {
                None => Err(PyException::new_err(format!(
                    "column name `{}` does not exist, use `create_cf` to creat it",
                    name
                ))),
                Some(cf) => Ok(Self {
                    db: Some(db.clone()),
                    write_opt: (&self.write_opt_py).into(),
                    flush_opt: self.flush_opt,
                    read_opt: (&self.read_opt_py).into(),
                    pickle_loads: self.pickle_loads.clone(),
                    pickle_dumps: self.pickle_dumps.clone(),
                    column_family: Some(cf),
                    write_opt_py: self.write_opt_py.clone(),
                    read_opt_py: self.read_opt_py.clone(),
                    opt_py: self.opt_py.clone(),
                    slice_transforms: self.slice_transforms.clone(),
                }),
            }
        } else {
            Err(PyException::new_err("DB already closed"))
        }
    }

    /// Use this method to obtain a ColumnFamily instance, which can be used in WriteBatch.
    ///
    /// Example:
    ///     ::
    ///
    ///         wb = WriteBatch()
    ///         for i in range(100):
    ///             wb.put(i, i**2, db.get_column_family_handle(cf_name_1))
    ///         db.write(wb)
    ///
    ///         wb = WriteBatch()
    ///         wb.set_default_column_family(db.get_column_family_handle(cf_name_2))
    ///         for i in range(100, 200):
    ///             wb[i] = i**2
    ///         db.write(wb)
    pub fn get_column_family_handle(&self, name: &str) -> PyResult<ColumnFamilyPy> {
        if let Some(db) = &self.db {
            match db.borrow().cf_handle(name) {
                None => Err(PyException::new_err(format!(
                    "column name `{}` does not exist, use `create_cf` to creat it",
                    name
                ))),
                Some(cf) => Ok(ColumnFamilyPy { cf, db: db.clone() }),
            }
        } else {
            Err(PyException::new_err("DB already closed"))
        }
    }

    /// A snapshot of the current column family.
    ///
    /// Examples:
    ///     ::
    ///
    ///         from rocksdict import Rdict
    ///
    ///         db = Rdict("tmp")
    ///         for i in range(100):
    ///             db[i] = i
    ///
    ///         # take a snapshot
    ///         snapshot = db.snapshot()
    ///
    ///         for i in range(90):
    ///             del db[i]
    ///
    ///         # 0-89 are no longer in db
    ///         for k, v in db.items():
    ///             print(f"{k} -> {v}")
    ///
    ///         # but they are still in the snapshot
    ///         for i in range(100):
    ///             assert snapshot[i] == i
    ///
    ///         # drop the snapshot
    ///         del snapshot, db
    ///
    ///         Rdict.destroy("tmp")
    fn snapshot(&self) -> PyResult<Snapshot> {
        Snapshot::new(self)
    }

    /// Loads a list of external SST files created with SstFileWriter
    /// into the current column family.
    ///
    /// Args:
    ///     paths: a list a paths
    ///     opts: IngestExternalFileOptionsPy instance
    #[pyo3(signature = (
        paths,
        opts = Python::with_gil(|py| Py::new(py, IngestExternalFileOptionsPy::new()).unwrap())
    ))]
    fn ingest_external_file(
        &self,
        paths: Vec<String>,
        opts: Py<IngestExternalFileOptionsPy>,
        py: Python,
    ) -> PyResult<()> {
        if let Some(db) = &self.db {
            let db = db.borrow();
            let ingest_result = if let Some(cf) = &self.column_family {
                db.ingest_external_file_cf_opts(cf.deref(), &opts.borrow(py).0, paths)
            } else {
                db.ingest_external_file_opts(&opts.borrow(py).0, paths)
            };
            match ingest_result {
                Ok(_) => Ok(()),
                Err(e) => Err(PyException::new_err(e.to_string())),
            }
        } else {
            Err(PyException::new_err("DB already closed"))
        }
    }

    /// Tries to catch up with the primary by reading as much as possible from the
    /// log files.
    pub fn try_catch_up_with_primary(&self) -> PyResult<()> {
        if let Some(db) = &self.db {
            let db = db.borrow();
            match db.try_catch_up_with_primary() {
                Ok(_) => Ok(()),
                Err(e) => Err(PyException::new_err(e.to_string())),
            }
        } else {
            Err(PyException::new_err("DB already closed"))
        }
    }

    /// Request stopping background work, if wait is true wait until it's done.
    pub fn cancel_all_background(&self, wait: bool) -> PyResult<()> {
        if let Some(db) = &self.db {
            db.borrow().cancel_all_background_work(wait);
            Ok(())
        } else {
            Err(PyException::new_err("DB already closed"))
        }
    }

    /// WriteBatch
    ///
    /// Notes:
    ///     This WriteBatch does not write to the current column family.
    ///
    /// Args:
    ///     write_batch: WriteBatch instance. This instance will be consumed.
    ///     write_opt: has default value.
    #[pyo3(signature = (write_batch, write_opt = WriteOptionsPy::new()))]
    pub fn write(&self, write_batch: &mut WriteBatchPy, write_opt: WriteOptionsPy) -> PyResult<()> {
        if let Some(db) = &self.db {
            if self.opt_py.raw_mode != write_batch.raw_mode {
                return if self.opt_py.raw_mode {
                    Err(PyException::new_err(
                        "must set raw_mode=True for WriteBatch",
                    ))
                } else {
                    Err(PyException::new_err(
                        "must set raw_mode=False for WriteBatch",
                    ))
                };
            }
            let db = db.borrow();
            match db.write_opt(write_batch.consume()?, &WriteOptions::from(&write_opt)) {
                Ok(_) => Ok(()),
                Err(e) => Err(PyException::new_err(e.to_string())),
            }
        } else {
            Err(PyException::new_err("DB already closed"))
        }
    }

    /// Removes the database entries in the range `["from", "to")` of the current column family.
    ///
    /// Args:
    ///     begin: included
    ///     end: excluded
    pub fn delete_range(&self, begin: &PyAny, end: &PyAny) -> PyResult<()> {
        if let Some(db) = &self.db {
            let db = db.borrow();
            let from = encode_key(begin, self.opt_py.raw_mode)?;
            let to = encode_key(end, self.opt_py.raw_mode)?;
            match &self.column_family {
                None => {
                    // manual implementation when there is no column
                    let mut r_opt = ReadOptions::default();
                    r_opt.set_iterate_lower_bound(from.to_vec());
                    r_opt.set_iterate_upper_bound(to.to_vec());
                    let mode = IteratorMode::From(&from, Direction::Forward);
                    let iter = db.iterator_opt(mode, r_opt);
                    for item in iter {
                        match item {
                            Ok((key, _)) => {
                                if let Err(e) = db.delete_opt(key, &self.write_opt) {
                                    return Err(PyException::new_err(e.to_string()));
                                }
                            }
                            Err(e) => {
                                return Err(PyException::new_err(e.to_string()));
                            }
                        }
                    }
                    Ok(())
                }
                Some(cf) => match db.delete_range_cf_opt(cf.deref(), from, to, &self.write_opt) {
                    Ok(_) => Ok(()),
                    Err(e) => Err(PyException::new_err(e.to_string())),
                },
            }
        } else {
            Err(PyException::new_err("DB already closed"))
        }
    }

    /// Flush memory to disk, and drop the current column family.
    ///
    /// Notes:
    ///     Calling `db.close()` is nearly equivalent to first calling
    ///     `db.flush()` and then `del db`. However, `db.close()` does
    ///     not guarantee the underlying RocksDB to be actually closed.
    ///     Other Column Family `Rdict` instances, `ColumnFamily`
    ///     (cf handle) instances, iterator instances such as`RdictIter`,
    ///     `RdictItems`, `RdictKeys`, `RdictValues` can all keep RocksDB
    ///     alive. `del` all associated instances mentioned above
    ///     to actually shut down RocksDB.
    ///
    fn close(&mut self) -> PyResult<()> {
        if let Some(db) = &self.db {
            let f_opt = &self.flush_opt;
            let db = db.borrow();
            let flush_result = if let Some(cf) = &self.column_family {
                db.flush_cf_opt(cf.deref(), &f_opt.into())
            } else {
                db.flush_opt(&f_opt.into())
            };
            drop(db);
            drop(self.column_family.take());
            drop(self.db.take());
            match flush_result {
                Ok(_) => Ok(()),
                Err(e) => Err(PyException::new_err(e.to_string())),
            }
        } else {
            Err(PyException::new_err("DB already closed"))
        }
    }

    /// Return current database path.
    fn path(&self) -> PyResult<String> {
        if let Some(db) = &self.db {
            let db = db.borrow();
            Ok(db.path().as_os_str().to_string_lossy().to_string())
        } else {
            Err(PyException::new_err("DB already closed"))
        }
    }

    /// Runs a manual compaction on the Range of keys given for the current Column Family.
    #[pyo3(signature = (begin, end, compact_opt = Python::with_gil(|py| Py::new(py, CompactOptionsPy::default()).unwrap())))]
    fn compact_range(
        &self,
        begin: &PyAny,
        end: &PyAny,
        compact_opt: Py<CompactOptionsPy>,
        py: Python,
    ) -> PyResult<()> {
        if let Some(db) = &self.db {
            let db = db.borrow();
            let from = if begin.is_none() {
                None
            } else {
                Some(encode_key(begin, self.opt_py.raw_mode)?)
            };
            let to = if end.is_none() {
                None
            } else {
                Some(encode_key(end, self.opt_py.raw_mode)?)
            };
            let opt = compact_opt.borrow(py);
            if let Some(cf) = &self.column_family {
                db.compact_range_cf_opt(cf.deref(), from, to, &opt.deref().0)
            } else {
                db.compact_range_opt(from, to, &opt.deref().0)
            };
            Ok(())
        } else {
            Err(PyException::new_err("DB already closed"))
        }
    }

    /// Set options for the current column family.
    fn set_options(&self, options: HashMap<String, String>) -> PyResult<()> {
        if let Some(db) = &self.db {
            let db = db.borrow();
            let options: Vec<(&str, &str)> = options
                .iter()
                .map(|(opt, v)| (opt.as_str(), v.as_str()))
                .collect();
            let set_opt_result = match &self.column_family {
                None => db.set_options(&options),
                Some(cf) => db.set_options_cf(cf.deref(), &options),
            };
            match set_opt_result {
                Ok(_) => Ok(()),
                Err(e) => Err(PyException::new_err(e.to_string())),
            }
        } else {
            Err(PyException::new_err("DB already closed"))
        }
    }

    /// Retrieves a RocksDB property by name, for the current column family.
    fn property_value(&self, name: &str) -> PyResult<Option<String>> {
        if let Some(db) = &self.db {
            let db = db.borrow();
            let result = match &self.column_family {
                None => db.property_value(name),
                Some(cf) => db.property_value_cf(cf.deref(), name),
            };
            match result {
                Ok(v) => Ok(v),
                Err(e) => Err(PyException::new_err(e.to_string())),
            }
        } else {
            Err(PyException::new_err("DB already closed"))
        }
    }

    /// Retrieves a RocksDB property and casts it to an integer
    /// (for the current column family).
    ///
    /// Full list of properties that return int values could be find
    /// [here](https://github.com/facebook/rocksdb/blob/08809f5e6cd9cc4bc3958dd4d59457ae78c76660/include/rocksdb/db.h#L654-L689).
    fn property_int_value(&self, name: &str) -> PyResult<Option<u64>> {
        if let Some(db) = &self.db {
            let db = db.borrow();
            let result = match &self.column_family {
                None => db.property_int_value(name),
                Some(cf) => db.property_int_value_cf(cf.deref(), name),
            };
            match result {
                Ok(v) => Ok(v),
                Err(e) => Err(PyException::new_err(e.to_string())),
            }
        } else {
            Err(PyException::new_err("DB already closed"))
        }
    }

    /// The sequence number of the most recent transaction.
    fn latest_sequence_number(&self) -> PyResult<u64> {
        if let Some(db) = &self.db {
            let db = db.borrow();
            Ok(db.latest_sequence_number())
        } else {
            Err(PyException::new_err("DB already closed"))
        }
    }

    /// Returns a list of all table files with their level, start key and end key
    fn live_files(&self, py: Python) -> PyResult<PyObject> {
        if let Some(db) = &self.db {
            let db = db.borrow();
            let lfs = db.live_files();
            match lfs {
                Ok(lfs) => {
                    let result = PyList::empty(py);
                    for lf in lfs {
                        result.append(display_live_file_dict(
                            lf,
                            py,
                            &self.pickle_loads,
                            self.opt_py.raw_mode,
                        )?)?
                    }
                    Ok(result.to_object(py))
                }
                Err(e) => Err(PyException::new_err(e.to_string())),
            }
        } else {
            Err(PyException::new_err("DB already closed"))
        }
    }

    /// Delete the database.
    ///
    /// Args:
    ///     path (str): path to this database
    ///     options (rocksdict.Options): Rocksdb options object
    #[staticmethod]
    #[pyo3(signature = (path, options = OptionsPy::new(false)))]
    fn destroy(path: &str, options: OptionsPy) -> PyResult<()> {
        fs::remove_file(config_file(path)).ok();
        match DB::destroy(&options.inner_opt, path) {
            Ok(_) => Ok(()),
            Err(e) => Err(PyException::new_err(e.to_string())),
        }
    }

    /// Repair the database.
    ///
    /// Args:
    ///     path (str): path to this database
    ///     options (rocksdict.Options): Rocksdb options object
    #[staticmethod]
    #[pyo3(signature = (path, options = OptionsPy::new(false)))]
    fn repair(path: &str, options: OptionsPy) -> PyResult<()> {
        match DB::repair(&options.inner_opt, path) {
            Ok(_) => Ok(()),
            Err(e) => Err(PyException::new_err(e.to_string())),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (path, options = OptionsPy::new(false)))]
    fn list_cf(path: &str, options: OptionsPy) -> PyResult<Vec<String>> {
        match DB::list_cf(&options.inner_opt, path) {
            Ok(vec) => Ok(vec),
            Err(e) => Err(PyException::new_err(e.to_string())),
        }
    }
}

fn display_live_file_dict(
    lf: LiveFile,
    py: Python,
    pickle_loads: &PyObject,
    raw_mode: bool,
) -> PyResult<PyObject> {
    let result = PyDict::new(py);
    let start_key = match lf.start_key {
        None => py.None(),
        Some(k) => decode_value(py, &k, pickle_loads, raw_mode)?,
    };
    let end_key = match lf.end_key {
        None => py.None(),
        Some(k) => decode_value(py, &k, pickle_loads, raw_mode)?,
    };
    result.set_item("name", lf.name)?;
    result.set_item("size", lf.size)?;
    result.set_item("level", lf.level)?;
    result.set_item("start_key", start_key)?;
    result.set_item("end_key", end_key)?;
    result.set_item("num_entries", lf.num_entries)?;
    result.set_item("num_deletions", lf.num_deletions)?;
    Ok(result.to_object(py))
}

#[inline(always)]
fn get_batch_inner<'a>(
    db: &RefCell<DB>,
    keys: &'a PyList,
    py: Python<'a>,
    read_opt: &ReadOptions,
    pickle_loads: &PyObject,
    column_family: &Option<Arc<ColumnFamily>>,
    raw_mode: bool,
) -> PyResult<&'a PyList> {
    let db = db.borrow();
    let values = if raw_mode {
        if let Some(cf) = column_family {
            let mut keys_cols: Vec<(&ColumnFamily, &[u8])> = Vec::with_capacity(keys.len());
            for key in keys {
                keys_cols.push((cf.deref(), encode_raw(key)?));
            }
            db.multi_get_cf_opt(keys_cols, read_opt)
        } else {
            let mut keys_batch = Vec::with_capacity(keys.len());
            for key in keys {
                keys_batch.push(encode_raw(key)?);
            }
            db.multi_get_opt(keys_batch, read_opt)
        }
    } else if let Some(cf) = column_family {
        let mut keys_cols: Vec<(&ColumnFamily, Box<[u8]>)> = Vec::with_capacity(keys.len());
        for key in keys {
            keys_cols.push((cf.deref(), encode_key(key, raw_mode)?));
        }
        db.multi_get_cf_opt(keys_cols, read_opt)
    } else {
        let mut keys_batch = Vec::with_capacity(keys.len());
        for key in keys {
            keys_batch.push(encode_key(key, raw_mode)?);
        }
        db.multi_get_opt(keys_batch, read_opt)
    };
    let result = PyList::empty(py);
    for v in values {
        match v {
            Ok(value) => match value {
                None => result.append(py.None())?,
                Some(slice) => {
                    result.append(decode_value(py, slice.as_ref(), pickle_loads, raw_mode)?)?
                }
            },
            Err(e) => return Err(PyException::new_err(e.to_string())),
        }
    }
    Ok(result)
}

impl Drop for Rdict {
    // flush
    fn drop(&mut self) {
        if let Some(db) = &self.db {
            let f_opt = &self.flush_opt;
            let db = db.borrow();
            let _ = if let Some(cf) = &self.column_family {
                db.flush_cf_opt(cf.deref(), &f_opt.into())
            } else {
                db.flush_opt(&f_opt.into())
            };
        }
        // important, always drop column families first
        // to ensure that CF handles have shorter life than DB.
        drop(self.column_family.take());
        drop(self.db.take());
    }
}

unsafe impl Send for Rdict {}

/// Column family handle. This can be used in WriteBatch to specify Column Family.
#[pyclass(name = "ColumnFamily")]
#[allow(dead_code)]
#[derive(Clone)]
pub(crate) struct ColumnFamilyPy {
    // must follow this drop order
    pub(crate) cf: Arc<ColumnFamily>,
    // must keep db alive
    db: Arc<RefCell<DB>>,
}

unsafe impl Send for ColumnFamilyPy {}

#[pymethods]
impl AccessType {
    /// Define DB Access Types.
    ///
    /// Notes:
    ///     There are four access types:
    ///      - ReadWrite: default value
    ///      - ReadOnly
    ///      - WithTTL
    ///      - Secondary
    ///
    /// Examples:
    ///     ::
    ///
    ///         from rocksdict import Rdict, AccessType
    ///
    ///         # open with 24 hours ttl
    ///         db = Rdict("./main_path", access_type = AccessType.with_ttl(24 * 3600))
    ///
    ///         # open as read_only
    ///         db = Rdict("./main_path", access_type = AccessType.read_only())
    ///
    ///         # open as secondary
    ///         db = Rdict("./main_path", access_type = AccessType.secondary("./secondary_path"))
    ///
    ///
    #[staticmethod]
    fn read_write() -> Self {
        AccessType(AccessTypeInner::ReadWrite)
    }

    /// Define DB Access Types.
    ///
    /// Notes:
    ///     There are four access types:
    ///       - ReadWrite: default value
    ///       - ReadOnly
    ///       - WithTTL
    ///       - Secondary
    ///
    /// Examples:
    ///     ::
    ///
    ///         from rocksdict import Rdict, AccessType
    ///
    ///         # open with 24 hours ttl
    ///         db = Rdict("./main_path", access_type = AccessType.with_ttl(24 * 3600))
    ///
    ///         # open as read_only
    ///         db = Rdict("./main_path", access_type = AccessType.read_only())
    ///
    ///         # open as secondary
    ///         db = Rdict("./main_path", access_type = AccessType.secondary("./secondary_path"))
    ///
    ///
    #[staticmethod]
    #[pyo3(signature = (error_if_log_file_exist = true))]
    fn read_only(error_if_log_file_exist: bool) -> Self {
        AccessType(AccessTypeInner::ReadOnly {
            error_if_log_file_exist,
        })
    }

    /// Define DB Access Types.
    ///
    /// Notes:
    ///     There are four access types:
    ///      - ReadWrite: default value
    ///      - ReadOnly
    ///      - WithTTL
    ///      - Secondary
    ///
    /// Examples:
    ///     ::
    ///
    ///         from rocksdict import Rdict, AccessType
    ///
    ///         # open with 24 hours ttl
    ///         db = Rdict("./main_path", access_type = AccessType.with_ttl(24 * 3600))
    ///
    ///         # open as read_only
    ///         db = Rdict("./main_path", access_type = AccessType.read_only())
    ///
    ///         # open as secondary
    ///         db = Rdict("./main_path", access_type = AccessType.secondary("./secondary_path"))
    ///
    ///
    #[staticmethod]
    fn secondary(secondary_path: String) -> Self {
        AccessType(AccessTypeInner::Secondary { secondary_path })
    }

    /// Define DB Access Types.
    ///
    /// Notes:
    ///     There are four access types:
    ///      - ReadWrite: default value
    ///      - ReadOnly
    ///      - WithTTL
    ///      - Secondary
    ///
    /// Examples:
    ///     ::
    ///
    ///         from rocksdict import Rdict, AccessType
    ///
    ///         # open with 24 hours ttl
    ///         db = Rdict("./main_path", access_type = AccessType.with_ttl(24 * 3600))
    ///
    ///         # open as read_only
    ///         db = Rdict("./main_path", access_type = AccessType.read_only())
    ///
    ///         # open as secondary
    ///         db = Rdict("./main_path", access_type = AccessType.secondary("./secondary_path"))
    ///
    ///
    #[staticmethod]
    fn with_ttl(duration: u64) -> Self {
        AccessType(AccessTypeInner::WithTTL {
            ttl: Duration::from_secs(duration),
        })
    }
}

#[derive(Clone)]
enum AccessTypeInner {
    ReadWrite,
    ReadOnly { error_if_log_file_exist: bool },
    Secondary { secondary_path: String },
    WithTTL { ttl: Duration },
}
