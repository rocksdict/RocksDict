use crate::db_reference::{DbReference, DbReferenceHolder};
use crate::encoder::{decode_value, encode_key, encode_value};
use crate::exceptions::DbClosedError;
use crate::iter::{RdictItems, RdictKeys, RdictValues};
use crate::options::{CachePy, EnvPy, SliceTransformType};
use crate::{
    CompactOptionsPy, FlushOptionsPy, IngestExternalFileOptionsPy, OptionsPy, RdictColumns,
    RdictEntities, RdictIter, ReadOptionsPy, Snapshot, WriteBatchPy, WriteOptionsPy,
};
use pyo3::exceptions::{PyException, PyKeyError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use rocksdb::{
    ColumnFamilyDescriptor, FlushOptions, Iterable as _, LiveFile, ReadOptions,
    UnboundColumnFamily, WriteOptions, DEFAULT_COLUMN_FAMILY_NAME,
};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
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

type DB = rocksdb::DBWithThreadMode<rocksdb::MultiThreaded>;

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
/// Opening DB created by other language is easy:
/// you don't need to manually configure Options and Column
/// Families. Just use `db = Rdict("./db_path")`.
/// It will automatically open the db in right Options and
/// Column Families for you in RAW MODE.
///
/// Args:
///     path (str): path to the database
///     options (Options): Options object
///     column_families (dict): (name, options) pairs, these `Options`
///         must have the same `raw_mode` argument as the main `Options`.
///         A column family called 'default' is always created.
///     access_type (AccessType): there are four access types:
///         ReadWrite, ReadOnly, WithTTL, and Secondary, use
///         AccessType class to create.
#[pyclass(name = "Rdict")]
pub(crate) struct Rdict {
    pub(crate) write_opt: WriteOptions,
    pub(crate) flush_opt: FlushOptionsPy,
    pub(crate) read_opt: ReadOptions,
    pub(crate) loads: PyObject,
    pub(crate) dumps: PyObject,
    pub(crate) write_opt_py: WriteOptionsPy,
    pub(crate) read_opt_py: ReadOptionsPy,
    pub(crate) column_family: Option<Arc<UnboundColumnFamily>>,
    pub(crate) opt_py: OptionsPy,
    pub(crate) access_type: AccessType,
    pub(crate) slice_transforms: Arc<RwLock<HashMap<String, SliceTransformType>>>,
    // drop DB last
    pub(crate) db: DbReferenceHolder,
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

impl Default for RocksDictConfig {
    fn default() -> Self {
        Self {
            raw_mode: true,
            prefix_extractors: Default::default(),
        }
    }
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
        let config_file = fs::File::options()
            .create(true)
            .truncate(true)
            .write(true)
            .open(path)?;
        match serde_json::to_writer(config_file, self) {
            Ok(_) => Ok(()),
            Err(e) => Err(PyException::new_err(e.to_string())),
        }
    }

    pub fn save_to_dir(&self, dir: &str) -> PyResult<()> {
        self.save(config_file(dir))
    }
}

impl Rdict {
    #[inline]
    pub(crate) fn config(&self) -> RocksDictConfig {
        RocksDictConfig {
            raw_mode: self.opt_py.raw_mode,
            prefix_extractors: self.slice_transforms.read().unwrap().clone(),
        }
    }

    fn dump_config(&self) -> PyResult<()> {
        self.config().save_to_dir(&self.path()?)
    }

    #[inline]
    pub(crate) fn get_db(&self) -> PyResult<&DbReference> {
        self.db
            .get()
            .ok_or_else(|| DbClosedError::new_err("DB instance already closed"))
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
        let pickle = PyModule::import(py, "pickle")?;
        // create db path if missing
        fs::create_dir_all(path).map_err(|e| PyException::new_err(e.to_string()))?;
        // load options
        let options_loaded = OptionsPy::load_latest_inner(
            path,
            EnvPy::default()?,
            false,
            CachePy::new_lru_cache(DEFAULT_LRU_CACHE_SIZE),
        );
        // prioritize passed options over loaded options
        let (options, column_families) = match (options_loaded, options, column_families) {
            (Ok((opt_loaded, cols_loaded)), opt, cols) => match (opt, cols) {
                (Some(opt), Some(cols)) => (opt, Some(cols)),
                (Some(opt), None) => (opt, Some(cols_loaded)),
                (None, Some(cols)) => (opt_loaded, Some(cols)),
                (None, None) => (opt_loaded, Some(cols_loaded)),
            },
            (Err(_), Some(opt), cols) => (opt, cols),
            (Err(_), None, cols) => (OptionsPy::new(false), cols),
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
        rocksdict_config.save(config_path)?;
        let opt_inner = &options.inner_opt;
        // define column families
        let cfs = match column_families {
            None => {
                vec![ColumnFamilyDescriptor::new(
                    DEFAULT_COLUMN_FAMILY_NAME,
                    opt_inner.clone(),
                )]
            }
            Some(mut cf) => {
                let mut has_default_cf = false;
                // check options_raw_mode for column families
                for (cf_name, cf_opt) in cf.iter_mut() {
                    cf_opt.raw_mode = options.raw_mode;
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
                cfs
            }
        };
        // open db
        let db = match &access_type.0 {
            AccessTypeInner::ReadWrite => DB::open_cf_descriptors(opt_inner, path, cfs),
            AccessTypeInner::ReadOnly {
                error_if_log_file_exist,
            } => DB::open_cf_descriptors_read_only(opt_inner, path, cfs, *error_if_log_file_exist),
            AccessTypeInner::Secondary { secondary_path } => {
                DB::open_cf_descriptors_as_secondary(opt_inner, path, secondary_path, cfs)
            }
            AccessTypeInner::WithTTL { ttl } => {
                DB::open_cf_descriptors_with_ttl(opt_inner, path, cfs, *ttl)
            }
        }
        .map_err(|e| PyException::new_err(e.to_string()))?;
        let r_opt = ReadOptionsPy::default(py)?;
        let w_opt = WriteOptionsPy::new();
        Ok(Rdict {
            db: DbReferenceHolder::new(db),
            write_opt: (&w_opt).into(),
            flush_opt: FlushOptionsPy::new(),
            read_opt: r_opt.to_read_options(options.raw_mode, py)?,
            loads: pickle.getattr("loads")?.unbind(),
            dumps: pickle.getattr("dumps")?.unbind(),
            write_opt_py: w_opt,
            read_opt_py: r_opt,
            column_family: None,
            opt_py: options.clone(),
            access_type,
            slice_transforms: Arc::new(RwLock::new(prefix_extractors)),
        })
    }

    /// set custom dumps function
    fn set_dumps(&mut self, dumps: PyObject) {
        self.dumps = dumps
    }

    /// set custom loads function
    fn set_loads(&mut self, loads: PyObject) {
        self.loads = loads
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
    fn set_read_options(&mut self, read_opt: &ReadOptionsPy, py: Python) -> PyResult<()> {
        self.read_opt = read_opt.to_read_options(self.opt_py.raw_mode, py)?;
        self.read_opt_py = read_opt.clone();
        Ok(())
    }

    /// Use list of keys for batch get.
    fn __getitem__<'py>(&self, key: &Bound<PyAny>, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match self.get(key, None, None, py) {
            Ok(v) => {
                if v.is_none() {
                    Err(PyKeyError::new_err(format!("key {key} not found")))
                } else {
                    Ok(v)
                }
            }
            Err(e) => Err(e),
        }
    }

    /// Get value from key or a list of keys.
    ///
    /// Args:
    ///     key: a single key or list of keys.
    ///     default: the default value to return if key not found.
    ///     read_opt: override preset read options
    ///         (or use Rdict.set_read_options to preset a read options used by default).
    ///
    /// Returns:
    ///    None or default value if the key does not exist.
    #[inline]
    #[pyo3(signature = (key, default = None, read_opt = None))]
    fn get<'py>(
        &self,
        key: &Bound<PyAny>,
        default: Option<Bound<'py, PyAny>>,
        read_opt: Option<&ReadOptionsPy>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let db = self.get_db()?;
        let read_opt_option = match read_opt {
            None => None,
            Some(opt) => Some(opt.to_read_options(self.opt_py.raw_mode, py)?),
        };
        let read_opt = match &read_opt_option {
            None => &self.read_opt,
            Some(opt) => opt,
        };
        let cf = match &self.column_family {
            None => {
                self.get_column_family_handle(DEFAULT_COLUMN_FAMILY_NAME)?
                    .cf
            }
            Some(cf) => cf.clone(),
        };
        if let Ok(keys) = key.downcast() {
            return Ok(get_batch_inner(
                db,
                keys,
                py,
                read_opt,
                &self.loads,
                &cf,
                self.opt_py.raw_mode,
            )?
            .into_any());
        }
        let key_bytes = encode_key(key, self.opt_py.raw_mode)?;
        let value_result = db
            .get_pinned_cf_opt(&cf, key_bytes, read_opt)
            .map_err(|e| PyException::new_err(e.to_string()))?;
        match value_result {
            None => {
                // try to return default value
                if let Some(default) = default {
                    Ok(default)
                } else {
                    Ok(py.None().bind(py).to_owned())
                }
            }
            Some(slice) => Ok(decode_value(
                py,
                slice.as_ref(),
                &self.loads,
                self.opt_py.raw_mode,
            )?),
        }
    }

    /// Get a wide-column from a key.
    ///
    /// Args:
    ///     key: a single key or list of keys.
    ///     default: the default value to return if key not found.
    ///     read_opt: override preset read options
    ///         (or use Rdict.set_read_options to preset a read options used by default).
    ///
    /// Returns:
    ///    A list of `(name, value)` tuples.
    ///    If the value is not an entity, returns a single-column
    ///    with default column name (empty bytes/string).
    ///    None or default value if the key does not exist.
    #[inline]
    #[pyo3(signature = (key, default = None, read_opt = None))]
    fn get_entity<'py>(
        &self,
        key: &Bound<PyAny>,
        default: Option<Bound<'py, PyAny>>,
        read_opt: Option<&ReadOptionsPy>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let db = self.get_db()?;
        let read_opt_option = match read_opt {
            None => None,
            Some(opt) => Some(opt.to_read_options(self.opt_py.raw_mode, py)?),
        };
        let read_opt = match &read_opt_option {
            None => &self.read_opt,
            Some(opt) => opt,
        };
        let cf = match &self.column_family {
            None => {
                self.get_column_family_handle(DEFAULT_COLUMN_FAMILY_NAME)?
                    .cf
            }
            Some(cf) => cf.clone(),
        };
        let key_bytes = encode_key(key, self.opt_py.raw_mode)?;
        let column_result = db
            .get_entity_cf_opt(&cf, key_bytes, read_opt)
            .map_err(|e| PyException::new_err(e.to_string()))?;
        match column_result {
            None => {
                // try to return default value
                if let Some(default) = default {
                    Ok(default)
                } else {
                    Ok(py.None().bind(py).to_owned())
                }
            }
            Some(columns) => {
                let result = PyList::empty(py);
                for column in columns.iter() {
                    let name = decode_value(py, column.name, &self.loads, self.opt_py.raw_mode)?;
                    let value = decode_value(py, column.value, &self.loads, self.opt_py.raw_mode)?;
                    result.append(PyTuple::new(py, [name, value])?)?;
                }
                Ok(result.into_any())
            }
        }
    }

    fn __setitem__(&self, key: &Bound<PyAny>, value: &Bound<PyAny>) -> PyResult<()> {
        self.put(key, value, None)
    }

    /// Insert key value into database.
    ///
    /// Args:
    ///     key: the key.
    ///     value: the value.
    ///     write_opt: override preset write options
    ///         (or use Rdict.set_write_options to preset a write options used by default).
    #[inline]
    #[pyo3(signature = (key, value, write_opt = None))]
    fn put(
        &self,
        key: &Bound<PyAny>,
        value: &Bound<PyAny>,
        write_opt: Option<&WriteOptionsPy>,
    ) -> PyResult<()> {
        let db = self.get_db()?;
        let key = encode_key(key, self.opt_py.raw_mode)?;
        let value = encode_value(value, &self.dumps, self.opt_py.raw_mode)?;
        let write_opt_option = write_opt.map(WriteOptions::from);
        let write_opt = match &write_opt_option {
            None => &self.write_opt,
            Some(opt) => opt,
        };
        if let Some(cf) = &self.column_family {
            db.put_cf_opt(cf, key, value, write_opt)
        } else {
            db.put_opt(key, value, write_opt)
        }
        .map_err(|e| PyException::new_err(e.to_string()))
    }

    /// Insert a wide-column.
    ///
    /// The length of `names` and `values` must be the same.
    ///
    /// Args:
    ///     key: the key.
    ///     names: the names of the columns.
    ///     values: the values of the columns.
    ///     write_opt: override preset write options
    ///         (or use Rdict.set_write_options to preset a write options used by default).
    #[inline]
    #[pyo3(signature = (key, names, values, write_opt = None))]
    fn put_entity(
        &self,
        key: &Bound<PyAny>,
        names: Vec<Bound<PyAny>>,
        values: Vec<Bound<PyAny>>,
        write_opt: Option<&WriteOptionsPy>,
    ) -> PyResult<()> {
        let db = self.get_db()?;
        let key = encode_key(key, self.opt_py.raw_mode)?;
        let write_opt_option = write_opt.map(WriteOptions::from);
        let write_opt = match &write_opt_option {
            None => &self.write_opt,
            Some(opt) => opt,
        };
        let cf = match &self.column_family {
            None => {
                self.get_column_family_handle(DEFAULT_COLUMN_FAMILY_NAME)?
                    .cf
            }
            Some(cf) => cf.clone(),
        };
        if names.len() != values.len() {
            return Err(PyException::new_err(
                "names and values must have the same length",
            ));
        }
        let mut names_vec = Vec::with_capacity(names.len());
        let mut values_vec = Vec::with_capacity(values.len());
        for name in names.iter() {
            names_vec.push(encode_value(name, &self.dumps, self.opt_py.raw_mode)?);
        }
        for value in values.iter() {
            values_vec.push(encode_value(value, &self.dumps, self.opt_py.raw_mode)?);
        }
        db.put_entity_cf_opt(&cf, key, &names_vec, &values_vec, write_opt)
            .map_err(|e| PyException::new_err(e.to_string()))
    }

    fn __contains__(&self, key: &Bound<PyAny>) -> PyResult<bool> {
        let db = self.get_db()?;
        let key = encode_key(key, self.opt_py.raw_mode)?;
        let may_exist = if let Some(cf) = &self.column_family {
            db.key_may_exist_cf_opt(cf, &key[..], &self.read_opt)
        } else {
            db.key_may_exist_opt(&key[..], &self.read_opt)
        };
        if may_exist {
            let value_result = if let Some(cf) = &self.column_family {
                db.get_pinned_cf_opt(cf, key, &self.read_opt)
            } else {
                db.get_pinned_opt(key, &self.read_opt)
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
    }

    /// Check if a key may exist without doing any IO.
    ///
    /// Notes:
    ///     If the key definitely does not exist in the database,
    ///     then this method returns False, else True.
    ///     If the caller wants to obtain value when the key is found in memory,
    ///     fetch should be set to True.
    ///     This check is potentially lighter-weight than invoking DB::get().
    ///     One way to make this lighter weight is to avoid doing any IOs.
    ///
    ///     The API follows the following principle:
    ///       - True, and value found => the key must exist.
    ///       - True => the key may or may not exist.
    ///       - False => the key definitely does not exist.
    ///
    ///     Flip it around:
    ///       - key exists => must return True, but value may or may not be found.
    ///       - key doesn't exists => might still return True.
    ///
    /// Args:
    ///     key: Key to check
    ///     read_opt: ReadOptions
    ///
    /// Returns:
    ///     if `fetch = False`,
    ///         returning True implies that the key may exist.
    ///         returning False implies that the key definitely does not exist.
    ///     if `fetch = True`,
    ///         returning (True, value) implies that the key is found and definitely exist.
    ///         returning (False, None) implies that the key definitely does not exist.
    ///         returning (True,  None) implies that the key may exist.
    #[pyo3(signature = (key, fetch = false, read_opt = None))]
    fn key_may_exist<'py>(
        &self,
        key: &Bound<PyAny>,
        fetch: bool,
        read_opt: Option<&ReadOptionsPy>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let db = self.get_db()?;
        let key = encode_key(key, self.opt_py.raw_mode)?;
        let read_opt_option = match read_opt {
            None => None,
            Some(opt) => Some(opt.to_read_options(self.opt_py.raw_mode, py)?),
        };
        let read_opt = match &read_opt_option {
            None => &self.read_opt,
            Some(opt) => opt,
        };
        let cf = match &self.column_family {
            None => {
                self.get_column_family_handle(DEFAULT_COLUMN_FAMILY_NAME)?
                    .cf
            }
            Some(cf) => cf.clone(),
        };
        if !fetch {
            Ok(db
                .key_may_exist_cf_opt(&cf, &key[..], read_opt)
                .into_pyobject(py)?
                .to_owned()
                .into_any())
        } else {
            let (may, value) = db.key_may_exist_cf_opt_value(&cf, &key[..], read_opt);
            match value {
                None => Ok((may, py.None()).into_pyobject(py)?.into_any()),
                Some(dat) => Ok((
                    may,
                    decode_value(py, dat.as_ref(), &self.loads, self.opt_py.raw_mode)?,
                )
                    .into_pyobject(py)?
                    .into_any()),
            }
        }
    }

    fn __delitem__(&self, key: &Bound<PyAny>) -> PyResult<()> {
        self.delete(key, None)
    }

    /// Delete entry from the database.
    ///
    /// Args:
    ///     key: the key.
    ///     write_opt: override preset write options
    ///         (or use Rdict.set_write_options to preset a write options used by default).
    #[inline]
    #[pyo3(signature = (key, write_opt = None))]
    fn delete(&self, key: &Bound<PyAny>, write_opt: Option<&WriteOptionsPy>) -> PyResult<()> {
        let db = self.get_db()?;
        let key = encode_key(key, self.opt_py.raw_mode)?;

        let write_opt_option = write_opt.map(WriteOptions::from);
        let write_opt = match &write_opt_option {
            None => &self.write_opt,
            Some(opt) => opt,
        };
        if let Some(cf) = &self.column_family {
            db.delete_cf_opt(cf, key, write_opt)
        } else {
            db.delete_opt(key, write_opt)
        }
        .map_err(|e| PyException::new_err(e.to_string()))
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
    ///     read_opt: ReadOptions
    ///
    /// Returns: Reversible
    #[pyo3(signature = (read_opt = None))]
    fn iter(&self, read_opt: Option<&ReadOptionsPy>, py: Python) -> PyResult<RdictIter> {
        let read_opt: ReadOptionsPy = match read_opt {
            None => ReadOptionsPy::default(py)?,
            Some(opt) => opt.clone(),
        };

        RdictIter::new(
            &self.db,
            &self.column_family,
            read_opt,
            &self.loads,
            self.opt_py.raw_mode,
            py,
        )
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
    ///     read_opt: ReadOptions
    #[pyo3(signature = (backwards = false, from_key = None, read_opt = None))]
    fn items(
        &self,
        backwards: bool,
        from_key: Option<&Bound<PyAny>>,
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
    ///     read_opt: ReadOptions
    #[pyo3(signature = (backwards = false, from_key = None, read_opt = None))]
    fn keys(
        &self,
        backwards: bool,
        from_key: Option<&Bound<PyAny>>,
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
        from_key: Option<&Bound<PyAny>>,
        read_opt: Option<&ReadOptionsPy>,
        py: Python,
    ) -> PyResult<RdictValues> {
        RdictValues::new(self.iter(read_opt, py)?, backwards, from_key)
    }

    /// Iterate through all values as widecolumns
    ///
    /// Examples:
    ///     ::
    ///
    ///         all_entities = [v for v in db.columns()]
    ///
    /// Args:
    ///     backwards: iteration direction, forward if `False`.
    ///     from_key: iterate from key, first seek to this key
    ///         or the nearest next key for iteration
    ///         (depending on iteration direction).
    ///     read_opt: ReadOptions, must have the same `raw_mode` argument.
    #[pyo3(signature = (backwards = false, from_key = None, read_opt = None))]
    fn columns(
        &self,
        backwards: bool,
        from_key: Option<&Bound<PyAny>>,
        read_opt: Option<&ReadOptionsPy>,
        py: Python,
    ) -> PyResult<RdictColumns> {
        RdictColumns::new(self.iter(read_opt, py)?, backwards, from_key)
    }

    /// Iterate through all keys and entities pairs.
    ///
    /// Examples:
    ///     ::
    ///
    ///         for k, v in db.entities():
    ///             print(f"{k} -> {v}")
    ///
    /// Args:
    ///     backwards: iteration direction, forward if `False`.
    ///     from_key: iterate from key, first seek to this key
    ///         or the nearest next key for iteration
    ///         (depending on iteration direction).
    ///     read_opt: ReadOptions
    #[pyo3(signature = (backwards = false, from_key = None, read_opt = None))]
    fn entities(
        &self,
        backwards: bool,
        from_key: Option<&Bound<PyAny>>,
        read_opt: Option<&ReadOptionsPy>,
        py: Python,
    ) -> PyResult<RdictEntities> {
        RdictEntities::new(self.iter(read_opt, py)?, backwards, from_key)
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
    fn flush(&self, wait: bool, py: Python) -> PyResult<()> {
        let db = self.get_db()?;

        py.allow_threads(|| {
            let mut f_opt = FlushOptions::new();
            f_opt.set_wait(wait);
            if let Some(cf) = &self.column_family {
                db.flush_cf_opt(cf, &f_opt)
            } else {
                db.flush_opt(&f_opt)
            }
        })
        .map_err(|e| PyException::new_err(e.into_string()))
    }

    /// Flushes the WAL buffer. If `sync` is set to `true`, also syncs
    /// the data to disk.
    #[pyo3(signature = (sync = true))]
    fn flush_wal(&self, sync: bool, py: Python) -> PyResult<()> {
        let db = self.get_db()?;
        py.allow_threads(|| db.flush_wal(sync))
            .map_err(|e| PyException::new_err(e.into_string()))
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
    fn create_column_family(&self, name: &str, options: OptionsPy, py: Python) -> PyResult<Rdict> {
        let db = self.get_db()?;
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
        db.create_cf(name, &options.inner_opt)
            .map_err(|e| PyException::new_err(e.to_string()))?;
        self.get_column_family(name, py)
    }

    /// Drops the column family with the given name
    fn drop_column_family(&self, name: &str) -> PyResult<()> {
        let db = self.get_db()?;
        db.drop_cf(name)
            .map_err(|e| PyException::new_err(e.to_string()))
    }

    /// Get a column family Rdict
    ///
    /// Args:
    ///     name: name of this column family
    ///     options: Rdict Options for this column family
    ///
    /// Return:
    ///     the column family Rdict of this name
    pub fn get_column_family(&self, name: &str, py: Python) -> PyResult<Self> {
        let db = self.get_db()?;
        match unsafe { db.cf_handle_unbounded(name) } {
            None => Err(PyException::new_err(format!(
                "column name `{name}` does not exist, use `create_cf` to creat it",
            ))),
            Some(cf) => Ok(Self {
                db: self.db.clone(),
                write_opt: (&self.write_opt_py).into(),
                flush_opt: self.flush_opt,
                read_opt: self.read_opt_py.to_read_options(self.opt_py.raw_mode, py)?,
                loads: self.loads.clone(),
                dumps: self.dumps.clone(),
                column_family: Some(cf),
                write_opt_py: self.write_opt_py.clone(),
                read_opt_py: self.read_opt_py.clone(),
                opt_py: self.opt_py.clone(),
                access_type: self.access_type.clone(),
                slice_transforms: self.slice_transforms.clone(),
            }),
        }
    }

    /// Use this method to obtain a ColumnFamily instance, which can be used in WriteBatch.
    ///
    /// The name of the default column family name is `"default"`.
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
        let db = self.get_db()?;
        match unsafe { db.cf_handle_unbounded(name) } {
            None => Err(PyException::new_err(format!(
                "column name `{name}` does not exist, use `create_cf` to creat it",
            ))),
            Some(cf) => Ok(ColumnFamilyPy {
                cf,
                db: self.db.clone(),
            }),
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
    fn snapshot(&self, py: Python) -> PyResult<Snapshot> {
        Snapshot::new(self, py)
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
        let db = self.get_db()?;
        let opts = &opts.borrow(py).0;
        if let Some(cf) = &self.column_family {
            db.ingest_external_file_cf_opts(cf, opts, paths)
        } else {
            db.ingest_external_file_opts(opts, paths)
        }
        .map_err(|e| PyException::new_err(e.to_string()))
    }

    /// Tries to catch up with the primary by reading as much as possible from the
    /// log files.
    pub fn try_catch_up_with_primary(&self) -> PyResult<()> {
        let db = self.get_db()?;
        db.try_catch_up_with_primary()
            .map_err(|e| PyException::new_err(e.to_string()))
    }

    /// Request stopping background work, if wait is true wait until it's done.
    pub fn cancel_all_background(&self, wait: bool) -> PyResult<()> {
        let db = self.get_db()?;
        db.cancel_all_background_work(wait);
        Ok(())
    }

    /// WriteBatch
    ///
    /// Notes:
    ///     This WriteBatch does not write to the current column family.
    ///
    /// Args:
    ///     write_batch: WriteBatch instance. This instance will be consumed.
    ///     write_opt: use default value if not provided.
    #[pyo3(signature = (write_batch, write_opt = None))]
    pub fn write(
        &self,
        write_batch: &mut WriteBatchPy,
        write_opt: Option<&WriteOptionsPy>,
    ) -> PyResult<()> {
        let db = self.get_db()?;
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
        let write_opt_option = write_opt.map(WriteOptions::from);
        let write_opt = match &write_opt_option {
            None => &self.write_opt,
            Some(opt) => opt,
        };
        db.write_opt(write_batch.consume()?, write_opt)
            .map_err(|e| PyException::new_err(e.to_string()))
    }

    /// Removes the database entries in the range `["from", "to")` of the current column family.
    ///
    /// Args:
    ///     begin: included
    ///     end: excluded
    ///     write_opt: WriteOptions
    #[pyo3(signature = (begin, end, write_opt=None))]
    pub fn delete_range(
        &self,
        begin: &Bound<PyAny>,
        end: &Bound<PyAny>,
        write_opt: Option<&WriteOptionsPy>,
    ) -> PyResult<()> {
        let db = self.get_db()?;
        let from = encode_key(begin, self.opt_py.raw_mode)?;
        let to = encode_key(end, self.opt_py.raw_mode)?;
        let cf = match &self.column_family {
            None => {
                self.get_column_family_handle(DEFAULT_COLUMN_FAMILY_NAME)?
                    .cf
            }
            Some(cf) => cf.clone(),
        };
        let write_opt_option = write_opt.map(WriteOptions::from);
        let write_opt = match &write_opt_option {
            None => &self.write_opt,
            Some(opt) => opt,
        };
        db.delete_range_cf_opt(&cf, from, to, write_opt)
            .map_err(|e| PyException::new_err(e.to_string()))
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
    ///     alive. `del` or `close` all associated instances mentioned
    ///     above to actually shut down RocksDB.
    ///
    fn close(&mut self, py: Python) -> PyResult<()> {
        // do not flush if readonly
        if let AccessTypeInner::ReadOnly { .. } | AccessTypeInner::Secondary { .. } =
            &self.access_type.0
        {
            py.allow_threads(|| {
                drop(self.column_family.take());
                self.db.close();
            });
            return Ok(());
        }

        let (flush_wal_result, flush_result) = py.allow_threads(|| {
            let f_opt = &self.flush_opt;
            let db = self.get_db()?;

            let flush_wal_result = db.flush_wal(true);
            let flush_result = if let Some(cf) = &self.column_family {
                db.flush_cf_opt(cf, &f_opt.into())
            } else {
                db.flush_opt(&f_opt.into())
            };
            drop(self.column_family.take());
            self.db.close();

            Ok::<_, PyErr>((flush_wal_result, flush_result))
        })?;
        match (flush_result, flush_wal_result) {
            (Ok(_), Ok(_)) => Ok(()),
            (Err(e), Ok(_)) => Err(PyException::new_err(e.to_string())),
            (Ok(_), Err(e)) => Err(PyException::new_err(e.to_string())),
            (Err(e), Err(wal_e)) => Err(PyException::new_err(format!("{e}; {wal_e}"))),
        }
    }

    /// Return current database path.
    fn path(&self) -> PyResult<String> {
        Ok(self
            .get_db()?
            .path()
            .as_os_str()
            .to_string_lossy()
            .to_string())
    }

    /// Runs a manual compaction on the Range of keys given for the current Column Family.
    #[pyo3(signature = (begin, end, compact_opt = Python::with_gil(|py| Py::new(py, CompactOptionsPy::default()).unwrap())))]
    fn compact_range(
        &self,
        begin: &Bound<PyAny>,
        end: &Bound<PyAny>,
        compact_opt: Py<CompactOptionsPy>,
        py: Python,
    ) -> PyResult<()> {
        let db = self.get_db()?;
        let opt = compact_opt.borrow(py);
        let opt_ref = opt.deref();
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
        py.allow_threads(|| {
            if let Some(cf) = &self.column_family {
                db.compact_range_cf_opt(cf, from, to, &opt_ref.0)
            } else {
                db.compact_range_opt(from, to, &opt_ref.0)
            };
        });
        Ok(())
    }

    /// Set options for the current column family.
    fn set_options(&self, options: HashMap<String, String>) -> PyResult<()> {
        let db = self.get_db()?;
        let options: Vec<(&str, &str)> = options
            .iter()
            .map(|(opt, v)| (opt.as_str(), v.as_str()))
            .collect();
        match &self.column_family {
            None => db.set_options(&options),
            Some(cf) => db.set_options_cf(cf, &options),
        }
        .map_err(|e| PyException::new_err(e.to_string()))
    }

    /// Retrieves a RocksDB property by name, for the current column family.
    fn property_value(&self, name: &str) -> PyResult<Option<String>> {
        let db = self.get_db()?;
        match &self.column_family {
            None => db.property_value(name),
            Some(cf) => db.property_value_cf(cf, name),
        }
        .map_err(|e| PyException::new_err(e.to_string()))
    }

    /// Retrieves a RocksDB property and casts it to an integer
    /// (for the current column family).
    ///
    /// Full list of properties that return int values could be find
    /// [here](https://github.com/facebook/rocksdb/blob/08809f5e6cd9cc4bc3958dd4d59457ae78c76660/include/rocksdb/db.h#L654-L689).
    fn property_int_value(&self, name: &str) -> PyResult<Option<u64>> {
        let db = self.get_db()?;
        match &self.column_family {
            None => db.property_int_value(name),
            Some(cf) => db.property_int_value_cf(cf, name),
        }
        .map_err(|e| PyException::new_err(e.to_string()))
    }

    /// The sequence number of the most recent transaction.
    fn latest_sequence_number(&self) -> PyResult<u64> {
        Ok(self.get_db()?.latest_sequence_number())
    }

    /// Returns a list of all table files with their level, start key and end key
    fn live_files<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let db = self.get_db()?;
        match db.live_files() {
            Ok(lfs) => {
                let result = PyList::empty(py);
                for lf in lfs {
                    result.append(display_live_file_dict(
                        lf,
                        py,
                        &self.loads,
                        self.opt_py.raw_mode,
                    )?)?
                }
                Ok(result)
            }
            Err(e) => Err(PyException::new_err(e.to_string())),
        }
    }

    /// Delete the database.
    ///
    /// Args:
    ///     path (str): path to this database
    ///     options (rocksdict.Options): Rocksdb options object
    #[staticmethod]
    #[pyo3(signature = (path, options = OptionsPy::new(false)))]
    fn destroy(path: &str, options: OptionsPy, py: Python) -> PyResult<()> {
        let inner_opt = options.inner_opt;

        py.allow_threads(|| {
            fs::remove_file(config_file(path)).ok();
            DB::destroy(&inner_opt, path)
        })
        .map_err(|e| PyException::new_err(e.to_string()))
    }

    /// Repair the database.
    ///
    /// Args:
    ///     path (str): path to this database
    ///     options (rocksdict.Options): Rocksdb options object
    #[staticmethod]
    #[pyo3(signature = (path, options = OptionsPy::new(false)))]
    fn repair(path: &str, options: OptionsPy, py: Python) -> PyResult<()> {
        let inner_opt = options.inner_opt;

        py.allow_threads(|| DB::repair(&inner_opt, path))
            .map_err(|e| PyException::new_err(e.to_string()))
    }

    #[staticmethod]
    #[pyo3(signature = (path, options = OptionsPy::new(false)))]
    fn list_cf(path: &str, options: OptionsPy) -> PyResult<Vec<String>> {
        DB::list_cf(&options.inner_opt, path).map_err(|e| PyException::new_err(e.to_string()))
    }
}

fn display_live_file_dict<'py>(
    lf: LiveFile,
    py: Python<'py>,
    pickle_loads: &PyObject,
    raw_mode: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let result = PyDict::new(py);
    let start_key = match lf.start_key {
        None => py.None().bind(py).to_owned(),
        Some(k) => decode_value(py, &k, pickle_loads, raw_mode)?,
    };
    let end_key = match lf.end_key {
        None => py.None().bind(py).to_owned(),
        Some(k) => decode_value(py, &k, pickle_loads, raw_mode)?,
    };
    result.set_item("name", lf.name)?;
    result.set_item("size", lf.size)?;
    result.set_item("level", lf.level)?;
    result.set_item("start_key", start_key)?;
    result.set_item("end_key", end_key)?;
    result.set_item("num_entries", lf.num_entries)?;
    result.set_item("num_deletions", lf.num_deletions)?;
    Ok(result.into_any())
}

fn get_batch_inner<'a>(
    db: &DB,
    key_list: &Bound<PyList>,
    py: Python<'a>,
    read_opt: &ReadOptions,
    loads: &PyObject,
    cf: &Arc<UnboundColumnFamily>,
    raw_mode: bool,
) -> PyResult<Bound<'a, PyList>> {
    let keys_py = key_list.iter().collect::<Vec<_>>();
    let mut keys: Vec<Cow<[u8]>> = Vec::with_capacity(key_list.len());
    for key in keys_py.iter() {
        keys.push(encode_key(key, raw_mode)?);
    }
    let values = py.allow_threads(|| db.batched_multi_get_cf_opt(cf, &keys, false, read_opt));
    let result = PyList::empty(py);
    for v in values {
        match v {
            Ok(value) => match value {
                None => result.append(py.None())?,
                Some(slice) => result.append(decode_value(py, slice.as_ref(), loads, raw_mode)?)?,
            },
            Err(e) => return Err(PyException::new_err(e.to_string())),
        }
    }
    Ok(result)
}

impl Drop for Rdict {
    // flush
    fn drop(&mut self) {
        if let Some(db) = self.db.get() {
            let f_opt = &self.flush_opt;
            let _ = if let Some(cf) = &self.column_family {
                db.flush_cf_opt(cf, &f_opt.into())
            } else {
                db.flush_opt(&f_opt.into())
            };
        }
        // important, always drop column families first
        // to ensure that CF handles have shorter life than DB.
        drop(self.column_family.take());
        self.db.close();
    }
}

unsafe impl Send for Rdict {}

/// Column family handle. This can be used in WriteBatch to specify Column Family.
#[pyclass(name = "ColumnFamily")]
#[allow(dead_code)]
#[derive(Clone)]
pub(crate) struct ColumnFamilyPy {
    // must follow this drop order
    pub(crate) cf: Arc<UnboundColumnFamily>,
    // must keep db alive
    db: DbReferenceHolder,
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
    #[pyo3(signature = (error_if_log_file_exist = false))]
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
