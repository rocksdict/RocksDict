use crate::encoder::encode_key;
use crate::rdict::{RocksDictConfig, ROCKSDICT_CONFIG_FILE};
use libc::{c_char, c_uchar, size_t};
use num_bigint::BigInt;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use rocksdb::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::ffi::c_double;
use std::os::raw::{c_int, c_uint};
use std::path::{Path, PathBuf};

/// Database-wide options around performance and behavior.
///
/// Please read the official tuning [guide](https://github.com/facebook/rocksdb/wiki/RocksDB-Tuning-Guide)
/// and most importantly, measure performance under realistic workloads with realistic hardware.
///
/// Example:
///     ::
///
///         from rocksdict import Options, Rdict, DBCompactionStyle
///
///         def badly_tuned_for_somebody_elses_disk():
///
///             path = "path/for/rocksdb/storageX"
///
///             opts = Options()
///             opts.create_if_missing(true)
///             opts.set_max_open_files(10000)
///             opts.set_use_fsync(false)
///             opts.set_bytes_per_sync(8388608)
///             opts.optimize_for_point_lookup(1024)
///             opts.set_table_cache_num_shard_bits(6)
///             opts.set_max_write_buffer_number(32)
///             opts.set_write_buffer_size(536870912)
///             opts.set_target_file_size_base(1073741824)
///             opts.set_min_write_buffer_number_to_merge(4)
///             opts.set_level_zero_stop_writes_trigger(2000)
///             opts.set_level_zero_slowdown_writes_trigger(0)
///             opts.set_compaction_style(DBCompactionStyle.universal())
///             opts.set_disable_auto_compactions(true)
///
///             return Rdict(path, opts)
///
/// Args:
///     raw_mode (bool): set this to True to operate in raw mode (i.e.
///         it will only allow bytes as key-value pairs, and is compatible
///         with other RockDB database).
///
#[pyclass(name = "Options")]
#[derive(Clone)]
pub(crate) struct OptionsPy {
    pub(crate) inner_opt: Options,
    pub(crate) raw_mode: bool,
    pub(crate) prefix_extractor: Option<SliceTransformType>,
}

/// Optionally disable WAL or sync for this write.
///
/// Example:
///     ::
///
///         from rocksdict import Rdict, Options, WriteBatch, WriteOptions
///
///         path = "_path_for_rocksdb_storageY1"
///         db = Rdict(path, Options())
///
///         # set write options
///         write_options = WriteOptions()
///         write_options.set_sync(false)
///         write_options.disable_wal(true)
///         db.set_write_options(write_options)
///
///         # write to db
///         db["my key"] = "my value"
///         db["key2"] = "value2"
///         db["key3"] = "value3"
///
///         # remove db
///         del db
///         Rdict.destroy(path, Options())
#[pyclass(name = "WriteOptions")]
#[derive(Clone)]
pub(crate) struct WriteOptionsPy {
    /// Sets the sync mode. If true, the write will be flushed
    /// from the operating system buffer cache before the write is considered complete.
    /// If this flag is true, writes will be slower.
    ///
    /// Default: false
    #[pyo3(get, set)]
    sync: bool,

    /// Sets whether WAL should be active or not.
    /// If true, writes will not first go to the write ahead log,
    /// and the write may got lost after a crash.
    ///
    /// Default: false
    #[pyo3(get, set)]
    disable_wal: bool,

    /// If true and if user is trying to write to column families that don't exist (they were dropped),
    /// ignore the write (don't return an error). If there are multiple writes in a WriteBatch,
    /// other writes will succeed.
    ///
    /// Default: false
    #[pyo3(get, set)]
    ignore_missing_column_families: bool,

    /// If true and we need to wait or sleep for the write request, fails
    /// immediately with Status::Incomplete().
    ///
    /// Default: false
    #[pyo3(get, set)]
    no_slowdown: bool,

    /// If true, this write request is of lower priority if compaction is
    /// behind. In this case, no_slowdown = true, the request will be cancelled
    /// immediately with Status::Incomplete() returned. Otherwise, it will be
    /// slowed down. The slowdown value is determined by RocksDB to guarantee
    /// it introduces minimum impacts to high priority writes.
    ///
    /// Default: false
    #[pyo3(get, set)]
    low_pri: bool,

    /// If true, writebatch will maintain the last insert positions of each
    /// memtable as hints in concurrent write. It can improve write performance
    /// in concurrent writes if keys in one writebatch are sequential. In
    /// non-concurrent writes (when concurrent_memtable_writes is false) this
    /// option will be ignored.
    ///
    /// Default: false
    #[pyo3(get, set)]
    memtable_insert_hint_per_batch: bool,
}

/// Optionally wait for the memtable flush to be performed.
///
/// Example:
///     ::
///
///         from rocksdb import Rdict, Options, FlushOptions
///
///         path = "_path_for_rocksdb_storageY2"
///         db = Rdict(path, Options())
///
///         flush_options = FlushOptions()
///         flush_options.set_wait(true)
///
///         db.flush_opt(flush_options)
///         del db
///         Rdict.destroy(path, Options())
#[pyclass(name = "FlushOptions")]
#[derive(Copy, Clone)]
pub(crate) struct FlushOptionsPy {
    /// Waits until the flush is done.
    ///
    /// Default: true
    ///
    /// Example:
    ///     ::
    ///
    ///         from rocksdb import FlushOptions
    ///
    ///         let options = FlushOptions()
    ///         options.set_wait(False)
    #[pyo3(get, set)]
    pub(crate) wait: bool,
}

/// ReadOptions allows setting iterator bounds and so on.
///
/// Args:
///     raw_mode (bool): this must be the same as `Options` raw_mode
///         argument.
#[pyclass(name = "ReadOptions")]
#[derive(Clone)]
pub(crate) struct ReadOptionsPy {
    fill_cache: bool,
    iterate_upper_bound: PyObject,
    iterate_lower_bound: PyObject,
    prefix_same_as_start: bool,
    total_order_seek: bool,
    max_skippable_internal_keys: u64,
    background_purge_on_iterator_cleanup: bool,
    ignore_range_deletions: bool,
    verify_checksums: bool,
    readahead_size: usize,
    tailing: bool,
    pin_data: bool,
    async_io: bool,
}

pub(crate) struct ReadOpt(pub(crate) *mut librocksdb_sys::rocksdb_readoptions_t);

/// Defines the underlying memtable implementation.
/// See official [wiki](https://github.com/facebook/rocksdb/wiki/MemTable) for more information.
#[pyclass(name = "MemtableFactory")]
pub(crate) struct MemtableFactoryPy(MemtableFactory);

/// For configuring block-based file storage.
#[pyclass(name = "BlockBasedOptions")]
pub(crate) struct BlockBasedOptionsPy(BlockBasedOptions);

/// Configuration of cuckoo-based storage.
#[pyclass(name = "CuckooTableOptions")]
pub(crate) struct CuckooTableOptionsPy(CuckooTableOptions);

/// Used in `PlainTableFactoryOptions`.
#[derive(Clone)]
#[pyclass(name = "KeyEncodingType")]
pub(crate) struct KeyEncodingTypePy(KeyEncodingType);

#[pymethods]
impl KeyEncodingTypePy {
    /// Always write full keys.
    #[staticmethod]
    pub fn plain() -> Self {
        KeyEncodingTypePy(KeyEncodingType::Plain)
    }

    /// Find opportunities to write the same prefix for multiple rows.
    #[staticmethod]
    pub fn prefix() -> Self {
        KeyEncodingTypePy(KeyEncodingType::Prefix)
    }
}

///
/// Used with DBOptions::set_plain_table_factory.
/// See official [wiki](https://github.com/facebook/rocksdb/wiki/PlainTable-Format) for more
/// information.
///
/// Defaults:
///  user_key_length: 0 (variable length)
///  bloom_bits_per_key: 10
///  hash_table_ratio: 0.75
///  index_sparseness: 16
///
#[pyclass(name = "PlainTableFactoryOptions")]
pub(crate) struct PlainTableFactoryOptionsPy {
    #[pyo3(get, set)]
    user_key_length: u32,

    #[pyo3(get, set)]
    bloom_bits_per_key: i32,

    #[pyo3(get, set)]
    hash_table_ratio: f64,

    #[pyo3(get, set)]
    index_sparseness: usize,

    #[pyo3(get, set)]
    huge_page_tlb_size: usize,

    #[pyo3(get, set)]
    encoding_type: KeyEncodingTypePy,

    #[pyo3(get, set)]
    full_scan_mode: bool,

    #[pyo3(get, set)]
    store_index_in_file: bool,
}

#[pyclass(name = "Cache")]
#[derive(Clone)]
pub(crate) struct CachePy(Cache);

#[pyclass(name = "BlockBasedIndexType")]
pub(crate) struct BlockBasedIndexTypePy(BlockBasedIndexType);

#[pyclass(name = "DataBlockIndexType")]
pub(crate) struct DataBlockIndexTypePy(DataBlockIndexType);

#[pyclass(name = "SliceTransform")]
pub(crate) struct SliceTransformPy(SliceTransformType);

#[derive(Deserialize, Serialize, Clone)]
pub enum SliceTransformType {
    Fixed(size_t),
    MaxLen(usize),
    Noop,
}

#[pyclass(name = "DBPath")]
pub(crate) struct DBPathPy {
    path: PathBuf,
    target_size: u64,
}

/// This is to be treated as an enum.
///
/// Call the corresponding functions of each
/// to get one of the following.
/// - None
/// - Snappy
/// - Zlib
/// - Bz2
/// - Lz4
/// - Lz4hc
/// - Zstd
///
/// Below is an example to set compression type to Snappy.
///
/// Example:
///     ::
///
///         opt = Options()
///         opt.set_compression_type(DBCompressionType.snappy())
///
#[pyclass(name = "DBCompressionType")]
pub(crate) struct DBCompressionTypePy(DBCompressionType);

/// This is to be treated as an enum.
///
/// Call the corresponding functions of each
/// to get one of the following.
/// - Level
/// - Universal
/// - Fifo
///
/// Below is an example to set compaction style to Fifo.
///
/// Example:
///     ::
///
///         opt = Options()
///         opt.set_compaction_style(DBCompactionStyle.fifo())
///
#[pyclass(name = "DBCompactionStyle")]
pub(crate) struct DBCompactionStylePy(DBCompactionStyle);

/// <https://github.com/facebook/rocksdb/wiki/Write-Buffer-Manager>
/// Write buffer manager helps users control the total memory used
/// by memtables across multiple column families and/or DB instances.
/// Users can enable this control by 2 ways:
///
/// 1- Limit the total memtable usage across multiple column families
/// and DBs under a threshold.
/// 2- Cost the memtable memory usage to block cache so that memory of
/// RocksDB can be capped by the single limit.
/// The usage of a write buffer manager is similar to rate_limiter and sst_file_manager.
/// Users can create one write buffer manager object and pass it to all the options
/// of column families or DBs whose memtable size they want to be controlled by this object.
///
/// A memory limit is given when creating the write buffer manager object.
/// RocksDB will try to limit the total memory to under this limit.
///
/// a flush will be triggered on one column family of the DB you are inserting to,
///
/// If mutable memtable size exceeds about 90% of the limit,
/// If the total memory is over the limit, more aggressive flush may also be
/// triggered only if the mutable memtable size also exceeds 50% of the limit.
/// Both checks are needed because if already more than half memory is being
/// flushed, triggering more flush may not help.
///
/// The total memory is counted as total memory allocated in the arena,
/// even if some of that may not yet be used by memtable.
///
/// buffer_size: the memory limit in bytes.
/// allow_stall: If set true, it will enable stalling of all writers when
/// memory usage exceeds buffer_size (soft limit).
///             It will wait for flush to complete and memory usage to drop down
#[pyclass(name = "WriteBufferManager")]
pub(crate) struct WriteBufferManagerPy(WriteBufferManager);

/// Used by BlockBasedOptions::set_checksum_type.
///
/// Call the corresponding functions of each
/// to get one of the following.
/// - NoChecksum
/// - CRC32c
/// - XXHash
/// - XXHash64
/// - XXH3
///
#[pyclass(name = "ChecksumType")]
pub(crate) struct ChecksumTypePy(ChecksumType);

impl Clone for ChecksumTypePy {
    fn clone(&self) -> Self {
        match self.0 {
            ChecksumType::NoChecksum => ChecksumTypePy(ChecksumType::NoChecksum),
            ChecksumType::CRC32c => ChecksumTypePy(ChecksumType::CRC32c),
            ChecksumType::XXHash => ChecksumTypePy(ChecksumType::XXHash),
            ChecksumType::XXHash64 => ChecksumTypePy(ChecksumType::XXHash64),
            ChecksumType::XXH3 => ChecksumTypePy(ChecksumType::XXH3),
        }
    }
}

/// This is to be treated as an enum.
///
/// Calling the corresponding functions of each
/// to get one of the following.
/// - TolerateCorruptedTailRecords
/// - AbsoluteConsistency
/// - PointInTime
/// - SkipAnyCorruptedRecord
///
/// Below is an example to set recovery mode to PointInTime.
///
/// Example:
///     ::
///
///         opt = Options()
///         opt.set_wal_recovery_mode(DBRecoveryMode.point_in_time())
///
#[pyclass(name = "DBRecoveryMode")]
pub(crate) struct DBRecoveryModePy(DBRecoveryMode);

#[pyclass(name = "Env")]
#[derive(Clone)]
pub(crate) struct EnvPy(Env);

#[pyclass(name = "UniversalCompactOptions")]
pub(crate) struct UniversalCompactOptionsPy {
    /// Sets the percentage flexibility while comparing file size.
    /// If the candidate file(s) size is 1% smaller than the next file's size,
    /// then include next file into this candidate set.
    ///
    /// Default: 1
    #[pyo3(get, set)]
    size_ratio: c_int,

    /// Sets the minimum number of files in a single compaction run.
    ///
    /// Default: 2
    #[pyo3(get, set)]
    min_merge_width: c_int,

    /// Sets the maximum number of files in a single compaction run.
    ///
    /// Default: UINT_MAX
    #[pyo3(get, set)]
    max_merge_width: c_int,

    /// sets the size amplification.
    ///
    /// It is defined as the amount (in percentage) of
    /// additional storage needed to store a single byte of data in the database.
    /// For example, a size amplification of 2% means that a database that
    /// contains 100 bytes of user-data may occupy upto 102 bytes of
    /// physical storage. By this definition, a fully compacted database has
    /// a size amplification of 0%. Rocksdb uses the following heuristic
    /// to calculate size amplification: it assumes that all files excluding
    /// the earliest file contribute to the size amplification.
    ///
    /// Default: 200, which means that a 100 byte database could require upto 300 bytes of storage.
    #[pyo3(get, set)]
    max_size_amplification_percent: c_int,

    /// Sets the percentage of compression size.
    ///
    /// If this option is set to be -1, all the output files
    /// will follow compression type specified.
    ///
    /// If this option is not negative, we will try to make sure compressed
    /// size is just above this value. In normal cases, at least this percentage
    /// of data will be compressed.
    /// When we are compacting to a new file, here is the criteria whether
    /// it needs to be compressed: assuming here are the list of files sorted
    /// by generation time:
    ///    A1...An B1...Bm C1...Ct
    /// where A1 is the newest and Ct is the oldest, and we are going to compact
    /// B1...Bm, we calculate the total size of all the files as total_size, as
    /// well as  the total size of C1...Ct as total_C, the compaction output file
    /// will be compressed iff
    ///   total_C / total_size < this percentage
    ///
    /// Default: -1
    #[pyo3(get, set)]
    compression_size_percent: c_int,

    /// Sets the algorithm used to stop picking files into a single compaction run.
    ///
    /// Default: ::Total
    #[pyo3(get, set)]
    stop_style: UniversalCompactionStopStylePy,
}

#[pyclass(name = "UniversalCompactionStopStyle")]
#[derive(Copy, Clone)]
pub(crate) struct UniversalCompactionStopStylePy(UniversalCompactionStopStyle);

#[pyclass(name = "FifoCompactOptions")]
pub(crate) struct FifoCompactOptionsPy {
    /// Sets the max table file size.
    ///
    /// Once the total sum of table files reaches this, we will delete the oldest
    /// table file
    ///
    /// Default: 1GB
    #[pyo3(get, set)]
    max_table_files_size: u64,
}

#[pyclass(name = "IngestExternalFileOptions")]
pub(crate) struct IngestExternalFileOptionsPy(pub(crate) IngestExternalFileOptions);

#[pyclass(name = "BottommostLevelCompaction")]
#[derive(Clone)]
pub(crate) struct BottommostLevelCompactionPy(BottommostLevelCompaction);

#[pyclass(name = "CompactOptions")]
pub(crate) struct CompactOptionsPy(pub(crate) CompactOptions);

unsafe impl Send for CompactOptionsPy {}
unsafe impl Sync for CompactOptionsPy {}

impl OptionsPy {
    /// function that sets prefix extractor according to slice transform type
    fn set_prefix_extractor_inner(
        options: &mut Options,
        slice_transform_type: &SliceTransformType,
    ) -> PyResult<()> {
        let transform = match slice_transform_type {
            SliceTransformType::Fixed(len) => SliceTransform::create_fixed_prefix(*len),
            SliceTransformType::MaxLen(len) => match create_max_len_transform(*len) {
                Ok(f) => f,
                Err(_) => {
                    return Err(PyException::new_err(
                        "max len prefix only supports len from 1 to 128",
                    ))
                }
            },
            SliceTransformType::Noop => SliceTransform::create_noop(),
        };
        options.set_prefix_extractor(transform);
        Ok(())
    }

    /// load latest options from OPTIONS files and config files
    pub fn load_latest_inner(
        path: &str,
        env: EnvPy,
        ignore_unknown_options: bool,
        cache: CachePy,
    ) -> PyResult<(OptionsPy, HashMap<String, OptionsPy>)> {
        let mut config_path = PathBuf::from(path);
        config_path.push(ROCKSDICT_CONFIG_FILE);
        let rocksdict_config = RocksDictConfig::load(config_path).unwrap_or_default();
        let raw_mode = rocksdict_config.raw_mode;
        let slice_transforms = rocksdict_config.prefix_extractors;
        let load_result = Options::load_latest(path, env.0, ignore_unknown_options, cache.0);
        let (options, column_families) = match load_result {
            Ok(d) => d,
            Err(e) => return Err(PyException::new_err(e.to_string())),
        };
        let options = OptionsPy::compose_options_py(
            options,
            raw_mode,
            slice_transforms.get(DEFAULT_COLUMN_FAMILY_NAME).cloned(),
        )?;
        let column_families: PyResult<HashMap<_, _>> = column_families
            .into_iter()
            .map(|c| {
                let opt = OptionsPy::compose_options_py(
                    c.options,
                    raw_mode,
                    slice_transforms.get(&c.name).cloned(),
                );
                match opt {
                    Ok(opt) => Ok((c.name, opt)),
                    Err(e) => Err(e),
                }
            })
            .collect();
        Ok((options, column_families?))
    }

    /// convert `Options` into `OptionsPy` based on `raw_mode` and `prefix_extractor`
    fn compose_options_py(
        opt: Options,
        raw_mode: bool,
        prefix_extractor: Option<SliceTransformType>,
    ) -> PyResult<OptionsPy> {
        let mut opt = opt;
        if !raw_mode {
            OptionsPy::set_rocksdict_comparator(&mut opt);
        }
        if let Some(slice_transform) = &prefix_extractor {
            OptionsPy::set_prefix_extractor_inner(&mut opt, slice_transform)?
        }
        let options = OptionsPy {
            inner_opt: opt,
            raw_mode,
            prefix_extractor,
        };
        Ok(options)
    }

    fn set_rocksdict_comparator(opt: &mut Options) {
        opt.set_comparator(
            "rocksdict",
            Box::new(|v1, v2| {
                if let (Some(3), Some(3)) = (v1.first(), v2.first()) {
                    BigInt::from_signed_bytes_be(&v1[1..])
                        .cmp(&BigInt::from_signed_bytes_be(&v2[1..]))
                } else {
                    v1.cmp(v2)
                }
            }),
        );
    }
}

#[pymethods]
impl OptionsPy {
    #[new]
    #[pyo3(signature = (raw_mode = false))]
    pub fn new(raw_mode: bool) -> Self {
        let mut opt = Options::default();
        opt.create_if_missing(true);
        // if not raw_mode change default comparator
        if !raw_mode {
            OptionsPy::set_rocksdict_comparator(&mut opt);
        }
        OptionsPy {
            inner_opt: opt,
            raw_mode,
            prefix_extractor: None,
        }
    }

    /// Load latest options from the rocksdb path
    ///
    /// Returns a tuple, where the first item is `Options`
    /// and the second item is a `Dict` of column families.
    #[staticmethod]
    #[pyo3(signature = (
        path,
        env = EnvPy::default().unwrap(),
        ignore_unknown_options = false,
        cache = CachePy::new_lru_cache(8 * 1024 * 1204)
    ))]
    pub fn load_latest(
        path: &str,
        env: EnvPy,
        ignore_unknown_options: bool,
        cache: CachePy,
        py: Python,
    ) -> PyResult<PyObject> {
        let (options, column_families) =
            OptionsPy::load_latest_inner(path, env, ignore_unknown_options, cache)?;
        let options = Py::new(py, options)?;
        let columns = PyDict::new(py);
        for (name, opt) in column_families {
            columns.set_item(name, Py::new(py, opt)?)?
        }
        let returned_tuple = PyTuple::new(py, [options.to_object(py), columns.to_object(py)])?;
        Ok(returned_tuple.to_object(py))
    }

    /// By default, RocksDB uses only one background thread for flush and
    /// compaction. Calling this function will set it up such that total of
    /// `total_threads` is used. Good value for `total_threads` is the number of
    /// cores. You almost definitely want to call this function if your system is
    /// bottlenecked by RocksDB.
    pub fn increase_parallelism(&mut self, parallelism: i32) {
        self.inner_opt.increase_parallelism(parallelism)
    }

    /// Optimize level style compaction.
    ///
    /// Default values for some parameters in `Options` are not optimized for heavy
    /// workloads and big datasets, which means you might observe write stalls under
    /// some conditions.
    ///
    /// This can be used as one of the starting points for tuning RocksDB options in
    /// such cases.
    ///
    /// Internally, it sets `write_buffer_size`, `min_write_buffer_number_to_merge`,
    /// `max_write_buffer_number`, `level0_file_num_compaction_trigger`,
    /// `target_file_size_base`, `max_bytes_for_level_base`, so it can override if those
    /// parameters were set before.
    ///
    /// It sets buffer sizes so that memory consumption would be constrained by
    /// `memtable_memory_budget`.
    pub fn optimize_level_style_compaction(&mut self, memtable_memory_budget: usize) {
        self.inner_opt
            .optimize_level_style_compaction(memtable_memory_budget)
    }

    /// Optimize universal style compaction.
    ///
    /// Default values for some parameters in `Options` are not optimized for heavy
    /// workloads and big datasets, which means you might observe write stalls under
    /// some conditions.
    ///
    /// This can be used as one of the starting points for tuning RocksDB options in
    /// such cases.
    ///
    /// Internally, it sets `write_buffer_size`, `min_write_buffer_number_to_merge`,
    /// `max_write_buffer_number`, `level0_file_num_compaction_trigger`,
    /// `target_file_size_base`, `max_bytes_for_level_base`, so it can override if those
    /// parameters were set before.
    ///
    /// It sets buffer sizes so that memory consumption would be constrained by
    /// `memtable_memory_budget`.
    pub fn optimize_universal_style_compaction(&mut self, memtable_memory_budget: usize) {
        self.inner_opt
            .optimize_universal_style_compaction(memtable_memory_budget)
    }

    /// If true, any column families that didn't exist when opening the database
    /// will be created.
    ///
    /// Default: `true`
    pub fn create_if_missing(&mut self, create_if_missing: bool) {
        self.inner_opt.create_if_missing(create_if_missing)
    }

    /// If true, any column families that didn't exist when opening the database
    /// will be created.
    ///
    /// Default: `false`
    pub fn create_missing_column_families(&mut self, create_missing_cfs: bool) {
        self.inner_opt
            .create_missing_column_families(create_missing_cfs)
    }

    /// Specifies whether an error should be raised if the database already exists.
    ///
    /// Default: false
    pub fn set_error_if_exists(&mut self, enabled: bool) {
        self.inner_opt.set_error_if_exists(enabled)
    }

    /// Enable/disable paranoid checks.
    ///
    /// If true, the implementation will do aggressive checking of the
    /// data it is processing and will stop early if it detects any
    /// errors. This may have unforeseen ramifications: for example, a
    /// corruption of one DB entry may cause a large number of entries to
    /// become unreadable or for the entire DB to become unopenable.
    /// If any of the  writes to the database fails (Put, Delete, Merge, Write),
    /// the database will switch to read-only mode and fail all other
    /// Write operations.
    ///
    /// Default: false
    pub fn set_paranoid_checks(&mut self, enabled: bool) {
        self.inner_opt.set_paranoid_checks(enabled)
    }

    /// A list of paths where SST files can be put into, with its target size.
    /// Newer data is placed into paths specified earlier in the vector while
    /// older data gradually moves to paths specified later in the vector.
    ///
    /// For example, you have a flash device with 10GB allocated for the DB,
    /// as well as a hard drive of 2TB, you should config it to be:
    ///   [{"/flash_path", 10GB}, {"/hard_drive", 2TB}]
    ///
    /// The system will try to guarantee data under each path is close to but
    /// not larger than the target size. But current and future file sizes used
    /// by determining where to place a file are based on best-effort estimation,
    /// which means there is a chance that the actual size under the directory
    /// is slightly more than target size under some workloads. User should give
    /// some buffer room for those cases.
    ///
    /// If none of the paths has sufficient room to place a file, the file will
    /// be placed to the last path anyway, despite to the target size.
    ///
    /// Placing newer data to earlier paths is also best-efforts. User should
    /// expect user files to be placed in higher levels in some extreme cases.
    ///
    /// If left empty, only one path will be used, which is `path` passed when
    /// opening the DB.
    ///
    /// Default: empty
    ///
    ///     from rocksdict import Options, DBPath
    ///
    ///     opt = Options()
    ///     flash_path = DBPath("/flash_path", 10 * 1024 * 1024 * 1024) # 10 GB
    ///     hard_drive = DBPath("/hard_drive", 2 * 1024 * 1024 * 1024 * 1024) # 2 TB
    ///     opt.set_db_paths([flash_path, hard_drive])
    pub fn set_db_paths(&mut self, paths: &Bound<PyList>) -> PyResult<()> {
        let mut db_paths = Vec::with_capacity(paths.len());
        for p in paths.iter() {
            let path: &Bound<DBPathPy> = p.downcast()?;
            db_paths.push(
                match DBPath::new(&path.borrow().path, path.borrow().target_size) {
                    Ok(p) => p,
                    Err(e) => return Err(PyException::new_err(e.into_string())),
                },
            );
        }
        self.inner_opt.set_db_paths(&db_paths);
        Ok(())
    }

    /// Use the specified object to interact with the environment,
    /// e.g. to read/write files, schedule background work, etc. In the near
    /// future, support for doing storage operations such as read/write files
    /// through env will be deprecated in favor of file_system.
    pub fn set_env(&mut self, env: &EnvPy) {
        self.inner_opt.set_env(&env.0)
    }

    /// Sets the compression algorithm that will be used for compressing blocks.
    ///
    /// Default: `DBCompressionType::Snappy` (`DBCompressionType::None` if
    /// snappy feature is not enabled).
    ///
    /// Example:
    ///     ::
    ///
    ///         from rocksdict import Options, DBCompressionType
    ///
    ///         opts = Options()
    ///         opts.set_compression_type(DBCompressionType.snappy())
    pub fn set_compression_type(&mut self, t: &DBCompressionTypePy) {
        self.inner_opt.set_compression_type(t.0)
    }

    /// Sets the bottom-most compression algorithm that will be used for
    /// compressing blocks at the bottom-most level.
    ///
    /// Note that to actually enable bottom-most compression configuration after
    /// setting the compression type, it needs to be enabled by calling
    /// [`set_bottommost_compression_options`](#method.set_bottommost_compression_options) or
    /// [`set_bottommost_zstd_max_train_bytes`](#method.set_bottommost_zstd_max_train_bytes) method with `enabled` argument
    /// set to `true`.
    ///
    ///
    /// Example:
    ///     ::
    ///
    ///         from rocksdict import Options, DBCompressionType
    ///
    ///         opts = Options()
    ///         opts.set_bottommost_compression_type(DBCompressionType.snappy())
    pub fn set_bottommost_compression_type(&mut self, t: &DBCompressionTypePy) {
        self.inner_opt.set_bottommost_compression_type(t.0)
    }

    /// Sets compression options for blocks at the bottom-most level.  Meaning
    /// of all settings is the same as in [`set_compression_options`](#method.set_compression_options) method but
    /// affect only the bottom-most compression which is set using
    /// [`set_bottommost_compression_type`](#method.set_bottommost_compression_type) method.
    pub fn set_bottommost_compression_options(
        &mut self,
        w_bits: c_int,
        level: c_int,
        strategy: c_int,
        max_dict_bytes: c_int,
        enable: bool,
    ) {
        self.inner_opt.set_bottommost_compression_options(
            w_bits,
            level,
            strategy,
            max_dict_bytes,
            enable,
        )
    }

    /// Sets maximum size of training data passed to zstd's dictionary trainer
    /// when compressing the bottom-most level. Using zstd's dictionary trainer
    /// can achieve even better compression ratio improvements than using
    /// `max_dict_bytes` alone.
    ///
    /// The training data will be used to generate a dictionary of
    /// `max_dict_bytes`.
    ///
    /// Default: 0.
    pub fn set_bottommost_zstd_max_train_bytes(&mut self, value: c_int, enable: bool) {
        self.inner_opt
            .set_bottommost_zstd_max_train_bytes(value, enable)
    }

    /// Different levels can have different compression policies. There
    /// are cases where most lower levels would like to use quick compression
    /// algorithms while the higher levels (which have more data) use
    /// compression algorithms that have better compression but could
    /// be slower. This array, if non-empty, should have an entry for
    /// each level of the database; these override the value specified in
    /// the previous field 'compression'.
    ///
    /// Example:
    ///     ::
    ///
    ///         from rocksdict import Options, DBCompressionType
    ///
    ///         opts = Options()
    ///         opts.set_compression_per_level([
    ///             DBCompressionType.none(),
    ///             DBCompressionType.none(),
    ///             DBCompressionType.snappy(),
    ///             DBCompressionType.snappy(),
    ///             DBCompressionType.snappy()
    ///         ])
    pub fn set_compression_per_level(&mut self, level_types: &Bound<PyList>) -> PyResult<()> {
        let mut result = Vec::with_capacity(level_types.len());
        for py_any in level_types.iter() {
            let level_type: &Bound<DBCompressionTypePy> = py_any.downcast()?;
            result.push(level_type.borrow().0)
        }
        self.inner_opt.set_compression_per_level(&result);
        Ok(())
    }

    /// Maximum size of dictionaries used to prime the compression library.
    /// Enabling dictionary can improve compression ratios when there are
    /// repetitions across data blocks.
    ///
    /// The dictionary is created by sampling the SST file data. If
    /// `zstd_max_train_bytes` is nonzero, the samples are passed through zstd's
    /// dictionary generator. Otherwise, the random samples are used directly as
    /// the dictionary.
    ///
    /// When compression dictionary is disabled, we compress and write each block
    /// before buffering data for the next one. When compression dictionary is
    /// enabled, we buffer all SST file data in-memory so we can sample it, as data
    /// can only be compressed and written after the dictionary has been finalized.
    /// So users of this feature may see increased memory usage.
    ///
    /// Default: `0`
    pub fn set_compression_options(
        &mut self,
        w_bits: c_int,
        level: c_int,
        strategy: c_int,
        max_dict_bytes: c_int,
    ) {
        self.inner_opt
            .set_compression_options(w_bits, level, strategy, max_dict_bytes)
    }

    /// Sets maximum size of training data passed to zstd's dictionary trainer. Using zstd's
    /// dictionary trainer can achieve even better compression ratio improvements than using
    /// `max_dict_bytes` alone.
    ///
    /// The training data will be used to generate a dictionary of max_dict_bytes.
    ///
    /// Default: 0.
    pub fn set_zstd_max_train_bytes(&mut self, value: c_int) {
        self.inner_opt.set_zstd_max_train_bytes(value)
    }

    /// If non-zero, we perform bigger reads when doing compaction. If you're
    /// running RocksDB on spinning disks, you should set this to at least 2MB.
    /// That way RocksDB's compaction is doing sequential instead of random reads.
    ///
    /// When non-zero, we also force new_table_reader_for_compaction_inputs to
    /// true.
    ///
    /// Default: `0`
    pub fn set_compaction_readahead_size(&mut self, compaction_readahead_size: usize) {
        self.inner_opt
            .set_compaction_readahead_size(compaction_readahead_size)
    }

    /// Allow RocksDB to pick dynamic base of bytes for levels.
    /// With this feature turned on, RocksDB will automatically adjust max bytes for each level.
    /// The goal of this feature is to have lower bound on size amplification.
    ///
    /// Default: false.
    pub fn set_level_compaction_dynamic_level_bytes(&mut self, v: bool) {
        self.inner_opt.set_level_compaction_dynamic_level_bytes(v)
    }

    // pub fn set_merge_operator_associative<F: MergeFn + Clone>(&mut self, name: &str, full_merge_fn: F) {
    //     self.inner_opt.set_merge_operator_associative(name, full_merge_fn)
    // }

    // pub fn set_merge_operator<F: MergeFn, PF: MergeFn>(&mut self, name: &str, full_merge_fn: F, partial_merge_fn: PF,) {
    //     self.inner_opt.set_merge_operator(name, full_merge_fn, partial_merge_fn,)
    // }

    // pub fn add_merge_operator<F: MergeFn + Clone>(&mut self, name: &str, merge_fn: F) {
    //     self.0.add_merge_operator(name, merge_fn)
    // }

    // pub fn set_compaction_filter<F>(&mut self, name: &str, filter_fn: F) {
    //     self.inner_opt.set_compaction_filter(name, filter_fn)
    // }

    // pub fn set_compaction_filter_factory<F>(&mut self, factory: F) {
    //     self.inner_opt.set_compaction_filter_factory(factory)
    // }

    // pub fn set_comparator(&mut self, name: &str, compare_fn: CompareFn) {
    //     self.inner_opt.set_comparator(name, compare_fn)
    // }

    pub fn set_prefix_extractor(&mut self, prefix_extractor: &SliceTransformPy) -> PyResult<()> {
        let transform = match &prefix_extractor.0 {
            SliceTransformType::Fixed(len) => SliceTransform::create_fixed_prefix(*len),
            SliceTransformType::MaxLen(len) => match create_max_len_transform(*len) {
                Ok(f) => f,
                Err(_) => {
                    return Err(PyException::new_err(
                        "max len prefix only supports len from 1 to 128",
                    ))
                }
            },
            SliceTransformType::Noop => SliceTransform::create_noop(),
        };
        self.prefix_extractor = Some(prefix_extractor.0.clone());
        self.inner_opt.set_prefix_extractor(transform);
        Ok(())
    }

    // pub fn add_comparator(&mut self, name: &str, compare_fn: CompareFn) {
    //     self.0.add_comparator(name, compare_fn)
    // }

    pub fn optimize_for_point_lookup(&mut self, cache_size: u64) {
        self.inner_opt.optimize_for_point_lookup(cache_size)
    }

    /// Sets the optimize_filters_for_hits flag
    ///
    /// Default: `false`
    pub fn set_optimize_filters_for_hits(&mut self, optimize_for_hits: bool) {
        self.inner_opt
            .set_optimize_filters_for_hits(optimize_for_hits)
    }

    /// Sets the periodicity when obsolete files get deleted.
    ///
    /// The files that get out of scope by compaction
    /// process will still get automatically delete on every compaction,
    /// regardless of this setting.
    ///
    /// Default: 6 hours
    pub fn set_delete_obsolete_files_period_micros(&mut self, micros: u64) {
        self.inner_opt
            .set_delete_obsolete_files_period_micros(micros)
    }

    /// Prepare the DB for bulk loading.
    ///
    /// All data will be in level 0 without any automatic compaction.
    /// It's recommended to manually call CompactRange(NULL, NULL) before reading
    /// from the database, because otherwise the read can be very slow.
    pub fn prepare_for_bulk_load(&mut self) {
        self.inner_opt.prepare_for_bulk_load()
    }

    /// Sets the number of open files that can be used by the DB. You may need to
    /// increase this if your database has a large working set. Value `-1` means
    /// files opened are always kept open. You can estimate number of files based
    /// on target_file_size_base and target_file_size_multiplier for level-based
    /// compaction. For universal-style compaction, you can usually set it to `-1`.
    ///
    /// Default: `-1`
    pub fn set_max_open_files(&mut self, nfiles: c_int) {
        self.inner_opt.set_max_open_files(nfiles)
    }

    /// If max_open_files is -1, DB will open all files on DB::Open(). You can
    /// use this option to increase the number of threads used to open the files.
    /// Default: 16
    pub fn set_max_file_opening_threads(&mut self, nthreads: c_int) {
        self.inner_opt.set_max_file_opening_threads(nthreads)
    }

    /// If true, then every store to stable storage will issue a fsync.
    /// If false, then every store to stable storage will issue a fdatasync.
    /// This parameter should be set to true while storing data to
    /// filesystem like ext3 that can lose files after a reboot.
    ///
    /// Default: `false`
    pub fn set_use_fsync(&mut self, useit: bool) {
        self.inner_opt.set_use_fsync(useit)
    }

    /// Specifies the absolute info LOG dir.
    ///
    /// If it is empty, the log files will be in the same dir as data.
    /// If it is non empty, the log files will be in the specified dir,
    /// and the db data dir's absolute path will be used as the log file
    /// name's prefix.
    ///
    /// Default: empty
    pub fn set_db_log_dir(&mut self, path: &str) {
        self.inner_opt.set_db_log_dir(Path::new(path))
    }

    /// Allows OS to incrementally sync files to disk while they are being
    /// written, asynchronously, in the background. This operation can be used
    /// to smooth out write I/Os over time. Users shouldn't rely on it for
    /// persistency guarantee.
    /// Issue one request for every bytes_per_sync written. `0` turns it off.
    ///
    /// Default: `0`
    ///
    /// You may consider using rate_limiter to regulate write rate to device.
    /// When rate limiter is enabled, it automatically enables bytes_per_sync
    /// to 1MB.
    ///
    /// This option applies to table files
    pub fn set_bytes_per_sync(&mut self, nbytes: u64) {
        self.inner_opt.set_bytes_per_sync(nbytes)
    }

    /// Same as bytes_per_sync, but applies to WAL files.
    ///
    /// Default: 0, turned off
    ///
    /// Dynamically changeable through SetDBOptions() API.
    pub fn set_wal_bytes_per_sync(&mut self, nbytes: u64) {
        self.inner_opt.set_wal_bytes_per_sync(nbytes)
    }

    /// Sets the maximum buffer size that is used by WritableFileWriter.
    ///
    /// On Windows, we need to maintain an aligned buffer for writes.
    /// We allow the buffer to grow until it's size hits the limit in buffered
    /// IO and fix the buffer size when using direct IO to ensure alignment of
    /// write requests if the logical sector size is unusual
    ///
    /// Default: 1024 * 1024 (1 MB)
    ///
    /// Dynamically changeable through SetDBOptions() API.
    pub fn set_writable_file_max_buffer_size(&mut self, nbytes: u64) {
        self.inner_opt.set_writable_file_max_buffer_size(nbytes)
    }

    /// If true, allow multi-writers to update mem tables in parallel.
    /// Only some memtable_factory-s support concurrent writes; currently it
    /// is implemented only for SkipListFactory.  Concurrent memtable writes
    /// are not compatible with inplace_update_support or filter_deletes.
    /// It is strongly recommended to set enable_write_thread_adaptive_yield
    /// if you are going to use this feature.
    ///
    /// Default: true
    pub fn set_allow_concurrent_memtable_write(&mut self, allow: bool) {
        self.inner_opt.set_allow_concurrent_memtable_write(allow)
    }

    /// If true, threads synchronizing with the write batch group leader will wait for up to
    /// write_thread_max_yield_usec before blocking on a mutex. This can substantially improve
    /// throughput for concurrent workloads, regardless of whether allow_concurrent_memtable_write
    /// is enabled.
    ///
    /// Default: true
    pub fn set_enable_write_thread_adaptive_yield(&mut self, enabled: bool) {
        self.inner_opt
            .set_enable_write_thread_adaptive_yield(enabled)
    }

    /// Specifies whether an iteration->Next() sequentially skips over keys with the same user-key or not.
    ///
    /// This number specifies the number of keys (with the same userkey)
    /// that will be sequentially skipped before a reseek is issued.
    ///
    /// Default: 8
    pub fn set_max_sequential_skip_in_iterations(&mut self, num: u64) {
        self.inner_opt.set_max_sequential_skip_in_iterations(num)
    }

    /// Enable direct I/O mode for reading
    /// they may or may not improve performance depending on the use case
    ///
    /// Files will be opened in "direct I/O" mode
    /// which means that data read from the disk will not be cached or
    /// buffered. The hardware buffer of the devices may however still
    /// be used. Memory mapped files are not impacted by these parameters.
    ///
    /// Default: false
    pub fn set_use_direct_reads(&mut self, enabled: bool) {
        self.inner_opt.set_use_direct_reads(enabled)
    }

    /// Enable direct I/O mode for flush and compaction
    ///
    /// Files will be opened in "direct I/O" mode
    /// which means that data written to the disk will not be cached or
    /// buffered. The hardware buffer of the devices may however still
    /// be used. Memory mapped files are not impacted by these parameters.
    /// they may or may not improve performance depending on the use case
    ///
    /// Default: false
    pub fn set_use_direct_io_for_flush_and_compaction(&mut self, enabled: bool) {
        self.inner_opt
            .set_use_direct_io_for_flush_and_compaction(enabled)
    }

    /// Enable/dsiable child process inherit open files.
    ///
    /// Default: true
    pub fn set_is_fd_close_on_exec(&mut self, enabled: bool) {
        self.inner_opt.set_is_fd_close_on_exec(enabled)
    }

    /// Sets the number of shards used for table cache.
    ///
    /// Default: `6`
    pub fn set_table_cache_num_shard_bits(&mut self, nbits: c_int) {
        self.inner_opt.set_table_cache_num_shard_bits(nbits)
    }

    /// By default target_file_size_multiplier is 1, which means
    /// by default files in different levels will have similar size.
    ///
    /// Dynamically changeable through SetOptions() API
    pub fn set_target_file_size_multiplier(&mut self, multiplier: i32) {
        self.inner_opt.set_target_file_size_multiplier(multiplier)
    }

    /// Sets the minimum number of write buffers that will be merged together
    /// before writing to storage.  If set to `1`, then
    /// all write buffers are flushed to L0 as individual files and this increases
    /// read amplification because a get request has to check in all of these
    /// files. Also, an in-memory merge may result in writing lesser
    /// data to storage if there are duplicate records in each of these
    /// individual write buffers.
    ///
    /// Default: `1`
    pub fn set_min_write_buffer_number(&mut self, nbuf: c_int) {
        self.inner_opt.set_min_write_buffer_number(nbuf)
    }

    /// Sets the maximum number of write buffers that are built up in memory.
    /// The default and the minimum number is 2, so that when 1 write buffer
    /// is being flushed to storage, new writes can continue to the other
    /// write buffer.
    /// If max_write_buffer_number > 3, writing will be slowed down to
    /// options.delayed_write_rate if we are writing to the last write buffer
    /// allowed.
    ///
    /// Default: `2`
    pub fn set_max_write_buffer_number(&mut self, nbuf: c_int) {
        self.inner_opt.set_max_write_buffer_number(nbuf)
    }

    /// Sets the amount of data to build up in memory (backed by an unsorted log
    /// on disk) before converting to a sorted on-disk file.
    ///
    /// Larger values increase performance, especially during bulk loads.
    /// Up to max_write_buffer_number write buffers may be held in memory
    /// at the same time,
    /// so you may wish to adjust this parameter to control memory usage.
    /// Also, a larger write buffer will result in a longer recovery time
    /// the next time the database is opened.
    ///
    /// Note that write_buffer_size is enforced per column family.
    /// See db_write_buffer_size for sharing memory across column families.
    ///
    /// Default: `0x4000000` (64MiB)
    ///
    /// Dynamically changeable through SetOptions() API
    pub fn set_write_buffer_size(&mut self, size: usize) {
        self.inner_opt.set_write_buffer_size(size)
    }

    /// Amount of data to build up in memtables across all column
    /// families before writing to disk.
    ///
    /// This is distinct from write_buffer_size, which enforces a limit
    /// for a single memtable.
    ///
    /// This feature is disabled by default. Specify a non-zero value
    /// to enable it.
    ///
    /// Default: 0 (disabled)
    pub fn set_db_write_buffer_size(&mut self, size: usize) {
        self.inner_opt.set_db_write_buffer_size(size)
    }

    /// Control maximum total data size for a level.
    /// max_bytes_for_level_base is the max total for level-1.
    /// Maximum number of bytes for level L can be calculated as
    /// (max_bytes_for_level_base) * (max_bytes_for_level_multiplier ^ (L-1))
    /// For example, if max_bytes_for_level_base is 200MB, and if
    /// max_bytes_for_level_multiplier is 10, total data size for level-1
    /// will be 200MB, total file size for level-2 will be 2GB,
    /// and total file size for level-3 will be 20GB.
    ///
    /// Default: `0x10000000` (256MiB).
    ///
    /// Dynamically changeable through SetOptions() API
    pub fn set_max_bytes_for_level_base(&mut self, size: u64) {
        self.inner_opt.set_max_bytes_for_level_base(size)
    }

    /// Default: `10`
    pub fn set_max_bytes_for_level_multiplier(&mut self, mul: f64) {
        self.inner_opt.set_max_bytes_for_level_multiplier(mul)
    }

    /// The manifest file is rolled over on reaching this limit.
    /// The older manifest file be deleted.
    /// The default value is MAX_INT so that roll-over does not take place.
    pub fn set_max_manifest_file_size(&mut self, size: usize) {
        self.inner_opt.set_max_manifest_file_size(size)
    }

    /// Sets the target file size for compaction.
    /// target_file_size_base is per-file size for level-1.
    /// Target file size for level L can be calculated by
    /// target_file_size_base * (target_file_size_multiplier ^ (L-1))
    /// For example, if target_file_size_base is 2MB and
    /// target_file_size_multiplier is 10, then each file on level-1 will
    /// be 2MB, and each file on level 2 will be 20MB,
    /// and each file on level-3 will be 200MB.
    ///
    /// Default: `0x4000000` (64MiB)
    ///
    /// Dynamically changeable through SetOptions() API
    pub fn set_target_file_size_base(&mut self, size: u64) {
        self.inner_opt.set_target_file_size_base(size)
    }

    /// Sets the minimum number of write buffers that will be merged together
    /// before writing to storage.  If set to `1`, then
    /// all write buffers are flushed to L0 as individual files and this increases
    /// read amplification because a get request has to check in all of these
    /// files. Also, an in-memory merge may result in writing lesser
    /// data to storage if there are duplicate records in each of these
    /// individual write buffers.
    ///
    /// Default: `1`
    pub fn set_min_write_buffer_number_to_merge(&mut self, to_merge: c_int) {
        self.inner_opt
            .set_min_write_buffer_number_to_merge(to_merge)
    }

    /// Sets the number of files to trigger level-0 compaction. A value < `0` means that
    /// level-0 compaction will not be triggered by number of files at all.
    ///
    /// Default: `4`
    ///
    /// Dynamically changeable through SetOptions() API
    pub fn set_level_zero_file_num_compaction_trigger(&mut self, n: c_int) {
        self.inner_opt.set_level_zero_file_num_compaction_trigger(n)
    }

    /// Sets the soft limit on number of level-0 files. We start slowing down writes at this
    /// point. A value < `0` means that no writing slow down will be triggered by
    /// number of files in level-0.
    ///
    /// Default: `20`
    ///
    /// Dynamically changeable through SetOptions() API
    pub fn set_level_zero_slowdown_writes_trigger(&mut self, n: c_int) {
        self.inner_opt.set_level_zero_slowdown_writes_trigger(n)
    }

    /// Sets the maximum number of level-0 files.  We stop writes at this point.
    ///
    /// Default: `24`
    ///
    /// Dynamically changeable through SetOptions() API
    pub fn set_level_zero_stop_writes_trigger(&mut self, n: c_int) {
        self.inner_opt.set_level_zero_stop_writes_trigger(n)
    }

    /// Sets the compaction style.
    ///
    /// Default: DBCompactionStyle.level()
    pub fn set_compaction_style(&mut self, style: &DBCompactionStylePy) {
        self.inner_opt.set_compaction_style(style.0)
    }

    /// Sets the options needed to support Universal Style compactions.
    pub fn set_universal_compaction_options(&mut self, uco: &UniversalCompactOptionsPy) {
        self.inner_opt.set_universal_compaction_options(&uco.into())
    }

    /// Sets the options for FIFO compaction style.
    pub fn set_fifo_compaction_options(&mut self, fco: &FifoCompactOptionsPy) {
        self.inner_opt.set_fifo_compaction_options(&fco.into())
    }

    /// Sets unordered_write to true trades higher write throughput with
    /// relaxing the immutability guarantee of snapshots. This violates the
    /// repeatability one expects from ::Get from a snapshot, as well as
    ///:MultiGet and Iterator's consistent-point-in-time view property.
    /// If the application cannot tolerate the relaxed guarantees, it can implement
    /// its own mechanisms to work around that and yet benefit from the higher
    /// throughput. Using TransactionDB with WRITE_PREPARED write policy and
    /// two_write_queues=true is one way to achieve immutable snapshots despite
    /// unordered_write.
    ///
    /// By default, i.e., when it is false, rocksdb does not advance the sequence
    /// number for new snapshots unless all the writes with lower sequence numbers
    /// are already finished. This provides the immutability that we except from
    /// snapshots. Moreover, since Iterator and MultiGet internally depend on
    /// snapshots, the snapshot immutability results into Iterator and MultiGet
    /// offering consistent-point-in-time view. If set to true, although
    /// Read-Your-Own-Write property is still provided, the snapshot immutability
    /// property is relaxed: the writes issued after the snapshot is obtained (with
    /// larger sequence numbers) will be still not visible to the reads from that
    /// snapshot, however, there still might be pending writes (with lower sequence
    /// number) that will change the state visible to the snapshot after they are
    /// landed to the memtable.
    ///
    /// Default: false
    pub fn set_unordered_write(&mut self, unordered: bool) {
        self.inner_opt.set_unordered_write(unordered)
    }

    /// Sets maximum number of threads that will
    /// concurrently perform a compaction job by breaking it into multiple,
    /// smaller ones that are run simultaneously.
    ///
    /// Default: 1 (i.e. no subcompactions)
    pub fn set_max_subcompactions(&mut self, num: u32) {
        self.inner_opt.set_max_subcompactions(num)
    }

    /// Sets maximum number of concurrent background jobs
    /// (compactions and flushes).
    ///
    /// Default: 2
    ///
    /// Dynamically changeable through SetDBOptions() API.
    pub fn set_max_background_jobs(&mut self, jobs: c_int) {
        self.inner_opt.set_max_background_jobs(jobs)
    }

    /// Disables automatic compactions. Manual compactions can still
    /// be issued on this column family
    ///
    /// Default: `false`
    ///
    /// Dynamically changeable through SetOptions() API
    pub fn set_disable_auto_compactions(&mut self, disable: bool) {
        self.inner_opt.set_disable_auto_compactions(disable)
    }

    /// SetMemtableHugePageSize sets the page size for huge page for
    /// arena used by the memtable.
    /// If <=0, it won't allocate from huge page but from malloc.
    /// Users are responsible to reserve huge pages for it to be allocated. For
    /// example:
    ///      sysctl -w vm.nr_hugepages=20
    /// See linux doc Documentation/vm/hugetlbpage.txt
    /// If there isn't enough free huge page available, it will fall back to
    /// malloc.
    ///
    /// Dynamically changeable through SetOptions() API
    pub fn set_memtable_huge_page_size(&mut self, size: size_t) {
        self.inner_opt.set_memtable_huge_page_size(size)
    }

    /// Sets the maximum number of successive merge operations on a key in the memtable.
    ///
    /// When a merge operation is added to the memtable and the maximum number of
    /// successive merges is reached, the value of the key will be calculated and
    /// inserted into the memtable instead of the merge operation. This will
    /// ensure that there are never more than max_successive_merges merge
    /// operations in the memtable.
    ///
    /// Default: 0 (disabled)
    pub fn set_max_successive_merges(&mut self, num: usize) {
        self.inner_opt.set_max_successive_merges(num)
    }

    /// Control locality of bloom filter probes to improve cache miss rate.
    /// This option only applies to memtable prefix bloom and plaintable
    /// prefix bloom. It essentially limits the max number of cache lines each
    /// bloom filter check can touch.
    ///
    /// This optimization is turned off when set to 0. The number should never
    /// be greater than number of probes. This option can boost performance
    /// for in-memory workload but should use with care since it can cause
    /// higher false positive rate.
    ///
    /// Default: 0
    pub fn set_bloom_locality(&mut self, v: u32) {
        self.inner_opt.set_bloom_locality(v)
    }

    /// Enable/disable thread-safe inplace updates.
    ///
    /// Requires updates if
    /// * key exists in current memtable
    /// * new sizeof(new_value) <= sizeof(old_value)
    /// * old_value for that key is a put i.e. kTypeValue
    ///
    /// Default: false.
    pub fn set_inplace_update_support(&mut self, enabled: bool) {
        self.inner_opt.set_inplace_update_support(enabled)
    }

    /// Sets the number of locks used for inplace update.
    ///
    /// Default: 10000 when inplace_update_support = true, otherwise 0.
    pub fn set_inplace_update_locks(&mut self, num: usize) {
        self.inner_opt.set_inplace_update_locks(num)
    }

    /// Different max-size multipliers for different levels.
    /// These are multiplied by max_bytes_for_level_multiplier to arrive
    /// at the max-size of each level.
    ///
    /// Default: 1
    ///
    /// Dynamically changeable through SetOptions() API
    pub fn set_max_bytes_for_level_multiplier_additional(&mut self, level_values: Vec<i32>) {
        self.inner_opt
            .set_max_bytes_for_level_multiplier_additional(&level_values)
    }

    /// If true, then DB::Open() will not fetch and check sizes of all sst files.
    /// This may significantly speed up startup if there are many sst files,
    /// especially when using non-default Env with expensive GetFileSize().
    /// We'll still check that all required sst files exist.
    /// If paranoid_checks is false, this option is ignored, and sst files are
    /// not checked at all.
    ///
    /// Default: false
    pub fn set_skip_checking_sst_file_sizes_on_db_open(&mut self, value: bool) {
        self.inner_opt
            .set_skip_checking_sst_file_sizes_on_db_open(value)
    }

    /// The total maximum size(bytes) of write buffers to maintain in memory
    /// including copies of buffers that have already been flushed. This parameter
    /// only affects trimming of flushed buffers and does not affect flushing.
    /// This controls the maximum amount of write history that will be available
    /// in memory for conflict checking when Transactions are used. The actual
    /// size of write history (flushed Memtables) might be higher than this limit
    /// if further trimming will reduce write history total size below this
    /// limit. For example, if max_write_buffer_size_to_maintain is set to 64MB,
    /// and there are three flushed Memtables, with sizes of 32MB, 20MB, 20MB.
    /// Because trimming the next Memtable of size 20MB will reduce total memory
    /// usage to 52MB which is below the limit, RocksDB will stop trimming.
    ///
    /// When using an OptimisticTransactionDB:
    /// If this value is too low, some transactions may fail at commit time due
    /// to not being able to determine whether there were any write conflicts.
    ///
    /// When using a TransactionDB:
    /// If Transaction::SetSnapshot is used, TransactionDB will read either
    /// in-memory write buffers or SST files to do write-conflict checking.
    /// Increasing this value can reduce the number of reads to SST files
    /// done for conflict detection.
    ///
    /// Setting this value to 0 will cause write buffers to be freed immediately
    /// after they are flushed. If this value is set to -1,
    /// 'max_write_buffer_number * write_buffer_size' will be used.
    ///
    /// Default:
    /// If using a TransactionDB/OptimisticTransactionDB, the default value will
    /// be set to the value of 'max_write_buffer_number * write_buffer_size'
    /// if it is not explicitly set by the user.  Otherwise, the default is 0.
    pub fn set_max_write_buffer_size_to_maintain(&mut self, size: i64) {
        self.inner_opt.set_max_write_buffer_size_to_maintain(size)
    }

    /// By default, a single write thread queue is maintained. The thread gets
    /// to the head of the queue becomes write batch group leader and responsible
    /// for writing to WAL and memtable for the batch group.
    ///
    /// If enable_pipelined_write is true, separate write thread queue is
    /// maintained for WAL write and memtable write. A write thread first enter WAL
    /// writer queue and then memtable writer queue. Pending thread on the WAL
    /// writer queue thus only have to wait for previous writers to finish their
    /// WAL writing but not the memtable writing. Enabling the feature may improve
    /// write throughput and reduce latency of the prepare phase of two-phase
    /// commit.
    ///
    /// Default: false
    pub fn set_enable_pipelined_write(&mut self, value: bool) {
        self.inner_opt.set_enable_pipelined_write(value)
    }

    /// Defines the underlying memtable implementation.
    /// See official [wiki](https://github.com/facebook/rocksdb/wiki/MemTable) for more information.
    /// Defaults to using a skiplist.
    ///
    /// Example:
    ///     ::
    ///
    ///         from rocksdict import Options, MemtableFactory
    ///         opts = Options()
    ///         factory = MemtableFactory.hash_skip_list(bucket_count=1_000_000,
    ///                                                  height=4,
    ///                                                  branching_factor=4)
    ///
    ///         opts.set_allow_concurrent_memtable_write(false)
    ///         opts.set_memtable_factory(factory)
    pub fn set_memtable_factory(&mut self, factory: &MemtableFactoryPy) {
        self.inner_opt.set_memtable_factory(match factory.0 {
            MemtableFactory::Vector => MemtableFactory::Vector,
            MemtableFactory::HashSkipList {
                bucket_count,
                height,
                branching_factor,
            } => MemtableFactory::HashSkipList {
                bucket_count,
                height,
                branching_factor,
            },
            MemtableFactory::HashLinkList { bucket_count } => {
                MemtableFactory::HashLinkList { bucket_count }
            }
        })
    }

    pub fn set_block_based_table_factory(&mut self, factory: &BlockBasedOptionsPy) {
        self.inner_opt.set_block_based_table_factory(&factory.0)
    }

    /// Sets the table factory to a CuckooTableFactory (the default table
    /// factory is a block-based table factory that provides a default
    /// implementation of TableBuilder and TableReader with default
    /// BlockBasedTableOptions).
    /// See official [wiki](https://github.com/facebook/rocksdb/wiki/CuckooTable-Format) for more information on this table format.
    /// Example:
    ///     ::
    ///
    ///         from rocksdict import Options, CuckooTableOptions
    ///
    ///         opts = Options()
    ///         factory_opts = CuckooTableOptions()
    ///         factory_opts.set_hash_ratio(0.8)
    ///         factory_opts.set_max_search_depth(20)
    ///         factory_opts.set_cuckoo_block_size(10)
    ///         factory_opts.set_identity_as_first_hash(true)
    ///         factory_opts.set_use_module_hash(false)
    ///
    ///         opts.set_cuckoo_table_factory(factory_opts)
    pub fn set_cuckoo_table_factory(&mut self, factory: &CuckooTableOptionsPy) {
        self.inner_opt.set_cuckoo_table_factory(&factory.0)
    }

    /// This is a factory that provides TableFactory objects.
    /// Default: a block-based table factory that provides a default
    /// implementation of TableBuilder and TableReader with default
    /// BlockBasedTableOptions.
    /// Sets the factory as plain table.
    /// See official [wiki](https://github.com/facebook/rocksdb/wiki/PlainTable-Format) for more
    /// information.
    ///
    /// Example:
    ///     ::
    ///
    ///         from rocksdict import Options, PlainTableFactoryOptions
    ///
    ///         opts = Options()
    ///         factory_opts = PlainTableFactoryOptions()
    ///         factory_opts.user_key_length = 0
    ///         factory_opts.bloom_bits_per_key = 20
    ///         factory_opts.hash_table_ratio = 0.75
    ///         factory_opts.index_sparseness = 16
    ///
    ///         opts.set_plain_table_factory(factory_opts)
    pub fn set_plain_table_factory(&mut self, options: &PlainTableFactoryOptionsPy) {
        self.inner_opt
            .set_plain_table_factory(&options.to_opt(self.raw_mode))
    }

    /// Sets the start level to use compression.
    pub fn set_min_level_to_compress(&mut self, lvl: c_int) {
        self.inner_opt.set_min_level_to_compress(lvl)
    }

    /// Measure IO stats in compactions and flushes, if `true`.
    ///
    /// Default: `false`
    pub fn set_report_bg_io_stats(&mut self, enable: bool) {
        self.inner_opt.set_report_bg_io_stats(enable)
    }

    /// Once write-ahead logs exceed this size, we will start forcing the flush of
    /// column families whose memtables are backed by the oldest live WAL file
    /// (i.e. the ones that are causing all the space amplification).
    ///
    /// Default: `0`
    pub fn set_max_total_wal_size(&mut self, size: u64) {
        self.inner_opt.set_max_total_wal_size(size)
    }

    /// Recovery mode to control the consistency while replaying WAL.
    ///
    /// Default: DBRecoveryMode::PointInTime
    pub fn set_wal_recovery_mode(&mut self, mode: &DBRecoveryModePy) {
        self.inner_opt.set_wal_recovery_mode(mode.0)
    }

    pub fn enable_statistics(&mut self) {
        self.inner_opt.enable_statistics()
    }

    pub fn get_statistics(&self) -> Option<String> {
        self.inner_opt.get_statistics()
    }

    /// If not zero, dump `rocksdb.stats` to LOG every `stats_dump_period_sec`.
    ///
    /// Default: `600` (10 mins)
    pub fn set_stats_dump_period_sec(&mut self, period: c_uint) {
        self.inner_opt.set_stats_dump_period_sec(period)
    }

    /// If not zero, dump rocksdb.stats to RocksDB to LOG every `stats_persist_period_sec`.
    ///
    /// Default: `600` (10 mins)
    pub fn set_stats_persist_period_sec(&mut self, period: c_uint) {
        self.inner_opt.set_stats_persist_period_sec(period)
    }

    /// When set to true, reading SST files will opt out of the filesystem's
    /// readahead. Setting this to false may improve sequential iteration
    /// performance.
    ///
    /// Default: `true`
    pub fn set_advise_random_on_open(&mut self, advise: bool) {
        self.inner_opt.set_advise_random_on_open(advise)
    }

    // pub fn set_access_hint_on_compaction_start(&mut self, pattern: AccessHint) {
    //     self.inner_opt.set_access_hint_on_compaction_start(pattern)
    // }

    /// Enable/disable adaptive mutex, which spins in the user space before resorting to kernel.
    ///
    /// This could reduce context switch when the mutex is not
    /// heavily contended. However, if the mutex is hot, we could end up
    /// wasting spin time.
    ///
    /// Default: false
    pub fn set_use_adaptive_mutex(&mut self, enabled: bool) {
        self.inner_opt.set_use_adaptive_mutex(enabled)
    }

    /// Sets the number of levels for this database.
    pub fn set_num_levels(&mut self, n: c_int) {
        self.inner_opt.set_num_levels(n)
    }

    /// When a `prefix_extractor` is defined through `opts.set_prefix_extractor` this
    /// creates a prefix bloom filter for each memtable with the size of
    /// `write_buffer_size * memtable_prefix_bloom_ratio` (capped at 0.25).
    ///
    /// Default: `0`
    pub fn set_memtable_prefix_bloom_ratio(&mut self, ratio: f64) {
        self.inner_opt.set_memtable_prefix_bloom_ratio(ratio)
    }

    /// Sets the maximum number of bytes in all compacted files.
    /// We try to limit number of bytes in one compaction to be lower than this
    /// threshold. But it's not guaranteed.
    ///
    /// Value 0 will be sanitized.
    ///
    /// Default: target_file_size_base * 25
    pub fn set_max_compaction_bytes(&mut self, nbytes: u64) {
        self.inner_opt.set_max_compaction_bytes(nbytes)
    }

    /// Specifies the absolute path of the directory the
    /// write-ahead log (WAL) should be written to.
    ///
    /// Default: same directory as the database
    pub fn set_wal_dir(&mut self, path: &str) {
        self.inner_opt.set_wal_dir(Path::new(path))
    }

    /// Sets the WAL ttl in seconds.
    ///
    /// The following two options affect how archived logs will be deleted.
    /// 1. If both set to 0, logs will be deleted asap and will not get into
    ///    the archive.
    /// 2. If wal_ttl_seconds is 0 and wal_size_limit_mb is not 0,
    ///    WAL files will be checked every 10 min and if total size is greater
    ///    then wal_size_limit_mb, they will be deleted starting with the
    ///    earliest until size_limit is met. All empty files will be deleted.
    /// 3. If wal_ttl_seconds is not 0 and wall_size_limit_mb is 0, then
    ///    WAL files will be checked every wal_ttl_seconds / 2 and those that
    ///    are older than wal_ttl_seconds will be deleted.
    /// 4. If both are not 0, WAL files will be checked every 10 min and both
    ///    checks will be performed with ttl being first.
    ///
    /// Default: 0
    pub fn set_wal_ttl_seconds(&mut self, secs: u64) {
        self.inner_opt.set_wal_ttl_seconds(secs)
    }

    /// Sets the WAL size limit in MB.
    ///
    /// If total size of WAL files is greater then wal_size_limit_mb,
    /// they will be deleted starting with the earliest until size_limit is met.
    ///
    /// Default: 0
    pub fn set_wal_size_limit_mb(&mut self, size: u64) {
        self.inner_opt.set_wal_size_limit_mb(size)
    }

    /// Sets the number of bytes to preallocate (via fallocate) the manifest files.
    ///
    /// Default is 4MB, which is reasonable to reduce random IO
    /// as well as prevent overallocation for mounts that preallocate
    /// large amounts of data (such as xfs's allocsize option).
    pub fn set_manifest_preallocation_size(&mut self, size: usize) {
        self.inner_opt.set_manifest_preallocation_size(size)
    }

    /// If true, then DB::Open() will not update the statistics used to optimize
    /// compaction decision by loading table properties from many files.
    /// Turning off this feature will improve DBOpen time especially in disk environment.
    ///
    /// Default: false
    pub fn set_skip_stats_update_on_db_open(&mut self, skip: bool) {
        self.inner_opt.set_skip_stats_update_on_db_open(skip)
    }

    /// Specify the maximal number of info log files to be kept.
    ///
    /// Default: 1000
    pub fn set_keep_log_file_num(&mut self, nfiles: usize) {
        self.inner_opt.set_keep_log_file_num(nfiles)
    }

    /// Allow the OS to mmap file for writing.
    ///
    /// Default: false
    pub fn set_allow_mmap_writes(&mut self, is_enabled: bool) {
        self.inner_opt.set_allow_mmap_writes(is_enabled)
    }

    /// Allow the OS to mmap file for reading sst tables.
    ///
    /// Default: false
    pub fn set_allow_mmap_reads(&mut self, is_enabled: bool) {
        self.inner_opt.set_allow_mmap_reads(is_enabled)
    }

    /// Guarantee that all column families are flushed together atomically.
    /// This option applies to both manual flushes (`db.flush()`) and automatic
    /// background flushes caused when memtables are filled.
    ///
    /// Note that this is only useful when the WAL is disabled. When using the
    /// WAL, writes are always consistent across column families.
    ///
    /// Default: false
    pub fn set_atomic_flush(&mut self, atomic_flush: bool) {
        self.inner_opt.set_atomic_flush(atomic_flush)
    }

    /// Sets global cache for table-level rows. Cache must outlive DB instance which uses it.
    ///
    /// Default: null (disabled)
    /// Not supported in ROCKSDB_LITE mode!
    pub fn set_row_cache(&mut self, cache: &CachePy) {
        self.inner_opt.set_row_cache(&cache.0)
    }

    /// Use to control write rate of flush and compaction. Flush has higher
    /// priority than compaction.
    /// If rate limiter is enabled, bytes_per_sync is set to 1MB by default.
    ///
    /// Default: disable
    ///
    pub fn set_ratelimiter(
        &mut self,
        rate_bytes_per_sec: i64,
        refill_period_us: i64,
        fairness: i32,
    ) {
        self.inner_opt
            .set_ratelimiter(rate_bytes_per_sec, refill_period_us, fairness)
    }

    /// Sets the maximal size of the info log file.
    ///
    /// If the log file is larger than `max_log_file_size`, a new info log file
    /// will be created. If `max_log_file_size` is equal to zero, all logs will
    /// be written to one log file.
    ///
    /// Default: 0
    ///
    /// Example:
    ///     ::
    ///
    ///         from rocksdict import Options
    ///
    ///         options = Options()
    ///         options.set_max_log_file_size(0)
    pub fn set_max_log_file_size(&mut self, size: usize) {
        self.inner_opt.set_max_log_file_size(size)
    }

    /// Sets the time for the info log file to roll (in seconds).
    ///
    /// If specified with non-zero value, log file will be rolled
    /// if it has been active longer than `log_file_time_to_roll`.
    /// Default: 0 (disabled)
    pub fn set_log_file_time_to_roll(&mut self, secs: usize) {
        self.inner_opt.set_log_file_time_to_roll(secs)
    }

    /// Controls the recycling of log files.
    ///
    /// If non-zero, previously written log files will be reused for new logs,
    /// overwriting the old data. The value indicates how many such files we will
    /// keep around at any point in time for later use. This is more efficient
    /// because the blocks are already allocated and fdatasync does not need to
    /// update the inode after each write.
    ///
    /// Default: 0
    ///
    /// Example:
    ///     ::
    ///
    ///         from rocksdict import Options
    ///
    ///         options = Options()
    ///         options.set_recycle_log_file_num(5)
    pub fn set_recycle_log_file_num(&mut self, num: usize) {
        self.inner_opt.set_recycle_log_file_num(num)
    }

    /// Sets the threshold at which all writes will be slowed down to at least delayed_write_rate if estimated
    /// bytes needed to be compaction exceed this threshold.
    ///
    /// Default: 64GB
    pub fn set_soft_pending_compaction_bytes_limit(&mut self, limit: usize) {
        self.inner_opt
            .set_soft_pending_compaction_bytes_limit(limit)
    }

    /// Sets the bytes threshold at which all writes are stopped if estimated bytes needed to be compaction exceed
    /// this threshold.
    ///
    /// Default: 256GB
    pub fn set_hard_pending_compaction_bytes_limit(&mut self, limit: usize) {
        self.inner_opt
            .set_hard_pending_compaction_bytes_limit(limit)
    }

    /// Sets the size of one block in arena memory allocation.
    ///
    /// If <= 0, a proper value is automatically calculated (usually 1/10 of
    /// writer_buffer_size).
    ///
    /// Default: 0
    pub fn set_arena_block_size(&mut self, size: usize) {
        self.inner_opt.set_arena_block_size(size)
    }

    /// If true, then print malloc stats together with rocksdb.stats when printing to LOG.
    ///
    /// Default: false
    pub fn set_dump_malloc_stats(&mut self, enabled: bool) {
        self.inner_opt.set_dump_malloc_stats(enabled)
    }

    /// Enable whole key bloom filter in memtable. Note this will only take effect
    /// if memtable_prefix_bloom_size_ratio is not 0. Enabling whole key filtering
    /// can potentially reduce CPU usage for point-look-ups.
    ///
    /// Default: false (disable)
    ///
    /// Dynamically changeable through SetOptions() API
    pub fn set_memtable_whole_key_filtering(&mut self, whole_key_filter: bool) {
        self.inner_opt
            .set_memtable_whole_key_filtering(whole_key_filter)
    }

    /// Enable the use of key-value separation.
    ///
    /// More details can be found here: [Integrated BlobDB](http://rocksdb.org/blog/2021/05/26/integrated-blob-db.html).
    ///
    /// Default: false (disable)
    ///
    /// Dynamically changeable through SetOptions() API
    pub fn set_enable_blob_files(&mut self, val: bool) {
        self.inner_opt.set_enable_blob_files(val)
    }
    /// Sets the minimum threshold value at or above which will be written
    /// to blob files during flush or compaction.
    ///
    /// Dynamically changeable through SetOptions() API
    pub fn set_min_blob_size(&mut self, val: u64) {
        self.inner_opt.set_min_blob_size(val)
    }
    /// Sets the size limit for blob files.
    ///
    /// Dynamically changeable through SetOptions() API
    pub fn set_blob_file_size(&mut self, val: u64) {
        self.inner_opt.set_blob_file_size(val)
    }
    /// Sets the blob compression type. All blob files use the same
    /// compression type.
    ///
    /// Dynamically changeable through SetOptions() API
    pub fn set_blob_compression_type(&mut self, val: &DBCompressionTypePy) {
        self.inner_opt.set_blob_compression_type(val.0)
    }
    /// If this is set to true RocksDB will actively relocate valid blobs from the oldest blob files
    /// as they are encountered during compaction.
    ///
    /// Dynamically changeable through SetOptions() API
    pub fn set_enable_blob_gc(&mut self, val: bool) {
        self.inner_opt.set_enable_blob_gc(val)
    }
    /// Sets the threshold that the GC logic uses to determine which blob files should be considered “old.”
    ///
    /// For example, the default value of 0.25 signals to RocksDB that blobs residing in the
    /// oldest 25% of blob files should be relocated by GC. This parameter can be tuned to adjust
    /// the trade-off between write amplification and space amplification.
    ///
    /// Dynamically changeable through SetOptions() API
    pub fn set_blob_gc_age_cutoff(&mut self, val: c_double) {
        self.inner_opt.set_blob_gc_age_cutoff(val)
    }
    /// Sets the blob GC force threshold.
    ///
    /// Dynamically changeable through SetOptions() API
    pub fn set_blob_gc_force_threshold(&mut self, val: c_double) {
        self.inner_opt.set_blob_gc_force_threshold(val)
    }
    /// Sets the blob compaction read ahead size.
    ///
    /// Dynamically changeable through SetOptions() API
    pub fn set_blob_compaction_readahead_size(&mut self, val: u64) {
        self.inner_opt.set_blob_compaction_readahead_size(val)
    }
    /// Set this option to true during creation of database if you want
    /// to be able to ingest behind (call IngestExternalFile() skipping keys
    /// that already exist, rather than overwriting matching keys).
    /// Setting this option to true has the following effects:
    /// 1) Disable some internal optimizations around SST file compression.
    /// 2) Reserve the last level for ingested files only.
    /// 3) Compaction will not include any file from the last level.
    /// Note that only Universal Compaction supports allow_ingest_behind.
    /// `num_levels` should be >= 3 if this option is turned on.
    ///
    /// DEFAULT: false
    /// Immutable.
    pub fn set_allow_ingest_behind(&mut self, val: bool) {
        self.inner_opt.set_allow_ingest_behind(val)
    }

    /// A factory of a table property collector that marks an SST
    /// file as need-compaction when it observe at least "D" deletion
    /// entries in any "N" consecutive entries, or the ratio of tombstone
    /// entries >= deletion_ratio.
    ///
    /// `window_size`: is the sliding window size "N"
    /// `num_dels_trigger`: is the deletion trigger "D"
    /// `deletion_ratio`: if <= 0 or > 1, disable triggering compaction based on
    /// deletion ratio.
    pub fn add_compact_on_deletion_collector_factory(
        &mut self,
        window_size: size_t,
        num_dels_trigger: size_t,
        deletion_ratio: f64,
    ) {
        self.inner_opt.add_compact_on_deletion_collector_factory(
            window_size,
            num_dels_trigger,
            deletion_ratio,
        )
    }

    /// Write buffer manager helps users control the total memory used by
    /// memtables across multiple column families and/or DB instances.
    ///
    /// <https://github.com/facebook/rocksdb/wiki/Write-Buffer-Manager>
    ///
    /// Users can enable this control by 2 ways:
    ///
    /// 1- Limit the total memtable usage across multiple column families and DBs under a threshold.
    /// 2- Cost the memtable memory usage to block cache so that memory of RocksDB can be capped by the single limit.
    ///
    /// The usage of a write buffer manager is similar to rate_limiter and sst_file_manager.
    /// Users can create one write buffer manager object and pass it to all the options of column families
    /// or DBs whose memtable size they want to be controlled by this object.
    pub fn set_write_buffer_manager(&mut self, write_buffer_manager: &WriteBufferManagerPy) {
        self.inner_opt
            .set_write_buffer_manager(&write_buffer_manager.0)
    }

    /// If true, working thread may avoid doing unnecessary and long-latency
    /// operation (such as deleting obsolete files directly or deleting memtable)
    /// and will instead schedule a background job to do it.
    ///
    /// Use it if you're latency-sensitive.
    ///
    /// Default: false (disabled)
    pub fn set_avoid_unnecessary_blocking_io(&mut self, val: bool) {
        self.inner_opt.set_avoid_unnecessary_blocking_io(val)
    }

    /// Use to control write rate of flush and compaction. Flush has higher
    /// priority than compaction.
    /// If rate limiter is enabled, bytes_per_sync is set to 1MB by default.
    ///
    /// Default: disable
    pub fn set_auto_tuned_ratelimiter(
        &mut self,
        rate_bytes_per_sec: i64,
        refill_period_us: i64,
        fairness: i32,
    ) {
        self.inner_opt
            .set_auto_tuned_ratelimiter(rate_bytes_per_sec, refill_period_us, fairness)
    }
}

#[pymethods]
impl WriteOptionsPy {
    #[new]
    pub fn new() -> Self {
        WriteOptionsPy {
            sync: false,
            disable_wal: false,
            ignore_missing_column_families: false,
            no_slowdown: false,
            low_pri: false,
            memtable_insert_hint_per_batch: false,
        }
    }
}

impl From<&WriteOptionsPy> for WriteOptions {
    fn from(w_opt: &WriteOptionsPy) -> Self {
        let mut opt = WriteOptions::default();
        opt.set_sync(w_opt.sync);
        opt.disable_wal(w_opt.disable_wal);
        opt.set_ignore_missing_column_families(w_opt.ignore_missing_column_families);
        opt.set_low_pri(w_opt.low_pri);
        opt.set_memtable_insert_hint_per_batch(w_opt.memtable_insert_hint_per_batch);
        opt.set_no_slowdown(w_opt.no_slowdown);
        opt
    }
}

/// Optionally wait for the memtable flush to be performed.
///
/// Example:
///     ::
///
///         from rocksdict import Rdict, Options, FlushOptions
///         path = "_path_for_rocksdb_storageY2"
///         db = Rdict(path, Options())
///         flush_options = FlushOptions()
///         flush_options.set_wait(True)
///
///         db.flush_opt(flush_options)
///
///         Rdict.destroy(path, Options())
#[pymethods]
impl FlushOptionsPy {
    #[new]
    pub fn new() -> Self {
        FlushOptionsPy { wait: true }
    }
}

impl From<&FlushOptionsPy> for FlushOptions {
    fn from(f_opt: &FlushOptionsPy) -> Self {
        let mut opt = FlushOptions::default();
        opt.set_wait(f_opt.wait);
        opt
    }
}

#[pymethods]
impl ReadOptionsPy {
    #[new]
    pub fn default(py: Python) -> PyResult<Self> {
        Ok(Self {
            fill_cache: true,
            iterate_upper_bound: py.None(),
            iterate_lower_bound: py.None(),
            prefix_same_as_start: false,
            total_order_seek: false,
            max_skippable_internal_keys: 0,
            background_purge_on_iterator_cleanup: false,
            ignore_range_deletions: false,
            verify_checksums: true,
            readahead_size: 0,
            tailing: false,
            pin_data: false,
            async_io: false,
        })
    }

    /// Specify whether the "data block"/"index block"/"filter block"
    /// read for this iteration should be cached in memory?
    /// Callers may wish to set this field to false for bulk scans.
    ///
    /// Default: true
    pub fn fill_cache(&mut self, v: bool) {
        self.fill_cache = v
    }

    /// Sets the upper bound for an iterator.
    pub fn set_iterate_upper_bound(&mut self, key: &Bound<PyAny>, py: Python) -> PyResult<()> {
        self.iterate_upper_bound = key.to_object(py);
        Ok(())
    }

    /// Sets the lower bound for an iterator.
    pub fn set_iterate_lower_bound(&mut self, key: &Bound<PyAny>, py: Python) -> PyResult<()> {
        self.iterate_upper_bound = key.to_object(py);
        Ok(())
    }

    /// Enforce that the iterator only iterates over the same
    /// prefix as the seek.
    /// This option is effective only for prefix seeks, i.e. prefix_extractor is
    /// non-null for the column family and total_order_seek is false.  Unlike
    /// iterate_upper_bound, prefix_same_as_start only works within a prefix
    /// but in both directions.
    ///
    /// Default: false
    pub fn set_prefix_same_as_start(&mut self, v: bool) {
        self.prefix_same_as_start = v
    }

    /// Enable a total order seek regardless of index format (e.g. hash index)
    /// used in the table. Some table format (e.g. plain table) may not support
    /// this option.
    ///
    /// If true when calling Get(), we also skip prefix bloom when reading from
    /// block based table. It provides a way to read existing data after
    /// changing implementation of prefix extractor.
    pub fn set_total_order_seek(&mut self, v: bool) {
        self.total_order_seek = v
    }

    /// Sets a threshold for the number of keys that can be skipped
    /// before failing an iterator seek as incomplete. The default value of 0 should be used to
    /// never fail a request as incomplete, even on skipping too many keys.
    ///
    /// Default: 0
    pub fn set_max_skippable_internal_keys(&mut self, num: u64) {
        self.max_skippable_internal_keys = num
    }

    /// If true, when PurgeObsoleteFile is called in CleanupIteratorState, we schedule a background job
    /// in the flush job queue and delete obsolete files in background.
    ///
    /// Default: false
    pub fn set_background_purge_on_iterator_cleanup(&mut self, v: bool) {
        self.background_purge_on_iterator_cleanup = v
    }

    /// If true, keys deleted using the DeleteRange() API will be visible to
    /// readers until they are naturally deleted during compaction. This improves
    /// read performance in DBs with many range deletions.
    ///
    /// Default: false
    pub fn set_ignore_range_deletions(&mut self, v: bool) {
        self.ignore_range_deletions = v
    }

    /// If true, all data read from underlying storage will be
    /// verified against corresponding checksums.
    ///
    /// Default: true
    pub fn set_verify_checksums(&mut self, v: bool) {
        self.verify_checksums = v
    }

    /// If non-zero, an iterator will create a new table reader which
    /// performs reads of the given size. Using a large size (> 2MB) can
    /// improve the performance of forward iteration on spinning disks.
    /// Default: 0
    ///
    /// from rocksdict import ReadOptions
    ///
    /// opts = ReadOptions()
    /// opts.set_readahead_size(4_194_304) # 4mb
    pub fn set_readahead_size(&mut self, v: usize) {
        self.readahead_size = v
    }

    /// If true, create a tailing iterator. Note that tailing iterators
    /// only support moving in the forward direction. Iterating in reverse
    /// or seek_to_last are not supported.
    pub fn set_tailing(&mut self, v: bool) {
        self.tailing = v
    }

    /// Specifies the value of "pin_data". If true, it keeps the blocks
    /// loaded by the iterator pinned in memory as long as the iterator is not deleted,
    /// If used when reading from tables created with
    /// BlockBasedTableOptions::use_delta_encoding = false,
    /// Iterator's property "rocksdb.iterator.is-key-pinned" is guaranteed to
    /// return 1.
    ///
    /// Default: false
    pub fn set_pin_data(&mut self, v: bool) {
        self.pin_data = v
    }

    /// Asynchronously prefetch some data.
    ///
    /// Used for sequential reads and internal automatic prefetching.
    ///
    /// Default: `false`
    pub fn set_async_io(&mut self, v: bool) {
        self.async_io = v
    }
}

impl ReadOptionsPy {
    pub(crate) fn to_read_options(&self, raw_mode: bool, py: Python) -> PyResult<ReadOptions> {
        let mut opt = ReadOptions::default();
        opt.fill_cache(self.fill_cache);
        if !self.iterate_lower_bound.is_none(py) {
            let lower_bound = encode_key(self.iterate_lower_bound.bind(py), raw_mode)?;
            opt.set_iterate_lower_bound(lower_bound);
        }
        if !self.iterate_upper_bound.is_none(py) {
            let upper_bound = encode_key(self.iterate_upper_bound.bind(py), raw_mode)?;
            opt.set_iterate_upper_bound(upper_bound);
        }
        opt.set_prefix_same_as_start(self.prefix_same_as_start);
        opt.set_total_order_seek(self.total_order_seek);
        opt.set_max_skippable_internal_keys(self.max_skippable_internal_keys);
        opt.set_background_purge_on_iterator_cleanup(self.background_purge_on_iterator_cleanup);
        opt.set_ignore_range_deletions(self.ignore_range_deletions);
        opt.set_verify_checksums(self.verify_checksums);
        opt.set_readahead_size(self.readahead_size);
        opt.set_tailing(self.tailing);
        opt.set_pin_data(self.pin_data);
        opt.set_async_io(self.async_io);
        Ok(opt)
    }

    pub(crate) fn to_read_opt(&self, raw_mode: bool, py: Python) -> PyResult<ReadOpt> {
        let opt = unsafe { ReadOpt(librocksdb_sys::rocksdb_readoptions_create()) };
        if !self.iterate_lower_bound.is_none(py) {
            let lower_bound = encode_key(self.iterate_lower_bound.bind(py), raw_mode)?;

            unsafe {
                librocksdb_sys::rocksdb_readoptions_set_iterate_lower_bound(
                    opt.0,
                    lower_bound.as_ptr() as *const c_char,
                    lower_bound.len() as size_t,
                );
            }
        }
        if !self.iterate_upper_bound.is_none(py) {
            let upper_bound = encode_key(self.iterate_upper_bound.bind(py), raw_mode)?;

            unsafe {
                librocksdb_sys::rocksdb_readoptions_set_iterate_upper_bound(
                    opt.0,
                    upper_bound.as_ptr() as *const c_char,
                    upper_bound.len() as size_t,
                );
            }
        }
        unsafe {
            librocksdb_sys::rocksdb_readoptions_set_fill_cache(opt.0, self.fill_cache as c_uchar);
            librocksdb_sys::rocksdb_readoptions_set_prefix_same_as_start(
                opt.0,
                self.prefix_same_as_start as c_uchar,
            );
            librocksdb_sys::rocksdb_readoptions_set_total_order_seek(
                opt.0,
                self.total_order_seek as c_uchar,
            );
            librocksdb_sys::rocksdb_readoptions_set_max_skippable_internal_keys(
                opt.0,
                self.max_skippable_internal_keys,
            );
            librocksdb_sys::rocksdb_readoptions_set_background_purge_on_iterator_cleanup(
                opt.0,
                self.background_purge_on_iterator_cleanup as c_uchar,
            );
            librocksdb_sys::rocksdb_readoptions_set_ignore_range_deletions(
                opt.0,
                self.ignore_range_deletions as c_uchar,
            );
            librocksdb_sys::rocksdb_readoptions_set_verify_checksums(
                opt.0,
                self.verify_checksums as c_uchar,
            );
            librocksdb_sys::rocksdb_readoptions_set_readahead_size(
                opt.0,
                self.readahead_size as size_t,
            );
            librocksdb_sys::rocksdb_readoptions_set_tailing(opt.0, self.tailing as c_uchar);
            librocksdb_sys::rocksdb_readoptions_set_pin_data(opt.0, self.pin_data as c_uchar);
        }
        Ok(opt)
    }
}

unsafe impl Send for ReadOpt {}

unsafe impl Sync for ReadOpt {}

impl Drop for ReadOpt {
    fn drop(&mut self) {
        unsafe { librocksdb_sys::rocksdb_readoptions_destroy(self.0) }
    }
}

#[pymethods]
impl MemtableFactoryPy {
    #[staticmethod]
    pub fn vector() -> Self {
        MemtableFactoryPy(MemtableFactory::Vector)
    }

    #[staticmethod]
    pub fn hash_skip_list(bucket_count: usize, height: i32, branching_factor: i32) -> Self {
        MemtableFactoryPy(MemtableFactory::HashSkipList {
            bucket_count,
            height,
            branching_factor,
        })
    }

    #[staticmethod]
    pub fn hash_link_list(bucket_count: usize) -> Self {
        MemtableFactoryPy(MemtableFactory::HashLinkList { bucket_count })
    }
}

#[pymethods]
impl BlockBasedOptionsPy {
    #[new]
    pub fn default() -> Self {
        BlockBasedOptionsPy(BlockBasedOptions::default())
    }

    /// Approximate size of user data packed per block. Note that the
    /// block size specified here corresponds to uncompressed data. The
    /// actual size of the unit read from disk may be smaller if
    /// compression is enabled. This parameter can be changed dynamically.
    pub fn set_block_size(&mut self, size: usize) {
        self.0.set_block_size(size)
    }

    /// Block size for partitioned metadata. Currently applied to indexes when
    /// kTwoLevelIndexSearch is used and to filters when partition_filters is used.
    /// Note: Since in the current implementation the filters and index partitions
    /// are aligned, an index/filter block is created when either index or filter
    /// block size reaches the specified limit.
    ///
    /// Note: this limit is currently applied to only index blocks; a filter
    /// partition is cut right after an index block is cut.
    pub fn set_metadata_block_size(&mut self, size: usize) {
        self.0.set_metadata_block_size(size)
    }

    /// Note: currently this option requires kTwoLevelIndexSearch to be set as
    /// well.
    ///
    /// Use partitioned full filters for each SST file. This option is
    /// incompatible with block-based filters.
    pub fn set_partition_filters(&mut self, size: bool) {
        self.0.set_partition_filters(size)
    }

    /// Sets global cache for blocks (user data is stored in a set of blocks, and
    /// a block is the unit of reading from disk). Cache must outlive DB instance which uses it.
    ///
    /// If set, use the specified cache for blocks.
    /// By default, rocksdb will automatically create and use an 8MB internal cache.
    pub fn set_block_cache(&mut self, cache: &CachePy) {
        self.0.set_block_cache(&cache.0)
    }

    /// Disable block cache
    pub fn disable_cache(&mut self) {
        self.0.disable_cache()
    }

    /// Sets the filter policy to reduce disk read
    pub fn set_bloom_filter(&mut self, bits_per_key: c_double, block_based: bool) {
        self.0.set_bloom_filter(bits_per_key, block_based)
    }

    pub fn set_cache_index_and_filter_blocks(&mut self, v: bool) {
        self.0.set_cache_index_and_filter_blocks(v)
    }

    /// Defines the index type to be used for SS-table lookups.
    ///
    /// Example:
    ///     ::
    ///
    ///         from rocksdict import BlockBasedOptions, BlockBasedIndexType, Options
    ///
    ///         opts = Options()
    ///         block_opts = BlockBasedOptions()
    ///         block_opts.set_index_type(BlockBasedIndexType.hash_search())
    ///         opts.set_block_based_table_factory(block_opts)
    pub fn set_index_type(&mut self, index_type: &BlockBasedIndexTypePy) {
        self.0.set_index_type(match index_type.0 {
            BlockBasedIndexType::BinarySearch => BlockBasedIndexType::BinarySearch,
            BlockBasedIndexType::HashSearch => BlockBasedIndexType::HashSearch,
            BlockBasedIndexType::TwoLevelIndexSearch => BlockBasedIndexType::TwoLevelIndexSearch,
        })
    }

    /// If cache_index_and_filter_blocks is true and the below is true, then
    /// filter and index blocks are stored in the cache, but a reference is
    /// held in the "table reader" object so the blocks are pinned and only
    /// evicted from cache when the table reader is freed.
    ///
    /// Default: false.
    pub fn set_pin_l0_filter_and_index_blocks_in_cache(&mut self, v: bool) {
        self.0.set_pin_l0_filter_and_index_blocks_in_cache(v)
    }

    /// If cache_index_and_filter_blocks is true and the below is true, then
    /// the top-level index of partitioned filter and index blocks are stored in
    /// the cache, but a reference is held in the "table reader" object so the
    /// blocks are pinned and only evicted from cache when the table reader is
    /// freed. This is not limited to l0 in LSM tree.
    ///
    /// Default: false.
    pub fn set_pin_top_level_index_and_filter(&mut self, v: bool) {
        self.0.set_pin_top_level_index_and_filter(v)
    }

    /// Format version, reserved for backward compatibility.
    ///
    /// See full [list](https://github.com/facebook/rocksdb/blob/f059c7d9b96300091e07429a60f4ad55dac84859/include/rocksdb/table.h#L249-L274)
    /// of the supported versions.
    ///
    /// Default: 2.
    pub fn set_format_version(&mut self, version: i32) {
        self.0.set_format_version(version)
    }

    /// Number of keys between restart points for delta encoding of keys.
    /// This parameter can be changed dynamically. Most clients should
    /// leave this parameter alone. The minimum value allowed is 1. Any smaller
    /// value will be silently overwritten with 1.
    ///
    /// Default: 16.
    pub fn set_block_restart_interval(&mut self, interval: i32) {
        self.0.set_block_restart_interval(interval)
    }

    /// Same as block_restart_interval but used for the index block.
    /// If you don't plan to run RocksDB before version 5.16 and you are
    /// using `index_block_restart_interval` > 1, you should
    /// probably set the `format_version` to >= 4 as it would reduce the index size.
    ///
    /// Default: 1.
    pub fn set_index_block_restart_interval(&mut self, interval: i32) {
        self.0.set_index_block_restart_interval(interval)
    }

    /// Set the data block index type for point lookups:
    ///  `DataBlockIndexType::BinarySearch` to use binary search within the data block.
    ///  `DataBlockIndexType::BinaryAndHash` to use the data block hash index in combination with
    ///  the normal binary search.
    ///
    /// The hash table utilization ratio is adjustable using [`set_data_block_hash_ratio`](#method.set_data_block_hash_ratio), which is
    /// valid only when using `DataBlockIndexType::BinaryAndHash`.
    ///
    /// Default: `BinarySearch`
    /// Example:
    ///     ::
    ///
    ///         from rocksdict import BlockBasedOptions, BlockBasedIndexType, Options
    ///
    ///         opts = Options()
    ///         block_opts = BlockBasedOptions()
    ///         block_opts.set_data_block_index_type(DataBlockIndexType.binary_and_hash())
    ///         block_opts.set_data_block_hash_ratio(0.85)
    ///         opts.set_block_based_table_factory(block_opts)
    pub fn set_data_block_index_type(&mut self, index_type: &DataBlockIndexTypePy) {
        self.0.set_data_block_index_type(match index_type.0 {
            DataBlockIndexType::BinarySearch => DataBlockIndexType::BinarySearch,
            DataBlockIndexType::BinaryAndHash => DataBlockIndexType::BinaryAndHash,
        })
    }

    /// Set the data block hash index utilization ratio.
    ///
    /// The smaller the utilization ratio, the less hash collisions happen, and so reduce the risk for a
    /// point lookup to fall back to binary search due to the collisions. A small ratio means faster
    /// lookup at the price of more space overhead.
    ///
    /// Default: 0.75
    pub fn set_data_block_hash_ratio(&mut self, ratio: f64) {
        self.0.set_data_block_hash_ratio(ratio)
    }

    /// Use the specified checksum type.
    /// Newly created table files will be protected with this checksum type.
    /// Old table files will still be readable, even though they have different checksum type.
    pub fn set_checksum_type(&mut self, checksum_type: ChecksumTypePy) {
        self.0.set_checksum_type(checksum_type.0)
    }
}

#[pymethods]
impl CuckooTableOptionsPy {
    #[new]
    pub fn default() -> Self {
        CuckooTableOptionsPy(CuckooTableOptions::default())
    }

    /// Determines the utilization of hash tables. Smaller values
    /// result in larger hash tables with fewer collisions.
    /// Default: 0.9
    pub fn set_hash_ratio(&mut self, ratio: f64) {
        self.0.set_hash_ratio(ratio)
    }

    /// A property used by builder to determine the depth to go to
    /// to search for a path to displace elements in case of
    /// collision. See Builder.MakeSpaceForKey method. Higher
    /// values result in more efficient hash tables with fewer
    /// lookups but take more time to build.
    /// Default: 100
    pub fn set_max_search_depth(&mut self, depth: u32) {
        self.0.set_max_search_depth(depth)
    }

    /// In case of collision while inserting, the builder
    /// attempts to insert in the next cuckoo_block_size
    /// locations before skipping over to the next Cuckoo hash
    /// function. This makes lookups more cache friendly in case
    /// of collisions.
    /// Default: 5
    pub fn set_cuckoo_block_size(&mut self, size: u32) {
        self.0.set_cuckoo_block_size(size)
    }

    /// If this option is enabled, user key is treated as uint64_t and its value
    /// is used as hash value directly. This option changes builder's behavior.
    /// Reader ignore this option and behave according to what specified in
    /// table property.
    /// Default: false
    pub fn set_identity_as_first_hash(&mut self, flag: bool) {
        self.0.set_identity_as_first_hash(flag)
    }

    /// If this option is set to true, module is used during hash calculation.
    /// This often yields better space efficiency at the cost of performance.
    /// If this option is set to false, # of entries in table is constrained to
    /// be power of two, and bit and is used to calculate hash, which is faster in general.
    /// Default: true
    pub fn set_use_module_hash(&mut self, flag: bool) {
        self.0.set_use_module_hash(flag)
    }
}

#[pymethods]
impl PlainTableFactoryOptionsPy {
    #[new]
    pub fn default() -> Self {
        PlainTableFactoryOptionsPy {
            user_key_length: 0,
            bloom_bits_per_key: 10,
            hash_table_ratio: 0.75,
            index_sparseness: 16,
            huge_page_tlb_size: 0,
            encoding_type: KeyEncodingTypePy(KeyEncodingType::Plain),
            full_scan_mode: false,
            store_index_in_file: false,
        }
    }
}

impl PlainTableFactoryOptionsPy {
    fn to_opt(&self, raw_mode: bool) -> PlainTableFactoryOptions {
        PlainTableFactoryOptions {
            // One extra byte for python object type
            user_key_length: if raw_mode {
                self.user_key_length
            } else if self.user_key_length > 0 {
                self.user_key_length + 1
            } else {
                0
            },
            bloom_bits_per_key: self.bloom_bits_per_key,
            hash_table_ratio: self.hash_table_ratio,
            index_sparseness: self.index_sparseness,
            huge_page_tlb_size: self.huge_page_tlb_size,
            encoding_type: self.encoding_type.0,
            full_scan_mode: self.full_scan_mode,
            store_index_in_file: self.store_index_in_file,
        }
    }
}

#[pymethods]
impl CachePy {
    /// Create a lru cache with capacity
    #[new]
    pub fn new_lru_cache(capacity: size_t) -> CachePy {
        CachePy(Cache::new_lru_cache(capacity))
    }

    /// Creates a HyperClockCache with capacity in bytes.
    ///
    /// `estimated_entry_charge` is an important tuning parameter. The optimal
    /// choice at any given time is
    /// `(cache.get_usage() - 64 * cache.get_table_address_count()) /
    /// cache.get_occupancy_count()`, or approximately `cache.get_usage() /
    /// cache.get_occupancy_count()`.
    ///
    /// However, the value cannot be changed dynamically, so as the cache
    /// composition changes at runtime, the following tradeoffs apply:
    ///
    /// * If the estimate is substantially too high (e.g., 25% higher),
    ///   the cache may have to evict entries to prevent load factors that
    ///   would dramatically affect lookup times.
    /// * If the estimate is substantially too low (e.g., less than half),
    ///   then meta data space overhead is substantially higher.
    ///
    /// The latter is generally preferable, and picking the larger of
    /// block size and meta data block size is a reasonable choice that
    /// errs towards this side.
    #[staticmethod]
    pub fn new_hyper_clock_cache(capacity: size_t, estimated_entry_charge: size_t) -> CachePy {
        CachePy(Cache::new_hyper_clock_cache(
            capacity,
            estimated_entry_charge,
        ))
    }

    /// Returns the Cache memory usage
    pub fn get_usage(&self) -> usize {
        self.0.get_usage()
    }

    /// Returns pinned memory usage
    pub fn get_pinned_usage(&self) -> usize {
        self.0.get_pinned_usage()
    }

    /// Sets cache capacity
    pub fn set_capacity(&mut self, capacity: size_t) {
        self.0.set_capacity(capacity)
    }
}

#[pymethods]
impl BlockBasedIndexTypePy {
    /// A space efficient index block that is optimized for
    /// binary-search-based index.
    #[staticmethod]
    pub fn binary_search() -> Self {
        BlockBasedIndexTypePy(BlockBasedIndexType::BinarySearch)
    }

    /// The hash index, if enabled, will perform a hash lookup if
    /// a prefix extractor has been provided through Options::set_prefix_extractor.
    #[staticmethod]
    pub fn hash_search() -> Self {
        BlockBasedIndexTypePy(BlockBasedIndexType::HashSearch)
    }

    /// A two-level index implementation. Both levels are binary search indexes.
    #[staticmethod]
    pub fn two_level_index_search() -> Self {
        BlockBasedIndexTypePy(BlockBasedIndexType::TwoLevelIndexSearch)
    }
}

#[pymethods]
impl DataBlockIndexTypePy {
    /// Use binary search when performing point lookup for keys in data blocks.
    /// This is the default.
    #[staticmethod]
    pub fn binary_search() -> Self {
        DataBlockIndexTypePy(DataBlockIndexType::BinarySearch)
    }

    /// Appends a compact hash table to the end of the data block for efficient indexing. Backwards
    /// compatible with databases created without this feature. Once turned on, existing data will
    /// be gradually converted to the hash index format.
    #[staticmethod]
    pub fn binary_and_hash() -> Self {
        DataBlockIndexTypePy(DataBlockIndexType::BinaryAndHash)
    }
}

#[pymethods]
impl SliceTransformPy {
    #[staticmethod]
    pub fn create_fixed_prefix(len: size_t) -> Self {
        SliceTransformPy(SliceTransformType::Fixed(len))
    }

    ///
    /// prefix max length at `len`. If key is longer than `len`,
    /// the prefix will have length `len`, if key is shorter than `len`,
    /// the prefix will have the same length as `len`.
    ///
    #[staticmethod]
    pub fn create_max_len_prefix(len: usize) -> Self {
        SliceTransformPy(SliceTransformType::MaxLen(len))
    }

    #[staticmethod]
    pub fn create_noop() -> Self {
        SliceTransformPy(SliceTransformType::Noop)
    }
}

#[pymethods]
impl DBPathPy {
    #[new]
    pub fn new(path: &str, target_size: u64) -> Self {
        DBPathPy {
            path: PathBuf::from(path),
            target_size,
        }
    }
}

#[pymethods]
impl DBCompressionTypePy {
    #[staticmethod]
    pub fn none() -> Self {
        DBCompressionTypePy(DBCompressionType::None)
    }

    #[staticmethod]
    pub fn snappy() -> Self {
        DBCompressionTypePy(DBCompressionType::Snappy)
    }

    #[staticmethod]
    pub fn zlib() -> Self {
        DBCompressionTypePy(DBCompressionType::Zlib)
    }

    #[staticmethod]
    pub fn bz2() -> Self {
        DBCompressionTypePy(DBCompressionType::Bz2)
    }

    #[staticmethod]
    pub fn lz4() -> Self {
        DBCompressionTypePy(DBCompressionType::Lz4)
    }

    #[staticmethod]
    pub fn lz4hc() -> Self {
        DBCompressionTypePy(DBCompressionType::Lz4hc)
    }

    #[staticmethod]
    pub fn zstd() -> Self {
        DBCompressionTypePy(DBCompressionType::Zstd)
    }
}

#[pymethods]
impl DBCompactionStylePy {
    #[staticmethod]
    pub fn level() -> Self {
        DBCompactionStylePy(DBCompactionStyle::Level)
    }

    #[staticmethod]
    pub fn universal() -> Self {
        DBCompactionStylePy(DBCompactionStyle::Universal)
    }

    #[staticmethod]
    pub fn fifo() -> Self {
        DBCompactionStylePy(DBCompactionStyle::Fifo)
    }
}

#[pymethods]
impl ChecksumTypePy {
    #[staticmethod]
    pub fn no_checksum() -> Self {
        ChecksumTypePy(ChecksumType::NoChecksum)
    }

    #[staticmethod]
    pub fn crc32c() -> Self {
        ChecksumTypePy(ChecksumType::CRC32c)
    }

    #[staticmethod]
    pub fn xxhash() -> Self {
        ChecksumTypePy(ChecksumType::XXHash)
    }

    #[staticmethod]
    pub fn xxhash64() -> Self {
        ChecksumTypePy(ChecksumType::XXHash64)
    }

    #[staticmethod]
    pub fn xxh3() -> Self {
        ChecksumTypePy(ChecksumType::XXH3)
    }
}

#[pymethods]
impl DBRecoveryModePy {
    #[staticmethod]
    pub fn tolerate_corrupted_tail_records() -> Self {
        DBRecoveryModePy(DBRecoveryMode::TolerateCorruptedTailRecords)
    }

    #[staticmethod]
    pub fn absolute_consistency() -> Self {
        DBRecoveryModePy(DBRecoveryMode::AbsoluteConsistency)
    }

    #[staticmethod]
    pub fn point_in_time() -> Self {
        DBRecoveryModePy(DBRecoveryMode::PointInTime)
    }

    #[staticmethod]
    pub fn skip_any_corrupted_record() -> Self {
        DBRecoveryModePy(DBRecoveryMode::SkipAnyCorruptedRecord)
    }
}

#[pymethods]
impl EnvPy {
    /// Returns default env
    #[new]
    pub fn default() -> PyResult<Self> {
        match Env::new() {
            Ok(env) => Ok(EnvPy(env)),
            Err(e) => Err(PyException::new_err(e.into_string())),
        }
    }

    /// Returns a new environment that stores its data in memory and delegates
    /// all non-file-storage tasks to base_env.
    #[staticmethod]
    pub fn mem_env() -> PyResult<Self> {
        match Env::mem_env() {
            Ok(env) => Ok(EnvPy(env)),
            Err(e) => Err(PyException::new_err(e.into_string())),
        }
    }

    /// Sets the number of background worker threads of a specific thread pool for this environment.
    /// `LOW` is the default pool.
    ///
    /// Default: 1
    pub fn set_background_threads(&mut self, num_threads: c_int) {
        self.0.set_background_threads(num_threads)
    }

    /// Sets the size of the high priority thread pool that can be used to
    /// prevent compactions from stalling memtable flushes.
    pub fn set_high_priority_background_threads(&mut self, n: c_int) {
        self.0.set_high_priority_background_threads(n)
    }

    /// Sets the size of the low priority thread pool that can be used to
    /// prevent compactions from stalling memtable flushes.
    pub fn set_low_priority_background_threads(&mut self, n: c_int) {
        self.0.set_low_priority_background_threads(n)
    }

    /// Sets the size of the bottom priority thread pool that can be used to
    /// prevent compactions from stalling memtable flushes.
    pub fn set_bottom_priority_background_threads(&mut self, n: c_int) {
        self.0.set_bottom_priority_background_threads(n)
    }

    /// Wait for all threads started by StartThread to terminate.
    pub fn join_all_threads(&mut self) {
        self.0.join_all_threads()
    }

    /// Lowering IO priority for threads from the specified pool.
    pub fn lower_thread_pool_io_priority(&mut self) {
        self.0.lower_thread_pool_io_priority()
    }

    /// Lowering IO priority for high priority thread pool.
    pub fn lower_high_priority_thread_pool_io_priority(&mut self) {
        self.0.lower_high_priority_thread_pool_io_priority()
    }

    /// Lowering CPU priority for threads from the specified pool.
    pub fn lower_thread_pool_cpu_priority(&mut self) {
        self.0.lower_thread_pool_cpu_priority()
    }

    /// Lowering CPU priority for high priority thread pool.
    pub fn lower_high_priority_thread_pool_cpu_priority(&mut self) {
        self.0.lower_high_priority_thread_pool_cpu_priority()
    }
}

#[pymethods]
impl UniversalCompactOptionsPy {
    #[new]
    pub fn default() -> Self {
        UniversalCompactOptionsPy {
            size_ratio: 1,
            min_merge_width: 2,
            max_merge_width: c_int::MAX,
            max_size_amplification_percent: 200,
            compression_size_percent: -1,
            stop_style: UniversalCompactionStopStylePy(UniversalCompactionStopStyle::Total),
        }
    }
}

impl From<&UniversalCompactOptionsPy> for UniversalCompactOptions {
    fn from(u_opt: &UniversalCompactOptionsPy) -> Self {
        let mut uni = UniversalCompactOptions::default();
        uni.set_size_ratio(u_opt.size_ratio);
        uni.set_min_merge_width(u_opt.min_merge_width);
        uni.set_max_merge_width(u_opt.max_merge_width);
        uni.set_max_size_amplification_percent(u_opt.max_size_amplification_percent);
        uni.set_compression_size_percent(u_opt.compression_size_percent);
        uni.set_stop_style(u_opt.stop_style.0);
        uni
    }
}

#[pymethods]
impl UniversalCompactionStopStylePy {
    #[staticmethod]
    pub fn similar() -> Self {
        UniversalCompactionStopStylePy(UniversalCompactionStopStyle::Similar)
    }

    #[staticmethod]
    pub fn total() -> Self {
        UniversalCompactionStopStylePy(UniversalCompactionStopStyle::Total)
    }
}

#[pymethods]
impl FifoCompactOptionsPy {
    #[new]
    pub fn new() -> Self {
        FifoCompactOptionsPy {
            max_table_files_size: 0x280000000,
        }
    }
}

impl From<&FifoCompactOptionsPy> for FifoCompactOptions {
    fn from(f_opt: &FifoCompactOptionsPy) -> Self {
        let mut opt = FifoCompactOptions::default();
        opt.set_max_table_files_size(f_opt.max_table_files_size);
        opt
    }
}

#[pymethods]
impl IngestExternalFileOptionsPy {
    #[new]
    pub fn new() -> Self {
        IngestExternalFileOptionsPy(IngestExternalFileOptions::default())
    }

    /// Can be set to true to move the files instead of copying them.
    pub fn set_move_files(&mut self, v: bool) {
        self.0.set_move_files(v)
    }

    /// If set to false, an ingested file keys could appear in existing snapshots
    /// that where created before the file was ingested.
    pub fn set_snapshot_consistency(&mut self, v: bool) {
        self.0.set_snapshot_consistency(v)
    }

    /// If set to false, IngestExternalFile() will fail if the file key range
    /// overlaps with existing keys or tombstones in the DB.
    pub fn set_allow_global_seqno(&mut self, v: bool) {
        self.0.set_allow_global_seqno(v)
    }

    /// If set to false and the file key range overlaps with the memtable key range
    /// (memtable flush required), IngestExternalFile will fail.
    pub fn set_allow_blocking_flush(&mut self, v: bool) {
        self.0.set_allow_blocking_flush(v)
    }

    /// Set to true if you would like duplicate keys in the file being ingested
    /// to be skipped rather than overwriting existing data under that key.
    /// Usecase: back-fill of some historical data in the database without
    /// over-writing existing newer version of data.
    /// This option could only be used if the DB has been running
    /// with allow_ingest_behind=true since the dawn of time.
    /// All files will be ingested at the bottommost level with seqno=0.
    pub fn set_ingest_behind(&mut self, v: bool) {
        self.0.set_ingest_behind(v)
    }
}

#[pymethods]
impl BottommostLevelCompactionPy {
    /// Skip bottommost level compaction
    #[staticmethod]
    pub fn skip() -> Self {
        BottommostLevelCompactionPy(BottommostLevelCompaction::Skip)
    }

    /// Only compact bottommost level if there is a compaction filter
    /// This is the default option
    #[staticmethod]
    pub fn if_have_compaction_filter() -> Self {
        BottommostLevelCompactionPy(BottommostLevelCompaction::IfHaveCompactionFilter)
    }

    /// Always compact bottommost level
    #[staticmethod]
    pub fn force() -> Self {
        BottommostLevelCompactionPy(BottommostLevelCompaction::Force)
    }

    /// Always compact bottommost level but in bottommost level avoid
    /// double-compacting files created in the same compaction
    #[staticmethod]
    pub fn force_optimized() -> Self {
        BottommostLevelCompactionPy(BottommostLevelCompaction::ForceOptimized)
    }
}

#[pymethods]
impl CompactOptionsPy {
    #[new]
    pub fn default() -> Self {
        CompactOptionsPy(CompactOptions::default())
    }

    /// If more than one thread calls manual compaction,
    /// only one will actually schedule it while the other threads will simply wait
    /// for the scheduled manual compaction to complete. If exclusive_manual_compaction
    /// is set to true, the call will disable scheduling of automatic compaction jobs
    /// and wait for existing automatic compaction jobs to finish.
    pub fn set_exclusive_manual_compaction(&mut self, v: bool) {
        self.0.set_exclusive_manual_compaction(v)
    }

    /// Sets bottommost level compaction.
    pub fn set_bottommost_level_compaction(&mut self, lvl: BottommostLevelCompactionPy) {
        self.0.set_bottommost_level_compaction(lvl.0)
    }

    /// If true, compacted files will be moved to the minimum level capable
    /// of holding the data or given level (specified non-negative target_level).
    pub fn set_change_level(&mut self, v: bool) {
        self.0.set_change_level(v)
    }

    /// If change_level is true and target_level have non-negative value, compacted
    /// files will be moved to target_level.
    pub fn set_target_level(&mut self, lvl: c_int) {
        self.0.set_target_level(lvl)
    }
}

#[pymethods]
impl WriteBufferManagerPy {
    #[new]
    pub fn new_write_buffer_manager(buffer_size: size_t, allow_stall: bool) -> Self {
        Self(WriteBufferManager::new_write_buffer_manager(
            buffer_size,
            allow_stall,
        ))
    }

    /// Users can set up RocksDB to cost memory used by memtables to block cache.
    /// This can happen no matter whether you enable memtable memory limit or not.
    /// This option is added to manage memory (memtables + block cache) under a single limit.
    ///
    /// buffer_size: the memory limit in bytes.
    /// allow_stall: If set true, it will enable stalling of all writers when memory usage exceeds buffer_size (soft limit).
    ///             It will wait for flush to complete and memory usage to drop down
    /// cache: the block cache instance
    #[staticmethod]
    pub fn new_write_buffer_manager_with_cache(
        buffer_size: size_t,
        allow_stall: bool,
        cache: CachePy,
    ) -> Self {
        Self(WriteBufferManager::new_write_buffer_manager_with_cache(
            buffer_size,
            allow_stall,
            cache.0,
        ))
    }

    /// Returns the WriteBufferManager memory usage in bytes.
    pub fn get_usage(&self) -> usize {
        self.0.get_usage()
    }

    /// Returns the current buffer size in bytes.
    pub fn get_buffer_size(&self) -> usize {
        self.0.get_buffer_size()
    }

    /// Set the buffer size in bytes.
    pub fn set_buffer_size(&self, new_size: usize) {
        self.0.set_buffer_size(new_size)
    }

    /// Returns if WriteBufferManager is enabled.
    pub fn enabled(&self) -> bool {
        self.0.enabled()
    }

    /// set the allow_stall flag.
    pub fn set_allow_stall(&self, allow_stall: bool) {
        self.0.set_allow_stall(allow_stall)
    }
}

#[macro_export]
macro_rules! implement_max_len_transform {
    ($($len:literal),*) => {
        fn create_max_len_transform(len: usize) -> Result<SliceTransform, ()> {
            match len {
                $($len => Ok(SliceTransform::create(
                    "max_len",
                    |slice| {
                        if slice.len() > $len {
                            &slice[0..$len]
                        } else {
                            slice
                        }
                    },
                    None,
                ))),*,
                _ => {
                    Err(())
                }
            }
        }
    };
}

implement_max_len_transform!(
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
    27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
    51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74,
    75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98,
    99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117,
    118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128
);
