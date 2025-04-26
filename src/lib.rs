mod checkpoints;
mod db_reference;
mod encoder;
mod exceptions;
mod iter;
mod options;
mod rdict;
mod snapshot;
mod sst_file_writer;
mod util;
mod write_batch;

use crate::exceptions::*;
use crate::iter::*;
use crate::options::*;
use crate::rdict::*;
use crate::snapshot::Snapshot;
use crate::sst_file_writer::*;
use crate::write_batch::*;
use checkpoints::CheckpointPy;
use pyo3::prelude::*;

/// ## Abstract
///
/// This package enables users to store, query, and delete
/// a large number of key-value pairs on disk.
///
/// This is especially useful when the data cannot fit into RAM.
/// If you have hundreds of GBs or many TBs of key-value data to store
/// and query from, this is the package for you.
///
/// ### Installation
///
/// This package is built for macOS (amd64/arm64), Windows , and Linux amd64/arm64.
/// It can be installed from pypi with `pip install rocksdict`.
///
/// ## Introduction
///
/// Below is a code example that shows how to do the following:
///
/// - Create Rdict
/// - Store something on disk
/// - Close Rdict
/// - Open Rdict again
/// - Check Rdict elements
/// - Iterate from Rdict
/// - Batch get
/// - Delete storage
///
/// Examples:
///     ::
///
///         from rocksdict import Rdict, Options
///
///         path = str("./test_dict")
///
///         # create a Rdict with default options at `path`
///         db = Rdict(path)
///
///         # storing numbers
///         db[1.0] = 1
///         db[1] = 1.0
///         db["huge integer"] = 2343546543243564534233536434567543
///         db["good"] = True
///         db["bad"] = False
///         db["bytes"] = b"bytes"
///         db["this is a list"] = [1, 2, 3]
///         db["store a dict"] = {0: 1}
///
///         # for example numpy array
///         import numpy as np
///         import pandas as pd
///         db[b"numpy"] = np.array([1, 2, 3])
///         db["a table"] = pd.DataFrame({"a": [1, 2], "b": [2, 1]})
///
///         # close Rdict
///         db.close()
///
///         # reopen Rdict from disk
///         db = Rdict(path)
///         assert db[1.0] == 1
///         assert db[1] == 1.0
///         assert db["huge integer"] == 2343546543243564534233536434567543
///         assert db["good"] == True
///         assert db["bad"] == False
///         assert db["bytes"] == b"bytes"
///         assert db["this is a list"] == [1, 2, 3]
///         assert db["store a dict"] == {0: 1}
///         assert np.all(db[b"numpy"] == np.array([1, 2, 3]))
///         assert np.all(db["a table"] == pd.DataFrame({"a": [1, 2], "b": [2, 1]}))
///
///         # iterate through all elements
///         for k, v in db.items():
///             print(f"{k} -> {v}")
///
///         # batch get:
///         print(db[["good", "bad", 1.0]])
///         # [True, False, 1]
///
///         # delete Rdict from dict
///         db.close()
///         Rdict.destroy(path)
///
/// Supported types:
///
/// - key: `int, float, bool, str, bytes`
/// - value: `int, float, bool, str, bytes` and anything that
///     supports `pickle`.
///
#[pymodule]
fn rocksdict(py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<Rdict>()?;
    m.add_class::<OptionsPy>()?;
    m.add_class::<MemtableFactoryPy>()?;
    m.add_class::<BlockBasedOptionsPy>()?;
    m.add_class::<CuckooTableOptionsPy>()?;
    m.add_class::<PlainTableFactoryOptionsPy>()?;
    m.add_class::<CachePy>()?;
    m.add_class::<BlockBasedIndexTypePy>()?;
    m.add_class::<DataBlockIndexTypePy>()?;
    m.add_class::<SliceTransformPy>()?;
    m.add_class::<DBPathPy>()?;
    m.add_class::<WriteOptionsPy>()?;
    m.add_class::<FlushOptionsPy>()?;
    m.add_class::<ReadOptionsPy>()?;
    m.add_class::<DBCompressionTypePy>()?;
    m.add_class::<DBCompactionStylePy>()?;
    m.add_class::<DBRecoveryModePy>()?;
    m.add_class::<UniversalCompactOptionsPy>()?;
    m.add_class::<UniversalCompactionStopStylePy>()?;
    m.add_class::<EnvPy>()?;
    m.add_class::<FifoCompactOptionsPy>()?;
    m.add_class::<RdictIter>()?;
    m.add_class::<RdictItems>()?;
    m.add_class::<RdictValues>()?;
    m.add_class::<RdictKeys>()?;
    m.add_class::<RdictColumns>()?;
    m.add_class::<RdictEntities>()?;
    m.add_class::<IngestExternalFileOptionsPy>()?;
    m.add_class::<SstFileWriterPy>()?;
    m.add_class::<WriteBatchPy>()?;
    m.add_class::<ColumnFamilyPy>()?;
    m.add_class::<AccessType>()?;
    m.add_class::<Snapshot>()?;
    m.add_class::<CompactOptionsPy>()?;
    m.add_class::<BottommostLevelCompactionPy>()?;
    m.add_class::<ChecksumTypePy>()?;
    m.add_class::<KeyEncodingTypePy>()?;
    m.add_class::<WriteBufferManagerPy>()?;
    m.add_class::<CheckpointPy>()?;

    m.add("DbClosedError", py.get_type::<DbClosedError>())?;

    Ok(())
}
