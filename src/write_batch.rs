use crate::encoder::{encode_key, encode_value};
use crate::ColumnFamilyPy;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use rocksdb::WriteBatch;

macro_rules! inner_ref {
    ($self:ident) => {
        match &$self.inner {
            Some(inner) => Ok(inner),
            None => Err(PyException::new_err(
                "this batch is already consumed, create a new one by calling `WriteBatch()`",
            )),
        }
    };
}

macro_rules! inner_mut {
    ($self:ident) => {
        match &mut $self.inner {
            Some(inner) => Ok(inner),
            None => Err(PyException::new_err(
                "this batch is already consumed, create a new one by calling `WriteBatch()`",
            )),
        }
    };
}

/// WriteBatch class. Use db.write() to ingest WriteBatch.
///
/// Notes:
///     A WriteBatch instance can only be ingested once,
///     otherwise an Exception will be raised.
///
/// Args:
///     raw_mode (bool): make sure that this is consistent with the Rdict.
#[pyclass(name = "WriteBatch")]
pub(crate) struct WriteBatchPy {
    inner: Option<WriteBatch>,
    default_column_family: Option<ColumnFamilyPy>,
    dumps: PyObject,
    pub(crate) raw_mode: bool,
}

#[pymethods]
impl WriteBatchPy {
    /// WriteBatch class. Use db.write() to ingest WriteBatch.
    ///
    /// Notes:
    ///     A WriteBatch instance can only be ingested once,
    ///     otherwise an Exception will be raised.
    ///
    /// Args:
    ///     raw_mode (bool): make sure that this is consistent with the Rdict.
    #[new]
    #[pyo3(signature = (raw_mode = false))]
    pub fn default(py: Python, raw_mode: bool) -> PyResult<Self> {
        let pickle = PyModule::import(py, "pickle")?.unbind();
        Ok(WriteBatchPy {
            inner: Some(WriteBatch::default()),
            default_column_family: None,
            dumps: pickle.getattr(py, "dumps")?,
            raw_mode,
        })
    }

    /// change to a custom dumps function
    pub fn set_dumps(&mut self, dumps: PyObject) {
        self.dumps = dumps
    }

    pub fn __len__(&self) -> PyResult<usize> {
        self.len()
    }

    pub fn __setitem__(&mut self, key: &Bound<PyAny>, value: &Bound<PyAny>) -> PyResult<()> {
        let inner = inner_mut!(self)?;
        let key = encode_key(key, self.raw_mode)?;
        let value = encode_value(value, &self.dumps, self.raw_mode)?;
        match &self.default_column_family {
            None => inner.put(key, value),
            Some(cf) => inner.put_cf(&cf.cf, key, value),
        }
        Ok(())
    }

    pub fn __delitem__(&mut self, key: &Bound<PyAny>) -> PyResult<()> {
        let inner = inner_mut!(self)?;
        let key = encode_key(key, self.raw_mode)?;
        match &self.default_column_family {
            None => inner.delete(key),
            Some(cf) => inner.delete_cf(&cf.cf, key),
        }
        Ok(())
    }

    /// Set the default item for `a[i] = j` and `del a[i]` syntax.
    ///
    /// You can also use `put(key, value, column_family)` to explicitly choose column family.
    ///
    /// Args:
    ///     - column_family (ColumnFamily | None): column family descriptor or None (for default family).
    #[pyo3(signature = (column_family=None))]
    pub fn set_default_column_family(
        &mut self,
        column_family: Option<ColumnFamilyPy>,
    ) -> PyResult<()> {
        self.default_column_family = column_family;
        Ok(())
    }

    /// length of the batch
    pub fn len(&self) -> PyResult<usize> {
        let inner = inner_ref!(self)?;
        Ok(inner.len())
    }

    /// Return WriteBatch serialized size (in bytes).
    pub fn size_in_bytes(&self) -> PyResult<usize> {
        let inner = inner_ref!(self)?;
        Ok(inner.size_in_bytes())
    }

    /// Check whether the batch is empty.
    pub fn is_empty(&self) -> PyResult<bool> {
        let inner = inner_ref!(self)?;
        Ok(inner.is_empty())
    }

    /// Insert a value into the database under the given key.
    ///
    /// Args:
    ///     column_family: override the default column family set by set_default_column_family
    #[pyo3(signature = (key, value, column_family = None))]
    pub fn put(
        &mut self,
        key: &Bound<PyAny>,
        value: &Bound<PyAny>,
        column_family: Option<ColumnFamilyPy>,
    ) -> PyResult<()> {
        let inner = inner_mut!(self)?;
        let key = encode_key(key, self.raw_mode)?;
        let value = encode_value(value, &self.dumps, self.raw_mode)?;
        match column_family {
            Some(cf) => inner.put_cf(&cf.cf, key, value),
            None => inner.put(key, value),
        }
        Ok(())
    }

    /// Insert a wide-column.
    ///
    /// The length of `names` and `values` must be the same.
    ///
    /// Args:
    ///     key: the key.
    ///     names: the names of the columns.
    ///     values: the values of the columns.
    #[inline]
    #[pyo3(signature = (key, names, values))]
    fn put_entity(
        &mut self,
        key: &Bound<PyAny>,
        names: Vec<Bound<PyAny>>,
        values: Vec<Bound<PyAny>>,
    ) -> PyResult<()> {
        let inner = inner_mut!(self)?;
        let key = encode_key(key, self.raw_mode)?;
        let cf = if let Some(cf) = &self.default_column_family {
            cf
        } else {
            return Err(PyException::new_err(
                "Cannot call this method without first WriteBatch.set_default_column_family()",
            ));
        };
        if names.len() != values.len() {
            return Err(PyException::new_err(
                "names and values must have the same length",
            ));
        }
        let mut names_vec = Vec::with_capacity(names.len());
        let mut values_vec = Vec::with_capacity(values.len());
        for name in names.iter() {
            names_vec.push(encode_value(name, &self.dumps, self.raw_mode)?);
        }
        for value in values.iter() {
            values_vec.push(encode_value(value, &self.dumps, self.raw_mode)?);
        }
        inner
            .put_entity_cf_opt(&cf.cf, key, &names_vec, &values_vec)
            .map_err(|e| PyException::new_err(e.to_string()))
    }

    /// Removes the database entry for key. Does nothing if the key was not found.
    ///
    /// Args:
    ///     column_family: override the default column family set by set_default_column_family
    #[pyo3(signature = (key, column_family = None))]
    pub fn delete(
        &mut self,
        key: &Bound<PyAny>,
        column_family: Option<ColumnFamilyPy>,
    ) -> PyResult<()> {
        let inner = inner_mut!(self)?;
        let key = encode_key(key, self.raw_mode)?;
        match column_family {
            Some(cf) => inner.delete_cf(&cf.cf, key),
            None => inner.delete(key),
        }
        Ok(())
    }

    /// Remove database entries in column family from start key to end key.
    ///
    /// Notes:
    ///     Removes the database entries in the range ["begin_key", "end_key"), i.e.,
    ///     including "begin_key" and excluding "end_key". It is not an error if no
    ///     keys exist in the range ["begin_key", "end_key").
    ///
    /// Args:
    ///     begin: begin key
    ///     end: end key
    ///     column_family: override the default column family set by set_default_column_family
    #[pyo3(signature = (begin, end, column_family = None))]
    pub fn delete_range(
        &mut self,
        begin: &Bound<PyAny>,
        end: &Bound<PyAny>,
        column_family: Option<ColumnFamilyPy>,
    ) -> PyResult<()> {
        let inner = inner_mut!(self)?;
        let from = encode_key(begin, self.raw_mode)?;
        let to = encode_key(end, self.raw_mode)?;
        match column_family {
            Some(cf) => inner.delete_range_cf(&cf.cf, from, to),
            None => inner.delete_range(from, to),
        }
        Ok(())
    }

    /// Clear all updates buffered in this batch.
    pub fn clear(&mut self) -> PyResult<()> {
        let inner = inner_mut!(self)?;
        inner.clear();
        Ok(())
    }
}

impl WriteBatchPy {
    #[inline]
    pub(crate) fn consume(&mut self) -> PyResult<WriteBatch> {
        if let Some(inner) = self.inner.take() {
            drop(self.default_column_family.take());
            Ok(inner)
        } else {
            Err(PyException::new_err(
                "this batch is already consumed, create a new one by calling `WriteBatch()`",
            ))
        }
    }
}

unsafe impl Sync for WriteBatchPy {}
