use num_bigint::BigInt;
use pyo3::exceptions::{PyException, PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyFloat, PyInt, PyString};
use std::borrow::Cow;

pub(crate) enum ValueTypes<'a, 'b> {
    Bytes(&'a [u8]),
    String(&'a str),
    Int(BigInt),
    Float(f64),
    Bool(bool),
    Any(&'a Bound<'b, PyAny>),
}

#[inline(always)]
pub(crate) fn encoding_byte(v_type: &ValueTypes) -> u8 {
    match v_type {
        ValueTypes::Bytes(_) => 1,
        ValueTypes::String(_) => 2,
        ValueTypes::Int(_) => 3,
        ValueTypes::Float(_) => 4,
        ValueTypes::Bool(_) => 5,
        ValueTypes::Any(_) => 6,
    }
}

#[inline(always)]
pub(crate) fn encode_key<'a>(key: &'a Bound<PyAny>, raw_mode: bool) -> PyResult<Cow<'a, [u8]>> {
    if raw_mode {
        return if let Ok(value) = key.downcast::<PyBytes>() {
            Ok(Cow::Borrowed(value.as_bytes()))
        } else {
            Err(PyKeyError::new_err("raw mode only support bytes"))
        };
    }
    let bytes = py_to_value_types(key)?;
    let type_encoding = encoding_byte(&bytes);
    let owned_bytes = match bytes {
        ValueTypes::Bytes(value) => Ok(concat_type_encoding(type_encoding, value)),
        ValueTypes::String(value) => Ok(concat_type_encoding(type_encoding, value.as_bytes())),
        ValueTypes::Int(value) => Ok(concat_type_encoding(
            type_encoding,
            &value.to_signed_bytes_be()[..],
        )),
        ValueTypes::Float(value) => Ok(concat_type_encoding(
            type_encoding,
            &value.to_be_bytes()[..],
        )),
        ValueTypes::Bool(value) => Ok(concat_type_encoding(
            type_encoding,
            if value { &[1u8] } else { &[0u8] },
        )),
        ValueTypes::Any(_) => Err(PyException::new_err(
            "Only support `string`, `int`, `float`, `bool`, and `bytes` as keys",
        )),
    }?;
    Ok(Cow::Owned(owned_bytes))
}

///
/// Convert string, int, float, bytes to byte encodings.
///
/// The first byte is used for encoding value types
///
#[inline(always)]
pub(crate) fn encode_value<'a>(
    value: &'a Bound<PyAny>,
    dumps: &PyObject,
    raw_mode: bool,
) -> PyResult<Cow<'a, [u8]>> {
    if raw_mode {
        if let Ok(value) = value.downcast::<PyBytes>() {
            Ok(Cow::Borrowed(value.as_bytes()))
        } else {
            Err(PyValueError::new_err("raw mode only support bytes"))
        }
    } else {
        let bytes = py_to_value_types(value)?;
        let type_encoding = encoding_byte(&bytes);
        let owned_bytes = match bytes {
            ValueTypes::Bytes(value) => concat_type_encoding(type_encoding, value),
            ValueTypes::String(value) => concat_type_encoding(type_encoding, value.as_bytes()),
            ValueTypes::Int(value) => {
                concat_type_encoding(type_encoding, &value.to_signed_bytes_be()[..])
            }
            ValueTypes::Float(value) => {
                concat_type_encoding(type_encoding, &value.to_be_bytes()[..])
            }
            ValueTypes::Bool(value) => {
                concat_type_encoding(type_encoding, if value { &[1u8] } else { &[0u8] })
            }
            ValueTypes::Any(value) => {
                let py = value.py();
                let pickle_bytes = dumps.call1(py, (value,))?;
                let bytes: &[u8] = pickle_bytes.downcast_bound::<PyBytes>(py)?.as_bytes();
                concat_type_encoding(type_encoding, bytes)
            }
        };
        Ok(Cow::Owned(owned_bytes))
    }
}

#[inline(always)]
fn py_to_value_types<'a, 'b>(value: &'a Bound<'b, PyAny>) -> PyResult<ValueTypes<'a, 'b>> {
    if let Ok(value) = value.downcast::<PyBool>() {
        return Ok(ValueTypes::Bool(value.extract()?));
    }
    if let Ok(value) = value.downcast::<PyBytes>() {
        return Ok(ValueTypes::Bytes(value.as_bytes()));
    }
    if let Ok(value) = value.downcast::<PyString>() {
        return Ok(ValueTypes::String(value.to_str()?));
    }
    if let Ok(value) = value.downcast::<PyInt>() {
        return Ok(ValueTypes::Int(value.extract()?));
    }
    if let Ok(value) = value.downcast::<PyFloat>() {
        return Ok(ValueTypes::Float(value.value()));
    }
    Ok(ValueTypes::Any(value))
}

/// this function is used for decoding value from bytes
#[inline(always)]
pub(crate) fn decode_value(
    py: Python,
    bytes: &[u8],
    loads: &PyObject,
    raw_mode: bool,
) -> PyResult<PyObject> {
    // directly return bytes if raw_mode is true
    if raw_mode {
        return Ok(PyBytes::new(py, bytes).to_object(py));
    }
    match bytes.first() {
        // deal with empty value returned by entities
        None => Ok(PyString::new(py, "").to_object(py)),
        Some(byte) => match byte {
            1 => Ok(PyBytes::new(py, &bytes[1..]).to_object(py)),
            2 => {
                let string = match String::from_utf8(bytes[1..].to_vec()) {
                    Ok(s) => s,
                    Err(_) => return Err(PyException::new_err("utf-8 decoding error")),
                };
                Ok(PyString::new(py, &string).to_object(py))
            }
            3 => {
                let big_int = BigInt::from_signed_bytes_be(&bytes[1..]);
                Ok(big_int.to_object(py))
            }
            4 => {
                let float: f64 = f64::from_be_bytes(bytes[1..].try_into().unwrap());
                Ok(float.into_py(py))
            }
            5 => Ok(PyBool::new(py, bytes[1] != 0).to_object(py)),
            6 => loads.call1(py, (PyBytes::new(py, &bytes[1..]),)),
            _ => Err(PyException::new_err("Unknown value type")),
        },
    }
}

#[inline(always)]
fn concat_type_encoding(encoding: u8, payload: &[u8]) -> Vec<u8> {
    let mut output = Vec::with_capacity(payload.len() + 1);
    output.push(encoding);
    output.extend_from_slice(payload);
    output
}
