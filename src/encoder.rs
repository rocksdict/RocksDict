use integer_encoding::VarInt;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyFloat, PyInt, PyString};

pub(crate) enum ValueTypes<'a> {
    Bytes(&'a [u8]),
    String(String),
    Int(i64),
    Float(f64),
    Unsupported,
}

#[inline(always)]
pub(crate) fn encoding_byte(v_type: &ValueTypes) -> u8 {
    match v_type {
        ValueTypes::Bytes(_) => 1,
        ValueTypes::String(_) => 2,
        ValueTypes::Int(_) => 3,
        ValueTypes::Float(_) => 4,
        ValueTypes::Unsupported => 0,
    }
}

///
/// Convert string, int, float, bytes to byte encodings.
///
/// The first byte is used for encoding value types
///
#[inline(always)]
pub(crate) fn encode_value(value: &PyAny) -> PyResult<Box<[u8]>> {
    let bytes = py_to_value_types(value)?;
    let type_encoding = encoding_byte(&bytes);
    match bytes {
        ValueTypes::Bytes(value) => Ok(concat_type_encoding(type_encoding, value)),
        ValueTypes::String(value) => Ok(concat_type_encoding(type_encoding, value.as_bytes())),
        ValueTypes::Int(value) => Ok(concat_type_encoding(
            type_encoding,
            &value.encode_var_vec()[..],
        )),
        ValueTypes::Float(value) => Ok(concat_type_encoding(
            type_encoding,
            &value.to_be_bytes()[..],
        )),
        ValueTypes::Unsupported => Err(PyException::new_err(
            "Only support `string`, `int`, `float`, and `bytes` as keys / values",
        )),
    }
}

#[inline(always)]
fn py_to_value_types(value: &PyAny) -> PyResult<ValueTypes> {
    if let Ok(value) = <PyBytes as PyTryFrom>::try_from(value) {
        return Ok(ValueTypes::Bytes(value.as_bytes()));
    }
    if let Ok(value) = <PyString as PyTryFrom>::try_from(value) {
        return Ok(ValueTypes::String(value.to_string()));
    }
    if let Ok(value) = <PyInt as PyTryFrom>::try_from(value) {
        return Ok(ValueTypes::Int(value.extract()?));
    }
    if let Ok(value) = <PyFloat as PyTryFrom>::try_from(value) {
        return Ok(ValueTypes::Float(value.extract()?));
    }
    Ok(ValueTypes::Unsupported)
}


#[inline(always)]
pub(crate) fn decode_value(py: Python, bytes: &[u8]) -> PyResult<PyObject> {
    match bytes.get(0) {
        None => Err(PyException::new_err("Unknown value type")),
        Some(byte) => match byte {
            1 => Ok(PyBytes::new(py, &bytes[1..]).to_object(py)),
            2 => {
                let string = match String::from_utf8(bytes[1..].to_vec()) {
                    Ok(s) => s,
                    Err(_) => return Err(PyException::new_err("utf-8 decoding error")),
                };
                Ok(string.into_py(py))
            }
            3 => {
                if let Some((int, _)) = i64::decode_var(bytes[1..].try_into().unwrap()) {
                    Ok(int.into_py(py))
                } else {
                    Err(PyException::new_err("varint decoding error"))
                }
            }
            4 => {
                let float: f64 = f64::from_be_bytes(bytes[1..].try_into().unwrap());
                Ok(float.into_py(py))
            }
            _ => Err(PyException::new_err("Unknown value type")),
        },
    }
}

#[inline(always)]
fn concat_type_encoding(encoding: u8, payload: &[u8]) -> Box<[u8]> {
    let mut output = Vec::with_capacity(payload.len() + 1);
    output.push(encoding);
    output.extend_from_slice(payload);
    output.into_boxed_slice()
}
