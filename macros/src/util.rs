use crate::dl_generated::dl;

pub fn i64_to_i8_checked(v: i64) -> Result<i8, String> {
    if v < i8::MIN as i64 || v > i8::MAX as i64 {
        return Err(format!("exponent {} out of i8 range", v));
    }
    Ok(v as i8)
}

pub fn i64_to_u16_checked(v: i64, what: &str) -> Result<u16, String> {
    if v < 0 || v > u16::MAX as i64 {
        return Err(format!("{} {} out of u16 range", what, v));
    }
    Ok(v as u16)
}

/// Flatten raw_data AlignedBytes to a byte vector, preserving the full 16-byte chunked storage.
/// If typed arrays are present instead of raw_data, serialize them in little-endian.
pub fn bytes_from_tensor_full(t: &dl::Tensor) -> Result<Vec<u8>, String> {
    if let Some(raw) = t.raw_data() {
        let mut out = Vec::with_capacity(raw.len() * 16);
        for i in 0..raw.len() {
            let chunk = raw.get(i);
            let arr = chunk.bytes();
            for b in 0..16 {
                out.push(arr.get(b));
            }
        }
        Ok(out)
    } else if let Some(v) = t.int32_data() {
        let mut out = Vec::with_capacity(v.len() * 4);
        for i in 0..v.len() {
            out.extend_from_slice(&v.get(i).to_le_bytes());
        }
        Ok(out)
    } else if let Some(v) = t.float_data() {
        let mut out = Vec::with_capacity(v.len() * 4);
        for i in 0..v.len() {
            out.extend_from_slice(&v.get(i).to_le_bytes());
        }
        Ok(out)
    } else if let Some(v) = t.int64_data() {
        let mut out = Vec::with_capacity(v.len() * 8);
        for i in 0..v.len() {
            out.extend_from_slice(&v.get(i).to_le_bytes());
        }
        Ok(out)
    } else {
        Err("unsupported tensor storage; expected raw_data or typed ints".into())
    }
}
