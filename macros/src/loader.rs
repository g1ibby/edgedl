use std::path::Path;

/// Resolve the model path relative to `CARGO_MANIFEST_DIR` and common fallbacks,
/// then read and return the file bytes.
pub fn resolve_and_read_model(manifest_dir: &str, rel_path: &str) -> Result<Vec<u8>, String> {
    let abs = Path::new(manifest_dir).join(rel_path);

    // Helper to try reading a candidate path
    let try_path = |p: &Path| -> Option<Vec<u8>> { std::fs::read(p).ok() };

    if let Some(bytes) = try_path(&abs) {
        return Ok(bytes);
    }

    Err(format!("failed to read {}", abs.display()))
}

/// Validate and strip the 16-byte ESPDL header, returning the FlatBuffers payload slice.
pub fn parse_edl_header(bytes: &[u8]) -> Result<&[u8], String> {
    if bytes.len() < 16 {
        return Err(format!(".espdl file too small: {} bytes", bytes.len()));
    }
    let magic = &bytes[0..4];
    if magic != b"EDL1" && magic != b"EDL2" {
        return Err(format!("bad magic: {:?}", magic));
    }
    let enc = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    if enc != 0 {
        return Err("encrypted ESPDL model is not supported".to_string());
    }
    let len = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
    if bytes.len() < 16 + len {
        return Err(format!(
            "truncated file: have {} bytes, need {}",
            bytes.len(),
            16 + len
        ));
    }
    Ok(&bytes[16..16 + len])
}
