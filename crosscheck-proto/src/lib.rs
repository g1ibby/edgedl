//! Shared wire format for tensor-probe RTT frames.
//!
//! Both `hil-test` (device, `no_std`) and `probe-tool` (host) link to this crate
//! so the byte layout has a single source of truth. Extending the format means
//! touching one struct here; both sides pick up the change.
//!
//! Frame (little-endian):
//!
//! ```text
//! u32   magic = 0xED_D1_DA_7A
//! u16   run_tag_len
//! [u8]  run_tag
//! [u8; 14] header (see FrameHeader::to_le_bytes)
//! u32   payload_len
//! [u8]  payload
//! u16   fletcher16(magic..=payload)
//! ```

#![cfg_attr(not(any(test, feature = "alloc")), no_std)]
#![forbid(unsafe_code)]

#[cfg(feature = "alloc")]
extern crate alloc;

/// Frame magic — also used as a resync anchor by the host parser.
pub const MAGIC: u32 = 0xED_D1_DA_7A;

/// Fixed-size header portion written after the run_tag and before payload_len.
pub const HEADER_LEN: usize = 14;

// Op-type byte. Must stay in sync with the device-side `op_type_byte` that
// maps `edgedl::model::NodeOp` variants to these constants.
pub const OP_CONV2D: u8 = 0;
pub const OP_PAD: u8 = 1;
pub const OP_REDUCE_MEAN: u8 = 2;
pub const OP_LINEAR: u8 = 3;
pub const OP_RELU: u8 = 4;

/// Map an op-type byte to its stable textual tag (matches
/// `edgedl::model::NodeOp::tag()`), so host-crosscheck filenames line up.
pub fn op_name(op_type: u8) -> &'static str {
    match op_type {
        OP_CONV2D => "conv2d",
        OP_PAD => "pad",
        OP_REDUCE_MEAN => "reduce_mean",
        OP_LINEAR => "linear",
        OP_RELU => "relu",
        _ => "unknown",
    }
}

/// Fixed fields between the variable-length `run_tag` and `payload_len`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FrameHeader {
    pub step: u16,
    pub value_id: u16,
    pub op_type: u8,
    pub exp: i8,
    pub shape_n: u16,
    pub shape_h: u16,
    pub shape_w: u16,
    pub shape_c: u16,
}

impl FrameHeader {
    pub fn to_le_bytes(&self) -> [u8; HEADER_LEN] {
        let mut b = [0u8; HEADER_LEN];
        b[0..2].copy_from_slice(&self.step.to_le_bytes());
        b[2..4].copy_from_slice(&self.value_id.to_le_bytes());
        b[4] = self.op_type;
        b[5] = self.exp as u8;
        b[6..8].copy_from_slice(&self.shape_n.to_le_bytes());
        b[8..10].copy_from_slice(&self.shape_h.to_le_bytes());
        b[10..12].copy_from_slice(&self.shape_w.to_le_bytes());
        b[12..14].copy_from_slice(&self.shape_c.to_le_bytes());
        b
    }

    pub fn from_le_bytes(b: &[u8; HEADER_LEN]) -> Self {
        Self {
            step: u16::from_le_bytes([b[0], b[1]]),
            value_id: u16::from_le_bytes([b[2], b[3]]),
            op_type: b[4],
            exp: b[5] as i8,
            shape_n: u16::from_le_bytes([b[6], b[7]]),
            shape_h: u16::from_le_bytes([b[8], b[9]]),
            shape_w: u16::from_le_bytes([b[10], b[11]]),
            shape_c: u16::from_le_bytes([b[12], b[13]]),
        }
    }
}

/// Streaming Fletcher-16 over the entire frame, magic through payload.
#[derive(Clone, Copy, Debug, Default)]
pub struct Fletcher16 {
    pub s1: u16,
    pub s2: u16,
}

impl Fletcher16 {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn update(&mut self, data: &[u8]) {
        for &b in data {
            self.s1 = (self.s1 + b as u16) % 255;
            self.s2 = (self.s2 + self.s1) % 255;
        }
    }

    pub fn finalize(self) -> u16 {
        (self.s2 << 8) | self.s1
    }
}

/// Convenience for computing a checksum over a single contiguous buffer.
pub fn fletcher16(data: &[u8]) -> u16 {
    let mut c = Fletcher16::new();
    c.update(data);
    c.finalize()
}

/// Build a frame on the device side. Callers provide a sink that will receive
/// the bytes; on-device this is `UpChannel::write`, in tests it's a `Vec`.
///
/// The sink is called multiple times (one call per section) so the bytes are
/// handed to RTT in the same granularity as before — no intermediate buffer.
pub fn write_frame(
    run_tag: &[u8],
    header: &FrameHeader,
    payload: &[u8],
    mut sink: impl FnMut(&[u8]),
) {
    let magic = MAGIC.to_le_bytes();
    let run_tag_len = (run_tag.len() as u16).to_le_bytes();
    let header_bytes = header.to_le_bytes();
    let payload_len = (payload.len() as u32).to_le_bytes();

    let mut cksum = Fletcher16::new();
    cksum.update(&magic);
    cksum.update(&run_tag_len);
    cksum.update(run_tag);
    cksum.update(&header_bytes);
    cksum.update(&payload_len);
    cksum.update(payload);
    let cksum_bytes = cksum.finalize().to_le_bytes();

    sink(&magic);
    sink(&run_tag_len);
    sink(run_tag);
    sink(&header_bytes);
    sink(&payload_len);
    sink(payload);
    sink(&cksum_bytes);
}

// ---------------------------------------------------------------------------
// Streaming host parser (alloc-gated)

#[cfg(feature = "alloc")]
mod parser {
    use super::{FrameHeader, Fletcher16, HEADER_LEN, MAGIC};
    use alloc::string::String;
    use alloc::vec::Vec;

    /// One successfully-reassembled frame handed to the caller's callback.
    #[derive(Debug)]
    pub struct DecodedFrame<'a> {
        pub run_tag: &'a str,
        pub header: FrameHeader,
        pub payload: &'a [u8],
        /// True when the trailing checksum matched the recomputed Fletcher-16.
        pub checksum_ok: bool,
        /// Checksum as transmitted by the device (little-endian on the wire).
        pub device_checksum: u16,
        /// Checksum recomputed on the host from the received bytes.
        pub host_checksum: u16,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct ParserStats {
        pub frames_ok: usize,
        pub frames_bad_checksum: usize,
        pub frames_bad_magic: usize,
        pub bytes_fed: usize,
    }

    #[derive(Clone, Copy, Debug)]
    enum State {
        Magic,
        RunTagLen,
        RunTag,
        Header,
        PayloadLen,
        Payload,
        Checksum,
    }

    /// Incremental frame decoder. Feed it byte slices as they arrive; each
    /// completed frame is handed to the caller via a `FnMut` closure. The
    /// closure receives both successful and bad-checksum frames — the latter
    /// surfaced via `DecodedFrame::checksum_ok = false` so the host can decide
    /// whether to log, drop, or still dump the payload.
    pub struct FrameParser {
        state: State,
        buf: Vec<u8>,
        need: usize,
        run_tag: String,
        header: FrameHeader,
        payload: Vec<u8>,
        cksum: Fletcher16,
        stats: ParserStats,
    }

    impl Default for FrameParser {
        fn default() -> Self {
            Self::new()
        }
    }

    impl FrameParser {
        pub fn new() -> Self {
            Self {
                state: State::Magic,
                buf: Vec::new(),
                need: 4,
                run_tag: String::new(),
                header: FrameHeader::default(),
                payload: Vec::new(),
                cksum: Fletcher16::new(),
                stats: ParserStats {
                    frames_ok: 0,
                    frames_bad_checksum: 0,
                    frames_bad_magic: 0,
                    bytes_fed: 0,
                },
            }
        }

        pub fn stats(&self) -> ParserStats {
            self.stats
        }

        /// Feed bytes and deliver each completed frame to `on_frame`.
        pub fn feed<F>(&mut self, mut data: &[u8], mut on_frame: F)
        where
            F: FnMut(DecodedFrame<'_>),
        {
            self.stats.bytes_fed += data.len();
            while !data.is_empty() {
                let take = (self.need - self.buf.len()).min(data.len());
                self.buf.extend_from_slice(&data[..take]);
                data = &data[take..];
                if self.buf.len() == self.need {
                    self.transition(&mut on_frame);
                }
            }
        }

        fn reset_to_magic(&mut self) {
            self.state = State::Magic;
            self.need = 4;
            self.buf.clear();
            self.run_tag.clear();
            self.payload.clear();
            self.cksum = Fletcher16::new();
        }

        fn transition<F>(&mut self, on_frame: &mut F)
        where
            F: FnMut(DecodedFrame<'_>),
        {
            match self.state {
                State::Magic => {
                    let magic = u32::from_le_bytes([
                        self.buf[0], self.buf[1], self.buf[2], self.buf[3],
                    ]);
                    if magic != MAGIC {
                        self.stats.frames_bad_magic += 1;
                        self.reset_to_magic();
                        return;
                    }
                    self.cksum = Fletcher16::new();
                    self.cksum.update(&self.buf);
                    self.buf.clear();
                    self.state = State::RunTagLen;
                    self.need = 2;
                }
                State::RunTagLen => {
                    self.cksum.update(&self.buf);
                    let n = u16::from_le_bytes([self.buf[0], self.buf[1]]) as usize;
                    self.buf.clear();
                    self.run_tag.clear();
                    if n == 0 {
                        self.state = State::Header;
                        self.need = HEADER_LEN;
                    } else {
                        self.state = State::RunTag;
                        self.need = n;
                    }
                }
                State::RunTag => {
                    self.cksum.update(&self.buf);
                    self.run_tag = String::from_utf8_lossy(&self.buf).into_owned();
                    self.buf.clear();
                    self.state = State::Header;
                    self.need = HEADER_LEN;
                }
                State::Header => {
                    self.cksum.update(&self.buf);
                    let mut hdr = [0u8; HEADER_LEN];
                    hdr.copy_from_slice(&self.buf);
                    self.header = FrameHeader::from_le_bytes(&hdr);
                    self.buf.clear();
                    self.state = State::PayloadLen;
                    self.need = 4;
                }
                State::PayloadLen => {
                    self.cksum.update(&self.buf);
                    let n = u32::from_le_bytes([
                        self.buf[0], self.buf[1], self.buf[2], self.buf[3],
                    ]) as usize;
                    self.buf.clear();
                    self.payload.clear();
                    self.payload.reserve(n);
                    if n == 0 {
                        self.state = State::Checksum;
                        self.need = 2;
                    } else {
                        self.state = State::Payload;
                        self.need = n;
                    }
                }
                State::Payload => {
                    self.cksum.update(&self.buf);
                    self.payload.extend_from_slice(&self.buf);
                    self.buf.clear();
                    self.state = State::Checksum;
                    self.need = 2;
                }
                State::Checksum => {
                    let device = u16::from_le_bytes([self.buf[0], self.buf[1]]);
                    let host = self.cksum.finalize();
                    self.buf.clear();
                    let ok = device == host;
                    if ok {
                        self.stats.frames_ok += 1;
                    } else {
                        self.stats.frames_bad_checksum += 1;
                    }
                    on_frame(DecodedFrame {
                        run_tag: &self.run_tag,
                        header: self.header,
                        payload: &self.payload,
                        checksum_ok: ok,
                        device_checksum: device,
                        host_checksum: host,
                    });
                    self.state = State::Magic;
                    self.need = 4;
                    self.cksum = Fletcher16::new();
                }
            }
        }
    }
}

#[cfg(feature = "alloc")]
pub use parser::{DecodedFrame, FrameParser, ParserStats};

// ---------------------------------------------------------------------------
// Tests

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_header() -> FrameHeader {
        FrameHeader {
            step: 3,
            value_id: 7,
            op_type: OP_CONV2D,
            exp: -4,
            shape_n: 1,
            shape_h: 8,
            shape_w: 4,
            shape_c: 16,
        }
    }

    #[test]
    fn header_roundtrip_preserves_all_fields() {
        let h = sample_header();
        let bytes = h.to_le_bytes();
        assert_eq!(FrameHeader::from_le_bytes(&bytes), h);
    }

    #[test]
    fn header_byte_layout_is_stable() {
        // Pinned byte-for-byte so an accidental field reorder fails the test
        // instead of silently shifting the wire format.
        let h = FrameHeader {
            step: 0x0201,
            value_id: 0x0403,
            op_type: 0x05,
            exp: -2, // 0xFE
            shape_n: 0x0807,
            shape_h: 0x0A09,
            shape_w: 0x0C0B,
            shape_c: 0x0E0D,
        };
        assert_eq!(
            h.to_le_bytes(),
            [0x01, 0x02, 0x03, 0x04, 0x05, 0xFE, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E]
        );
    }

    #[test]
    fn negative_exp_survives_roundtrip() {
        let mut h = sample_header();
        h.exp = i8::MIN;
        let back = FrameHeader::from_le_bytes(&h.to_le_bytes());
        assert_eq!(back.exp, i8::MIN);
    }

    #[test]
    fn header_len_const_matches_encoding() {
        assert_eq!(HEADER_LEN, sample_header().to_le_bytes().len());
    }

    #[test]
    fn fletcher16_matches_rfc_reference_vector() {
        // Reference: Fletcher-16 of "abcde" per Wikipedia's fletcher-16 example
        // is 0xC8F0 (s2=0xC8, s1=0xF0).
        assert_eq!(fletcher16(b"abcde"), 0xC8F0);
    }

    #[test]
    fn fletcher16_incremental_matches_oneshot() {
        let data: Vec<u8> = (0u8..=200).collect();
        let oneshot = fletcher16(&data);
        let mut c = Fletcher16::new();
        c.update(&data[..17]);
        c.update(&data[17..73]);
        c.update(&data[73..]);
        assert_eq!(c.finalize(), oneshot);
    }

    #[test]
    fn op_name_covers_all_ops() {
        assert_eq!(op_name(OP_CONV2D), "conv2d");
        assert_eq!(op_name(OP_PAD), "pad");
        assert_eq!(op_name(OP_REDUCE_MEAN), "reduce_mean");
        assert_eq!(op_name(OP_LINEAR), "linear");
        assert_eq!(op_name(OP_RELU), "relu");
        assert_eq!(op_name(255), "unknown");
    }

    #[test]
    fn write_frame_emits_expected_byte_stream() {
        let run_tag = b"simd_off";
        let header = sample_header();
        let payload: Vec<u8> = (0u8..37).collect();

        let mut buf = Vec::new();
        write_frame(run_tag, &header, &payload, |chunk| buf.extend_from_slice(chunk));

        // Magic
        assert_eq!(&buf[0..4], &MAGIC.to_le_bytes());
        // run_tag_len
        assert_eq!(&buf[4..6], &(run_tag.len() as u16).to_le_bytes());
        // run_tag
        assert_eq!(&buf[6..6 + run_tag.len()], run_tag);
        // header
        let off = 6 + run_tag.len();
        assert_eq!(&buf[off..off + HEADER_LEN], &header.to_le_bytes());
        // payload_len
        let off = off + HEADER_LEN;
        assert_eq!(&buf[off..off + 4], &(payload.len() as u32).to_le_bytes());
        // payload
        let off = off + 4;
        assert_eq!(&buf[off..off + payload.len()], &payload[..]);
        // checksum matches what fletcher16 over everything-before gives
        let off = off + payload.len();
        let trailer = u16::from_le_bytes([buf[off], buf[off + 1]]);
        assert_eq!(trailer, fletcher16(&buf[..off]));
        assert_eq!(buf.len(), off + 2);
    }

    #[cfg(feature = "alloc")]
    mod parser_tests {
        use super::*;

        fn build_frame(run_tag: &[u8], header: &FrameHeader, payload: &[u8]) -> Vec<u8> {
            let mut buf = Vec::new();
            write_frame(run_tag, header, payload, |c| buf.extend_from_slice(c));
            buf
        }

        #[test]
        fn roundtrip_single_frame() {
            let run_tag = b"simd_off";
            let header = sample_header();
            let payload: Vec<u8> = (0u8..37).collect();
            let buf = build_frame(run_tag, &header, &payload);

            let mut frames = Vec::new();
            let mut parser = FrameParser::new();
            parser.feed(&buf, |f| {
                assert!(f.checksum_ok);
                frames.push((
                    f.run_tag.to_string(),
                    f.header,
                    f.payload.to_vec(),
                ));
            });
            assert_eq!(frames.len(), 1);
            let (tag, h, p) = &frames[0];
            assert_eq!(tag, "simd_off");
            assert_eq!(*h, header);
            assert_eq!(*p, payload);
            assert_eq!(parser.stats().frames_ok, 1);
            assert_eq!(parser.stats().frames_bad_checksum, 0);
        }

        #[test]
        fn feed_one_byte_at_a_time() {
            let header = sample_header();
            let payload: Vec<u8> = (0u8..9).collect();
            let buf = build_frame(b"t", &header, &payload);

            let mut count = 0usize;
            let mut parser = FrameParser::new();
            for b in &buf {
                parser.feed(core::slice::from_ref(b), |f| {
                    assert!(f.checksum_ok);
                    assert_eq!(f.header, header);
                    assert_eq!(f.payload, payload.as_slice());
                    count += 1;
                });
            }
            assert_eq!(count, 1);
        }

        #[test]
        fn two_frames_back_to_back() {
            let h1 = sample_header();
            let mut h2 = sample_header();
            h2.step = 99;
            h2.op_type = OP_LINEAR;
            let p1: Vec<u8> = (0u8..20).collect();
            let p2: Vec<u8> = (50u8..60).collect();
            let mut buf = build_frame(b"a", &h1, &p1);
            buf.extend(build_frame(b"bb", &h2, &p2));

            let mut seen = Vec::new();
            let mut parser = FrameParser::new();
            parser.feed(&buf, |f| seen.push((f.run_tag.to_string(), f.header, f.payload.to_vec())));
            assert_eq!(seen.len(), 2);
            assert_eq!(seen[0].0, "a");
            assert_eq!(seen[0].1, h1);
            assert_eq!(seen[0].2, p1);
            assert_eq!(seen[1].0, "bb");
            assert_eq!(seen[1].1, h2);
            assert_eq!(seen[1].2, p2);
        }

        #[test]
        fn empty_run_tag_and_empty_payload() {
            let header = FrameHeader {
                step: 0,
                value_id: 0,
                op_type: OP_RELU,
                exp: 0,
                shape_n: 1,
                shape_h: 1,
                shape_w: 1,
                shape_c: 1,
            };
            let buf = build_frame(b"", &header, &[]);
            let mut count = 0usize;
            let mut parser = FrameParser::new();
            parser.feed(&buf, |f| {
                assert!(f.checksum_ok);
                assert_eq!(f.run_tag, "");
                assert_eq!(f.payload.len(), 0);
                assert_eq!(f.header, header);
                count += 1;
            });
            assert_eq!(count, 1);
        }

        #[test]
        fn bad_checksum_delivers_frame_with_flag() {
            let header = sample_header();
            let payload: Vec<u8> = vec![1, 2, 3, 4];
            let mut buf = build_frame(b"tag", &header, &payload);
            // Corrupt the trailing checksum.
            let last = buf.len() - 1;
            buf[last] ^= 0xFF;

            let mut delivered = 0usize;
            let mut parser = FrameParser::new();
            parser.feed(&buf, |f| {
                assert!(!f.checksum_ok);
                assert_ne!(f.device_checksum, f.host_checksum);
                delivered += 1;
            });
            assert_eq!(delivered, 1);
            assert_eq!(parser.stats().frames_ok, 0);
            assert_eq!(parser.stats().frames_bad_checksum, 1);
        }

        #[test]
        fn bad_magic_resyncs_and_recovers() {
            let header = sample_header();
            let payload: Vec<u8> = (0u8..8).collect();
            let good = build_frame(b"t", &header, &payload);

            // Prefix 4 bytes of garbage (bad magic), then a valid frame.
            let mut buf = vec![0xAA, 0xBB, 0xCC, 0xDD];
            buf.extend_from_slice(&good);

            let mut delivered = 0usize;
            let mut parser = FrameParser::new();
            parser.feed(&buf, |f| {
                assert!(f.checksum_ok);
                delivered += 1;
            });
            assert_eq!(delivered, 1);
            assert_eq!(parser.stats().frames_bad_magic, 1);
            assert_eq!(parser.stats().frames_ok, 1);
        }

        #[test]
        fn stats_track_bytes_fed() {
            let buf = build_frame(b"x", &sample_header(), &[0u8; 4]);
            let mut parser = FrameParser::new();
            parser.feed(&buf[..3], |_| {});
            parser.feed(&buf[3..], |_| {});
            assert_eq!(parser.stats().bytes_fed, buf.len());
        }
    }
}
