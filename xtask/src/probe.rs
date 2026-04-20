//! In-tree HIL runner for edgedl.
//!
//! Flashes a HIL test ELF, drives the embedded-test protocol over semihosting
//! (v1: `run_addr <fn_ptr>` picked from the ELF's `.embedded_test` section),
//! and in parallel drains RTT up-channel 1 as raw binary bytes — the channel
//! `probe-rs run` would otherwise UTF-8-mangle and we want byte-exact so we
//! can diff tensor dumps against host-computed goldens.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use defmt_decoder::{DecodeError, Locations, StreamDecoder, Table};
use object::{Object, ObjectSection, ObjectSymbol};
use probe_rs::architecture::xtensa::communication_interface::XtensaError;
use probe_rs::rtt::{Rtt, ScanRegion};
use probe_rs::semihosting::SemihostingCommand;
use probe_rs::{
    BreakpointCause, Core, CoreStatus, HaltReason, MemoryInterface, Permissions, Session,
    SessionConfig,
};
use crosscheck_proto::{DecodedFrame, FrameParser, op_name};

/// How long we let `core.wait_for_core_halted` block per iteration before we
/// fall through to drain RTT. Must be short enough that the host stays
/// responsive, but long enough that we aren't spamming the USB-JTAG bridge —
/// polling `core.status()` at 100 Hz overran the macOS IOKit bulk pipe within
/// ~19 s in earlier iterations.
const HALT_POLL: Duration = Duration::from_millis(200);

const TARGET_NAME: &str = "esp32s3";
const TEST_TIMEOUT: Duration = Duration::from_secs(120);
const TENSOR_CHANNEL: usize = 1;

/// Flash an ELF, drive its embedded-test, drain RTT ch0 (defmt) + ch1 (tensor
/// frames). If `dump_dir` is set, channel-1 frames land there as
/// `<run_tag>_n<step>_<op>_v<value_id>.bin`.
pub fn run(elf_path: &Path, test_filter: Option<&str>, dump_dir: Option<&Path>) -> Result<()> {
    let elf_bytes =
        std::fs::read(elf_path).with_context(|| format!("reading {}", elf_path.display()))?;

    let (test_name, test_fn_addr) = find_test(&elf_bytes, test_filter)?;
    log::info!("probe: selected test '{test_name}' at {test_fn_addr:#x}");

    // Resolve the RTT control block address up front. Without this we'd have to
    // pass `ScanRegion::Ram` to `Rtt::attach_region`, which scans *all* of SRAM
    // every time we try to attach — hundreds of KB over USB per attempt. With
    // an exact address it's a 24-byte read.
    let rtt_addr = find_symbol_addr(&elf_bytes, "_SEGGER_RTT")
        .context("ELF has no `_SEGGER_RTT` symbol — is `rtt-target`'s linker section present?")?;
    log::info!("probe: _SEGGER_RTT at {rtt_addr:#x}");

    let defmt_table = Table::parse(&elf_bytes).context("parsing defmt table from ELF")?;
    let defmt_locs = match &defmt_table {
        Some(t) => t.get_locations(&elf_bytes).ok(),
        None => None,
    };
    let mut defmt_stream = DefmtStream::new(defmt_table.as_ref(), defmt_locs.as_ref());

    if let Some(d) = dump_dir {
        std::fs::create_dir_all(d).with_context(|| format!("creating {}", d.display()))?;
        log::info!("probe: writing channel-1 frames to {}", d.display());
    }

    // Flash via espflash (CDC serial path). On macOS the probe-rs library's IDF flash
    // triggers a USB endpoint-disappearance on the JTAG side we don't recover from;
    // espflash uses the ESP32-S3 ROM bootloader over USB-CDC, a separate USB interface.
    log::info!("probe: flashing via `espflash` subprocess");
    let flash_status = std::process::Command::new("espflash")
        .args(["flash", "--non-interactive", "--chip", TARGET_NAME])
        .arg(elf_path)
        .status()
        .context("failed to invoke `espflash` (is it installed on PATH?)")?;
    if !flash_status.success() {
        bail!("espflash exited with status {flash_status}");
    }

    std::thread::sleep(Duration::from_millis(500));

    log::info!("probe: attaching probe, target '{TARGET_NAME}'");
    let mut session = attach_with_retry()?;

    let mut core = session.core(0)?;
    core.reset_and_halt(Duration::from_millis(500))
        .context("post-flash reset_and_halt")?;
    log::debug!("probe: core reset and halted");

    let cmdline = format!("run_addr {test_fn_addr}");
    let scan_region = ScanRegion::Exact(rtt_addr);

    core.run().context("core.run() after reset")?;

    let mut rtt: Option<Rtt> = None;
    let mut handshake_done = false;
    let mut sink = FrameSink::new(dump_dir.map(|p| p.to_path_buf()));
    let mut parser = FrameParser::new();
    let mut stale_cmdline_retries: u32 = 0;
    const MAX_STALE_RETRIES: u32 = 6;
    let start = Instant::now();

    let outcome: Result<()> = loop {
        if start.elapsed() > TEST_TIMEOUT {
            break Err(anyhow!("probe: test timeout ({:?})", TEST_TIMEOUT));
        }

        match core.wait_for_core_halted(HALT_POLL) {
            Ok(()) => {}
            Err(probe_rs::Error::Timeout) | Err(probe_rs::Error::Xtensa(XtensaError::Timeout)) => {
                if rtt.is_none() {
                    match Rtt::attach_region(&mut core, &scan_region) {
                        Ok(mut r) => {
                            log::info!(
                                "probe: RTT attached ({} up channel(s))",
                                r.up_channels().len()
                            );
                            rtt = Some(r);
                        }
                        Err(probe_rs::rtt::Error::ControlBlockNotFound) => {}
                        Err(e) => log::debug!("probe: RTT attach error {e:?}"),
                    }
                }
                if let Some(r) = rtt.as_mut() {
                    drain_channel_0(r, &mut core, &mut defmt_stream)?;
                    drain_channel_1(r, &mut core, &mut parser, &mut sink)?;
                }
                continue;
            }
            Err(e) => return Err(e).context("wait_for_core_halted"),
        }

        if rtt.is_none() {
            match Rtt::attach_region(&mut core, &scan_region) {
                Ok(mut r) => {
                    log::info!(
                        "probe: RTT attached at halt ({} up channel(s))",
                        r.up_channels().len()
                    );
                    rtt = Some(r);
                }
                Err(probe_rs::rtt::Error::ControlBlockNotFound) => {}
                Err(e) => log::debug!("probe: RTT attach error {e:?}"),
            }
        }
        if let Some(r) = rtt.as_mut() {
            drain_channel_0(r, &mut core, &mut defmt_stream)?;
            drain_channel_1(r, &mut core, &mut parser, &mut sink)?;
        }

        match core.status()? {
            CoreStatus::Halted(HaltReason::Breakpoint(BreakpointCause::Semihosting(cmd))) => {
                match cmd {
                    SemihostingCommand::GetCommandLine(_req) if !handshake_done => {
                        let pc: u32 = core
                            .read_core_reg(core.program_counter().id())
                            .unwrap_or(0);
                        log::info!("probe: sending cmdline '{cmdline}' (PC={pc:#010x})");
                        match send_cmdline(&mut core, &cmdline) {
                            Ok(()) => {
                                handshake_done = true;
                                core.run()?;
                            }
                            Err(e) if e.is::<StaleCmdlineBlock>() => {
                                stale_cmdline_retries += 1;
                                if stale_cmdline_retries > MAX_STALE_RETRIES {
                                    break Err(e.context(format!(
                                        "probe: exhausted {MAX_STALE_RETRIES} reset retries \
                                         while trying to hand off cmdline"
                                    )));
                                }
                                log::warn!(
                                    "probe: stale cmdline block (attempt {stale_cmdline_retries}/\
                                     {MAX_STALE_RETRIES}); reset_and_halt + rerun",
                                );
                                // Force the firmware back through a full boot. Dropping any
                                // accumulated RTT state is fine — the test hasn't started yet,
                                // no tensor frames have been drained.
                                core.reset_and_halt(Duration::from_millis(500))
                                    .context("retry reset_and_halt after stale cmdline")?;
                                rtt = None;
                                core.run().context("retry core.run() after reset")?;
                                continue;
                            }
                            Err(e) => break Err(e),
                        }
                    }
                    SemihostingCommand::ExitSuccess => {
                        log::info!("probe: test reported success via semihosting");
                        break Ok(());
                    }
                    SemihostingCommand::ExitError(details) => {
                        break Err(anyhow!("probe: test failed via semihosting: {details}"));
                    }
                    SemihostingCommand::WriteConsole(req) => {
                        let s = req.read(&mut core)?;
                        eprint!("{s}");
                        core.run()?;
                    }
                    SemihostingCommand::Errno(_) | SemihostingCommand::Unknown(_) => {
                        core.run()?;
                    }
                    other => {
                        log::warn!("probe: unhandled semihosting cmd {other:?}");
                        core.run()?;
                    }
                }
            }
            CoreStatus::Halted(reason) => {
                break Err(anyhow!("probe: unexpected halt: {reason:?}"));
            }
            CoreStatus::LockedUp => {
                break Err(anyhow!("probe: core locked up"));
            }
            CoreStatus::Running | CoreStatus::Sleeping | CoreStatus::Unknown => {}
        }
    };

    if let Some(r) = rtt.as_mut() {
        for _ in 0..5 {
            drain_channel_0(r, &mut core, &mut defmt_stream)?;
            drain_channel_1(r, &mut core, &mut parser, &mut sink)?;
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    sink.print_summary(&parser);

    outcome
}

/// Signal that the target is halted at SYS_GET_CMDLINE but the `a3` block
/// read returned a clearly-bogus length — recoverable by reset+retry.
#[derive(Debug)]
struct StaleCmdlineBlock;

impl std::fmt::Display for StaleCmdlineBlock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("SYS_GET_CMDLINE block read was stale / pre-commit")
    }
}

impl std::error::Error for StaleCmdlineBlock {}

/// Respond to embedded-test's `SYS_GET_CMDLINE` (semihosting op 0x15) with our
/// `run_addr <fn_ptr>` command line.
///
/// Bypasses `GetCommandLineRequest::write_command_line_to_target` so we can
/// re-read the `(buf_ptr, buf_len)` block from `a3` ourselves. Background: on
/// ESP32-S3 we see a repeatable-but-intermittent fault where, although the PC
/// is genuinely at the `break 1, 14` inside
/// `semihosting::sys::arm_compat::sys_get_cmdline_uninit`, the live `a3` read
/// over USB-JTAG returns an address pointing into `.bss` (in practice: inside
/// `_SEGGER_RTT`) with `*a3 = {0, 0}` instead of the stack-allocated
/// `[buf_ptr, 1024]` the firmware set up. probe-rs's cached `Buffer::write`
/// then bails with "buffer not large enough".
///
/// The observed pattern is strict alternation of success/failure across
/// back-to-back runs, so we surface the bogus-block case as a distinct error
/// the caller can recover from by issuing a fresh `reset_and_halt + run` cycle.
fn send_cmdline(core: &mut Core<'_>, cmdline: &str) -> Result<()> {
    let mut payload = cmdline.to_owned().into_bytes();
    payload.push(0); // NUL terminator — embedded-test parses a C string.

    let regs = core.registers();
    let arg0_id = regs
        .get_argument_register(0)
        .context("probe: xtensa missing argument register 0 (return value)")?
        .id();
    let arg1_id = regs
        .get_argument_register(1)
        .context("probe: xtensa missing argument register 1 (cmdline block address)")?
        .id();

    let block_addr: u32 = core.read_core_reg(arg1_id)?;

    let mut block = [0u32; 2];
    core.read_32(block_addr as u64, &mut block)?;
    let buf_addr = block[0];
    let buf_len = block[1];
    log::debug!(
        "probe: cmdline block: addr={block_addr:#010x} buf={buf_addr:#010x} len={buf_len}"
    );

    if (buf_len as usize) < payload.len() {
        log::warn!(
            "probe: SYS_GET_CMDLINE block at {block_addr:#010x} advertises len={buf_len} \
             < {}; treating as stale read and requesting reset+retry",
            payload.len(),
        );
        return Err(StaleCmdlineBlock.into());
    }

    core.write_8(buf_addr as u64, &payload)?;
    // Update the block's length to the bytes written (excluding NUL), matching
    // what probe-rs's own `Buffer::write` does.
    let updated = [buf_addr, (payload.len() - 1) as u32];
    core.write_32(block_addr as u64, &updated)?;
    core.write_core_reg(arg0_id, 0u32)?;

    Ok(())
}

fn attach_with_retry() -> Result<Session> {
    let mut last_err: Option<probe_rs::Error> = None;
    for attempt in 0..5 {
        match Session::auto_attach(
            TARGET_NAME,
            SessionConfig {
                permissions: Permissions::default(),
                ..Default::default()
            },
        ) {
            Ok(s) => return Ok(s),
            Err(e) => {
                log::warn!("probe: attach attempt {} failed: {e}", attempt + 1);
                last_err = Some(e);
                std::thread::sleep(Duration::from_millis(500));
            }
        }
    }
    Err(last_err
        .map(anyhow::Error::from)
        .unwrap_or_else(|| anyhow!("probe: attach failed without a reported error")))
}

fn find_test(elf_bytes: &[u8], filter: Option<&str>) -> Result<(String, u64)> {
    let elf = object::File::parse(elf_bytes).context("parsing ELF")?;

    let et_section = elf.section_by_name(".embedded_test").context(
        "ELF has no `.embedded_test` section — is the `embedded-test` linker file wired in?",
    )?;

    let mut all: Vec<(String, u64)> = Vec::new();

    for sym in elf.symbols() {
        if !sym.is_global() || sym.size() != 12 || sym.section_index() != Some(et_section.index()) {
            continue;
        }
        let Ok(raw_name) = sym.name() else { continue };
        let Some(data) = et_section.data_range(sym.address(), 12).ok().flatten() else {
            continue;
        };
        let fn_ptr = u32::from_le_bytes(data[0..4].try_into().unwrap()) as u64;
        all.push((raw_name.to_string(), fn_ptr));
    }

    if all.is_empty() {
        bail!("no embedded-test test symbols found in ELF");
    }

    log::debug!("probe: {} test symbol(s) in ELF", all.len());
    for (name, addr) in &all {
        log::debug!("  {addr:#x}  {name}");
    }

    let picked = if let Some(f) = filter {
        all.iter()
            .find(|(name, _)| name.contains(f))
            .with_context(|| format!("no test matching filter {f:?}"))?
    } else {
        &all[0]
    };

    Ok((picked.0.clone(), picked.1))
}

fn find_symbol_addr(elf_bytes: &[u8], name: &str) -> Option<u64> {
    let elf = object::File::parse(elf_bytes).ok()?;
    elf.symbols()
        .find(|s| s.name().ok() == Some(name))
        .map(|s| s.address())
}

fn drain_channel_0(
    rtt: &mut Rtt,
    core: &mut probe_rs::Core<'_>,
    stream: &mut DefmtStream<'_>,
) -> Result<()> {
    let Some(ch) = rtt.up_channel(0) else {
        return Ok(());
    };
    let mut buf = [0u8; 2048];
    let n = ch.read(core, &mut buf)?;
    if n > 0 {
        stream.feed(&buf[..n]);
    }
    Ok(())
}

struct DefmtStream<'t> {
    decoder: Option<Box<dyn StreamDecoder + Send + Sync + 't>>,
    locs: Option<&'t Locations>,
}

impl<'t> DefmtStream<'t> {
    fn new(table: Option<&'t Table>, locs: Option<&'t Locations>) -> Self {
        let decoder = table.map(|t| t.new_stream_decoder());
        Self { decoder, locs }
    }

    fn feed(&mut self, data: &[u8]) {
        let Some(decoder) = self.decoder.as_mut() else {
            return;
        };
        decoder.received(data);
        loop {
            match decoder.decode() {
                Ok(frame) => {
                    let loc = self
                        .locs
                        .and_then(|l| l.get(&frame.index()))
                        .map(|l| format!(" ({}:{})", l.file.display(), l.line))
                        .unwrap_or_default();
                    eprintln!("[{}]{loc} {}", level_tag(&frame), frame.display_message());
                }
                Err(DecodeError::UnexpectedEof) => break,
                Err(DecodeError::Malformed) => {
                    eprintln!("[defmt] malformed frame, resyncing");
                    break;
                }
            }
        }
    }
}

fn level_tag(frame: &defmt_decoder::Frame<'_>) -> String {
    match frame.level() {
        Some(l) => format!("{l:?}").to_uppercase(),
        None => "-".to_string(),
    }
}

fn drain_channel_1(
    rtt: &mut Rtt,
    core: &mut probe_rs::Core<'_>,
    parser: &mut FrameParser,
    sink: &mut FrameSink,
) -> Result<()> {
    let Some(ch) = rtt.up_channel(TENSOR_CHANNEL) else {
        return Ok(());
    };
    let mut buf = [0u8; 8192];
    let n = ch.read(core, &mut buf)?;
    if n > 0 {
        parser.feed(&buf[..n], |frame| sink.handle(frame));
    }
    Ok(())
}

struct FrameSink {
    dump_dir: Option<PathBuf>,
}

impl FrameSink {
    fn new(dump_dir: Option<PathBuf>) -> Self {
        Self { dump_dir }
    }

    fn handle(&mut self, frame: DecodedFrame<'_>) {
        let op = op_name(frame.header.op_type);
        let file_name = format!(
            "{}_n{}_{}_v{}",
            frame.run_tag, frame.header.step, op, frame.header.value_id
        );

        if !frame.checksum_ok {
            log::warn!(
                "probe: frame '{file_name}' checksum mismatch (device={:#06x}, host={:#06x})",
                frame.device_checksum,
                frame.host_checksum
            );
            return;
        }

        log::info!(
            "probe: frame '{file_name}' ok step={} vid={} op={} exp={} shape=[{},{},{},{}] bytes={} cksum={:#06x}",
            frame.header.step,
            frame.header.value_id,
            op,
            frame.header.exp,
            frame.header.shape_n,
            frame.header.shape_h,
            frame.header.shape_w,
            frame.header.shape_c,
            frame.payload.len(),
            frame.device_checksum,
        );
        if let Some(dir) = self.dump_dir.as_ref() {
            let path = dir.join(format!("{file_name}.bin"));
            if let Err(e) = std::fs::write(&path, frame.payload) {
                log::warn!("probe: failed to write {}: {e}", path.display());
            }
        }
    }

    fn print_summary(&self, parser: &FrameParser) {
        let s = parser.stats();
        log::info!(
            "probe: channel 1 summary — bytes={} frames_ok={} bad_magic={} bad_checksum={}",
            s.bytes_fed,
            s.frames_ok,
            s.frames_bad_magic,
            s.frames_bad_checksum,
        );
    }
}
