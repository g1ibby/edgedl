use core::fmt;
#[cfg(all(feature = "stack-probe", target_arch = "xtensa"))]
use core::sync::atomic::AtomicUsize;

#[cfg(all(feature = "stack-probe", target_arch = "xtensa"))]
static MIN_SP: AtomicUsize = AtomicUsize::new(usize::MAX);

#[derive(Clone, Copy, Debug)]
pub struct MemoryReport {
    pub data_bytes: Option<usize>,
    pub bss_bytes: Option<usize>,
    pub stack_total_bytes: Option<usize>,
    pub stack: Option<StackUsage>,
}

#[derive(Clone, Copy, Debug)]
pub struct StackUsage {
    pub used_current_bytes: usize,
    pub used_max_bytes: usize,
    pub free_current_bytes: usize,
    pub free_min_bytes: usize,
}

impl MemoryReport {
    #[must_use]
    pub fn static_ram_bytes(self) -> Option<usize> {
        Some(self.data_bytes? + self.bss_bytes?)
    }
}

#[must_use]
pub fn report() -> MemoryReport {
    let (data_bytes, bss_bytes) = data_bss_sizes();
    let stack_total_bytes = stack_total_bytes();
    let stack = stack_usage();

    MemoryReport {
        data_bytes,
        bss_bytes,
        stack_total_bytes,
        stack,
    }
}

#[inline(always)]
pub fn probe_stack() {
    #[cfg(all(feature = "stack-probe", target_arch = "xtensa"))]
    stack_probe_xtensa();
}

#[cfg(all(feature = "stack-probe", target_arch = "xtensa"))]
fn stack_probe_xtensa() {
    use core::sync::atomic::Ordering;

    let sp = current_stack_pointer_xtensa();
    let mut min = MIN_SP.load(Ordering::Relaxed);
    while sp < min {
        match MIN_SP.compare_exchange_weak(min, sp, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break,
            Err(v) => min = v,
        }
    }
}

#[cfg(all(feature = "stack-probe", target_arch = "xtensa"))]
#[inline(always)]
fn current_stack_pointer_xtensa() -> usize {
    let sp: usize;
    unsafe {
        core::arch::asm!("mov {0}, sp", out(reg) sp, options(nostack));
    }
    sp
}

#[must_use]
fn data_bss_sizes() -> (Option<usize>, Option<usize>) {
    #[cfg(target_arch = "xtensa")]
    {
        unsafe extern "C" {
            static _bss_start: u32;
            static _bss_end: u32;
            static _data_start: u32;
            static _data_end: u32;
        }

        let data_start = core::ptr::addr_of!(_data_start) as usize;
        let data_end = core::ptr::addr_of!(_data_end) as usize;
        let bss_start = core::ptr::addr_of!(_bss_start) as usize;
        let bss_end = core::ptr::addr_of!(_bss_end) as usize;

        return (
            Some(data_end.saturating_sub(data_start)),
            Some(bss_end.saturating_sub(bss_start)),
        );
    }

    #[cfg(not(target_arch = "xtensa"))]
    (None, None)
}

#[must_use]
fn stack_total_bytes() -> Option<usize> {
    #[cfg(target_arch = "xtensa")]
    {
        unsafe extern "C" {
            static _stack_start_cpu0: u32;
            static _stack_end_cpu0: u32;
        }

        let stack_start = core::ptr::addr_of!(_stack_start_cpu0) as usize;
        let stack_end = core::ptr::addr_of!(_stack_end_cpu0) as usize;
        return Some(stack_start.saturating_sub(stack_end));
    }

    #[cfg(not(target_arch = "xtensa"))]
    None
}

#[must_use]
fn stack_usage() -> Option<StackUsage> {
    #[cfg(all(feature = "stack-probe", target_arch = "xtensa"))]
    {
        use core::sync::atomic::Ordering;

        unsafe extern "C" {
            static _stack_start_cpu0: u32;
            static _stack_end_cpu0: u32;
        }

        let stack_start = core::ptr::addr_of!(_stack_start_cpu0) as usize;
        let stack_end = core::ptr::addr_of!(_stack_end_cpu0) as usize;

        let sp = current_stack_pointer_xtensa();
        let min_sp = MIN_SP.load(Ordering::Relaxed);

        let used_current_bytes = stack_start.saturating_sub(sp);
        let free_current_bytes = sp.saturating_sub(stack_end);

        let (used_max_bytes, free_min_bytes) = if min_sp == usize::MAX {
            (used_current_bytes, free_current_bytes)
        } else {
            (
                stack_start.saturating_sub(min_sp),
                min_sp.saturating_sub(stack_end),
            )
        };

        return Some(StackUsage {
            used_current_bytes,
            used_max_bytes,
            free_current_bytes,
            free_min_bytes,
        });
    }

    #[cfg(not(all(feature = "stack-probe", target_arch = "xtensa")))]
    None
}

impl fmt::Display for MemoryReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let (Some(data), Some(bss)) = (self.data_bytes, self.bss_bytes) {
            write!(f, "data={}B bss={}B static={}B", data, bss, data + bss)?;
        } else {
            write!(f, "data/bss=unknown")?;
        }

        if let Some(total) = self.stack_total_bytes {
            write!(f, " stack_total={}B", total)?;
        }

        if let Some(stack) = self.stack {
            write!(
                f,
                " stack_used_cur={}B stack_used_max={}B stack_free_cur={}B stack_free_min={}B",
                stack.used_current_bytes,
                stack.used_max_bytes,
                stack.free_current_bytes,
                stack.free_min_bytes
            )?;
        } else {
            write!(f, " stack_probe=off")?;
        }

        Ok(())
    }
}
