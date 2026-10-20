//! Optional bounded semantic capture. Applications own capture lifetime and export.
use crate::{Layout, Storage, Tensor};
use std::{collections::BTreeMap, marker::PhantomData, rc::Rc};

#[derive(Clone, Debug)]
#[cfg_attr(feature = "d7-trace", derive(serde::Serialize))]
#[cfg_attr(feature = "d7-trace", serde(untagged))]
pub enum Value {
    Text(String),
    Number(u64),
    Bool(bool),
    Numbers(Vec<usize>),
    Object(Payload),
}

#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "d7-trace", derive(serde::Serialize))]
#[cfg_attr(feature = "d7-trace", serde(transparent))]
pub struct Payload(BTreeMap<String, Value>);
impl Payload {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn text(mut self, key: &str, value: impl ToString) -> Self {
        self.0.insert(key.into(), Value::Text(value.to_string()));
        self
    }
    pub fn number(mut self, key: &str, value: u64) -> Self {
        self.0.insert(key.into(), Value::Number(value));
        self
    }
    pub fn boolean(mut self, key: &str, value: bool) -> Self {
        self.0.insert(key.into(), Value::Bool(value));
        self
    }
    pub fn numbers(mut self, key: &str, value: &[usize]) -> Self {
        self.0.insert(key.into(), Value::Numbers(value.to_vec()));
        self
    }
    pub fn object(mut self, key: &str, value: Payload) -> Self {
        self.0.insert(key.into(), Value::Object(value));
        self
    }
}

pub fn tensor(t: &Tensor) -> Payload {
    Payload::new()
        .text("tensor_id", format!("{:?}", t.id()))
        .text("storage_dtype", t.dtype().as_str())
        .text("device", format!("{:?}", t.device().location()))
        .numbers("shape", t.dims())
        .numbers("strides", t.stride())
        .number("offset", t.layout().start_offset() as u64)
        .boolean("contiguous", t.is_contiguous())
        .text("accumulator_dtype", "unobserved")
        .text("alias_generation", "unobserved")
}

pub(crate) fn storage(s: &Storage, l: &Layout) -> Payload {
    Payload::new()
        .text("storage_dtype", s.dtype().as_str())
        .text("device", format!("{:?}", s.device().location()))
        .numbers("shape", l.dims())
        .numbers("strides", l.stride())
        .number("offset", l.start_offset() as u64)
        .boolean("contiguous", l.is_contiguous())
}

pub struct Scope {
    #[cfg(feature = "d7-trace")]
    inner: Option<capture::ActiveScope>,
    _same_thread: PhantomData<Rc<()>>,
}

#[inline]
pub fn enabled() -> bool {
    #[cfg(feature = "d7-trace")]
    {
        capture::enabled()
    }
    #[cfg(not(feature = "d7-trace"))]
    {
        false
    }
}

#[inline]
pub fn event(name: &str, payload: impl FnOnce() -> Payload) {
    #[cfg(feature = "d7-trace")]
    capture::event(name, payload);
    #[cfg(not(feature = "d7-trace"))]
    let _ = (name, payload);
}

#[inline]
pub fn scope(name: &str, payload: impl FnOnce() -> Payload) -> Scope {
    scope_in_window(name, None, payload)
}

#[inline]
pub fn scope_in_window(
    name: &str,
    window: Option<&str>,
    payload: impl FnOnce() -> Payload,
) -> Scope {
    #[cfg(feature = "d7-trace")]
    let inner = capture::begin(name, window, payload);
    #[cfg(not(feature = "d7-trace"))]
    let _ = (name, window, payload);
    Scope {
        #[cfg(feature = "d7-trace")]
        inner,
        _same_thread: PhantomData,
    }
}

#[cfg(feature = "d7-trace")]
impl Drop for Scope {
    fn drop(&mut self) {
        if let Some(s) = self.inner.take() {
            s.end();
        }
    }
}

#[cfg(feature = "d7-trace")]
pub use capture::Capture;

#[cfg(feature = "d7-trace")]
mod capture {
    use super::*;
    use serde::Serialize;
    use std::{
        cell::RefCell,
        fs::{self, OpenOptions},
        io::{self, Write},
        path::Path,
        sync::{
            atomic::{AtomicBool, AtomicU64, Ordering},
            Arc, Mutex, OnceLock, RwLock,
        },
        time::{Duration, Instant},
    };

    static ENABLED: AtomicBool = AtomicBool::new(false);
    static CURRENT: OnceLock<RwLock<Option<Arc<Collector>>>> = OnceLock::new();
    const DRAIN_TIMEOUT: Duration = Duration::from_secs(1);
    const DRAIN_POLL: Duration = Duration::from_millis(1);
    thread_local! { static STACK: RefCell<Vec<(u64, String)>> = const { RefCell::new(Vec::new()) }; }
    fn current() -> &'static RwLock<Option<Arc<Collector>>> {
        CURRENT.get_or_init(|| RwLock::new(None))
    }
    pub(super) fn enabled() -> bool {
        ENABLED.load(Ordering::Relaxed)
    }

    #[derive(Serialize)]
    struct Event {
        schema: &'static str,
        event: String,
        seq: usize,
        run_id: String,
        nonce: String,
        process_id: u32,
        thread_id: String,
        clock_domain: String,
        ts_ns: u64,
        scope_id: Option<u64>,
        parent_id: Option<u64>,
        window_id: String,
        observation: &'static str,
        payload: Payload,
    }
    struct Collector {
        run_id: String,
        nonce: String,
        start: Instant,
        capacity: usize,
        records: Mutex<Vec<Event>>,
        dropped: AtomicU64,
        open: AtomicU64,
        next_id: AtomicU64,
        closed: AtomicBool,
        nvtx: bool,
    }
    impl Collector {
        fn emit(
            &self,
            name: &str,
            scope_id: Option<u64>,
            parent_id: Option<u64>,
            window: &str,
            payload: Payload,
        ) {
            let Ok(mut records) = self.records.try_lock() else {
                self.dropped.fetch_add(1, Ordering::Relaxed);
                return;
            };
            if records.len() >= self.capacity || self.closed.load(Ordering::Relaxed) {
                self.dropped.fetch_add(1, Ordering::Relaxed);
                return;
            }
            let seq = records.len();
            records.push(Event {
                schema: "flow-adaptive-tuner.trace-event.v1",
                event: name.into(),
                seq,
                run_id: self.run_id.clone(),
                nonce: self.nonce.clone(),
                process_id: std::process::id(),
                thread_id: format!("{:?}", std::thread::current().id()),
                clock_domain: format!("host-monotonic-{}", std::process::id()),
                ts_ns: self.start.elapsed().as_nanos().min(u64::MAX as u128) as u64,
                scope_id,
                parent_id,
                window_id: window.into(),
                observation: "producer_asserted",
                payload,
            });
        }
    }
    fn collector() -> Option<Arc<Collector>> {
        if !enabled() {
            return None;
        }
        let slot = current().read().ok()?;
        if !enabled() {
            return None;
        }
        slot.clone()
    }
    fn parent() -> (Option<u64>, String) {
        STACK.with(|s| {
            s.borrow()
                .last()
                .map(|(id, w)| (Some(*id), w.clone()))
                .unwrap_or((None, "unclassified".into()))
        })
    }
    pub(super) fn event(name: &str, payload: impl FnOnce() -> Payload) {
        if let Some(c) = collector() {
            let (p, w) = parent();
            c.emit(name, None, p, &w, payload());
        }
    }
    pub(super) struct ActiveScope {
        c: Arc<Collector>,
        id: u64,
        parent: Option<u64>,
        name: String,
        window: String,
        nvtx: bool,
    }
    pub(super) fn begin(
        name: &str,
        window: Option<&str>,
        payload: impl FnOnce() -> Payload,
    ) -> Option<ActiveScope> {
        let c = collector()?;
        let (p, w) = parent();
        let w = if w == "capture" {
            w
        } else {
            window.unwrap_or(&w).to_owned()
        };
        let id = c.next_id.fetch_add(1, Ordering::Relaxed);
        c.open.fetch_add(1, Ordering::Relaxed);
        c.emit(&format!("{name}.begin"), Some(id), p, &w, payload());
        STACK.with(|s| s.borrow_mut().push((id, w.clone())));
        let nvtx = c.nvtx && nvtx::push(&format!("d7:{}:{id}:{name}", c.nonce));
        Some(ActiveScope {
            c,
            id,
            parent: p,
            name: name.into(),
            window: w,
            nvtx,
        })
    }
    impl ActiveScope {
        pub(super) fn end(self) {
            if self.nvtx {
                nvtx::pop();
            }
            STACK.with(|s| {
                let mut s = s.borrow_mut();
                if s.last().map(|x| x.0) != Some(self.id) {
                    self.c.dropped.fetch_add(1, Ordering::Relaxed);
                }
                s.retain(|x| x.0 != self.id);
            });
            self.c.emit(
                &format!("{}.end", self.name),
                Some(self.id),
                self.parent,
                &self.window,
                Payload::new().text(
                    "status",
                    if std::thread::panicking() {
                        "panic"
                    } else {
                        "scope_exit_not_completion"
                    },
                ),
            );
            self.c.open.fetch_sub(1, Ordering::Relaxed);
        }
    }

    pub struct Capture {
        c: Arc<Collector>,
        metadata: Payload,
        finished: bool,
    }
    impl Capture {
        pub fn start(
            run_id: String,
            nonce: String,
            capacity: usize,
            metadata: Payload,
            nvtx: bool,
        ) -> io::Result<Self> {
            if capacity == 0 || capacity > 2_000_000 || run_id.is_empty() || nonce.is_empty() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "invalid capture identity/capacity",
                ));
            }
            let mut slot = current()
                .write()
                .map_err(|_| io::Error::other("capture lock poisoned"))?;
            if slot.is_some() {
                return Err(io::Error::other("another capture is active"));
            }
            let c = Arc::new(Collector {
                run_id,
                nonce,
                start: Instant::now(),
                capacity,
                records: Mutex::new(Vec::with_capacity(capacity.min(8192))),
                dropped: AtomicU64::new(0),
                open: AtomicU64::new(0),
                next_id: AtomicU64::new(1),
                closed: AtomicBool::new(false),
                nvtx: nvtx && nvtx::available(),
            });
            *slot = Some(c.clone());
            ENABLED.store(true, Ordering::Release);
            Ok(Self {
                c,
                metadata,
                finished: false,
            })
        }
        pub fn finish(mut self, directory: &Path) -> io::Result<()> {
            {
                let _slot = current()
                    .write()
                    .map_err(|_| io::Error::other("capture lock poisoned"))?;
                ENABLED.store(false, Ordering::Release);
            }
            // Reserve the slot until admitted scopes and payload builders have released their collectors.
            let deadline = Instant::now() + DRAIN_TIMEOUT;
            while Arc::strong_count(&self.c) > 2 && Instant::now() < deadline {
                std::thread::sleep(DRAIN_POLL);
            }
            let drain_complete = Arc::strong_count(&self.c) == 2;
            let records = self
                .c
                .records
                .lock()
                .map_err(|_| io::Error::other("capture records poisoned"))?;
            self.c.closed.store(true, Ordering::Release);
            let dropped = self.c.dropped.load(Ordering::Acquire);
            let open = self.c.open.load(Ordering::Acquire);
            *current()
                .write()
                .map_err(|_| io::Error::other("capture lock poisoned"))? = None;
            self.finished = true;
            fs::create_dir_all(directory)?;
            let events = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(directory.join("events.ndjson"))?;
            let mut writer = io::BufWriter::new(events);
            for record in records.iter() {
                serde_json::to_writer(&mut writer, record)?;
                writer.write_all(b"\n")?;
            }
            writer.flush()?;
            let manifest = serde_json::json!({"schema":"flow-adaptive-tuner.trace-manifest.v1",
                "producer":"candle", "run_id":self.c.run_id,"nonce":self.c.nonce,"process_id":std::process::id(),
                "events":"events.ndjson","record_count":records.len(),"dropped_records":dropped,"open_scopes":open,
                "flush_complete":true,"drain_complete":drain_complete,
                "complete":drain_complete && dropped==0 && open==0,"profile":"diagnostic",
                "timestamp_unit":"ns","metadata":self.metadata,
                "capabilities":{
                    "storage_dispatch":"enabled", "tensor_dataflow":"partial", "asd_exact":"enabled_if_cuda",
                    "engine_context":"application_hooks", "gpu_completion":"external_collector_required",
                    "device_stream_ids":"unavailable", "cuda_graph_nodes":"unavailable",
                    "allocator_peak":"unavailable", "nvtx":if self.c.nvtx {"enabled"} else {"unavailable_or_disabled"},
                    "hardware_counters":"unavailable", "async_request_propagation":"unavailable", "speculative_checkpoints":"unavailable", "backend_algorithm_identity":"partial", "tensor_allocations":"unavailable"}});
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(directory.join("trace-manifest.json"))?;
            serde_json::to_writer_pretty(&mut file, &manifest)?;
            file.write_all(b"\n")?;
            Ok(())
        }
    }
    impl Drop for Capture {
        fn drop(&mut self) {
            if !self.finished {
                ENABLED.store(false, Ordering::Release);
                if let Ok(mut slot) = current().write() {
                    *slot = None;
                }
                self.c.closed.store(true, Ordering::Release);
            }
        }
    }

    mod nvtx {
        #[cfg(unix)]
        use std::sync::OnceLock;
        #[cfg(unix)]
        type Push = unsafe extern "C" fn(*const libc::c_char) -> libc::c_int;
        #[cfg(unix)]
        type Pop = unsafe extern "C" fn() -> libc::c_int;
        #[cfg(unix)]
        fn api() -> Option<(Push, Pop)> {
            static API: OnceLock<Option<(Push, Pop)>> = OnceLock::new();
            *API.get_or_init(|| unsafe {
                let handle = libc::dlopen(
                    c"libnvToolsExt.so.1".as_ptr(),
                    libc::RTLD_NOW | libc::RTLD_LOCAL,
                );
                if handle.is_null() {
                    return None;
                }
                let push = libc::dlsym(handle, c"nvtxRangePushA".as_ptr());
                let pop = libc::dlsym(handle, c"nvtxRangePop".as_ptr());
                if push.is_null() || pop.is_null() {
                    libc::dlclose(handle);
                    return None;
                }
                Some((
                    std::mem::transmute::<*mut libc::c_void, Push>(push),
                    std::mem::transmute::<*mut libc::c_void, Pop>(pop),
                ))
            })
        }
        pub(super) fn available() -> bool {
            #[cfg(unix)]
            {
                api().is_some()
            }
            #[cfg(not(unix))]
            {
                false
            }
        }
        pub(super) fn push(name: &str) -> bool {
            #[cfg(unix)]
            {
                if let (Some((push, _)), Ok(name)) = (api(), std::ffi::CString::new(name)) {
                    unsafe {
                        push(name.as_ptr());
                    }
                    return true;
                }
            }
            #[cfg(not(unix))]
            let _ = name;
            false
        }
        pub(super) fn pop() {
            #[cfg(unix)]
            if let Some((_, pop)) = api() {
                unsafe {
                    pop();
                }
            }
        }
    }
}
