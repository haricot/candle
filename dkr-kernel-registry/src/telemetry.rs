//! Opt-in, bounded process-local counters. No GPU operation or per-event I/O.
//! Read DKR_TELEMETRY=1 before first use; snapshot after quiescing caller threads.
//! Counters record host evidence, not GPU completion or graph replay.
use std::{collections::BTreeMap,sync::{Arc,Mutex,OnceLock,atomic::{AtomicU64,Ordering}}};
pub const MAX_ROWS:usize=4096;
#[derive(Clone,Copy)]
#[repr(usize)]
pub enum Event { Resolved, FunctionHit, FunctionMiss, ModuleLoaded, PlanHit, PlanMiss,
    PlanEvicted, PrepareError, Submitted, AdapterOk, AdapterError, FallbackSelected, Rejected }
const EVENTS:[&str;13]=["resolved","function_hit","function_miss","module_loaded","plan_hit","plan_miss",
    "plan_evicted","prepare_error","submitted","adapter_ok","adapter_error","fallback_selected","rejected"];
#[derive(Clone,Debug,Eq,Ord,PartialEq,PartialOrd)]
struct Key { abi:String, operation:String, entry:String, sha256:String, target_cc:u32 }
type Counters=[AtomicU64;EVENTS.len()];
#[derive(Default)]
pub struct Recorder { rows:Mutex<BTreeMap<Key,Arc<Counters>>>, dropped:AtomicU64 }
#[derive(Clone,Default)]
pub struct Handle(Option<Arc<Counters>>);
impl Handle{
    #[inline] pub fn record(&self,event:Event){
        if let Some(c)=&self.0 { c[event as usize].fetch_add(1,Ordering::Relaxed); }
    }
}
impl Recorder{
    pub fn handle(&self,abi:&str,operation:&str,entry:&str,sha256:&str,target_cc:u32)->Handle{
        let key=Key{abi:abi.into(),operation:operation.into(),entry:entry.into(),sha256:sha256.into(),target_cc};
        let Ok(mut rows)=self.rows.lock() else {self.dropped.fetch_add(1,Ordering::Relaxed);return Handle::default()};
        if let Some(c)=rows.get(&key){return Handle(Some(c.clone()))}
        if rows.len()>=MAX_ROWS{self.dropped.fetch_add(1,Ordering::Relaxed);return Handle::default()}
        let c=Arc::new(std::array::from_fn(|_|AtomicU64::new(0)));
        rows.insert(key,c.clone()); Handle(Some(c))
    }
    pub fn snapshot_json(&self)->String{
        fn q(s:&str)->String{
            let mut out=String::from("\"");
            for c in s.chars(){match c {'"'=>out.push_str("\\\""),'\\'=>out.push_str("\\\\"),
                c if c.is_control()=>out.push_str(&format!("\\u{:04x}",c as u32)),c=>out.push(c)}}
            out.push('"');out
        }
        let Ok(rows)=self.rows.lock() else{return "{\"schema\":\"DKR-RUNTIME-EVIDENCE-V1\",\"error\":\"poisoned\"}".into()};
        let mut out=format!("{{\"schema\":\"DKR-RUNTIME-EVIDENCE-V1\",\"pid\":{},\"dropped_rows\":{},\"rows\":[",
            std::process::id(),self.dropped.load(Ordering::Relaxed));
        for (i,(k,c)) in rows.iter().enumerate(){
            if i>0{out.push(',')}
            out.push_str(&format!("{{\"abi\":{},\"operation\":{},\"entry\":{},\"sha256\":{},\"target_cc\":{}",
                q(&k.abi),q(&k.operation),q(&k.entry),q(&k.sha256),k.target_cc));
            for (j,name) in EVENTS.iter().enumerate(){out.push_str(&format!(",\"{}\":{}",name,c[j].load(Ordering::Relaxed)))}
            out.push('}');
        }
        out.push_str("]}");out
    }
}
static GLOBAL:OnceLock<Option<Recorder>>=OnceLock::new();
fn configured()->Option<&'static Recorder>{
    GLOBAL.get_or_init(||(std::env::var_os("DKR_TELEMETRY").as_deref()==Some(std::ffi::OsStr::new("1"))).then(Recorder::default)).as_ref()
}
#[inline] pub fn handle(abi:&str,operation:&str,entry:&str,sha256:&str,target_cc:u32)->Handle{
    configured().map_or_else(Handle::default,|r|r.handle(abi,operation,entry,sha256,target_cc))
}
pub fn snapshot_json()->Option<String>{configured().map(Recorder::snapshot_json)}
/// Emits one cumulative snapshot at explicit scope exit, including on unwinding.
/// For nextest each test has its own process; no signal/atexit dependency.
pub struct EvidenceGuard;
impl Drop for EvidenceGuard{
    fn drop(&mut self){if let Some(s)=snapshot_json(){eprintln!("DKR_RUNTIME_EVIDENCE {s}")}}
}
#[cfg(test)]
mod tests{
    use super::*;
    #[test] fn separates_resolution_submission_and_fallback(){
        let r=Recorder::default();
        let a=r.handle("native","op","entry","digest",61);
        a.record(Event::Resolved);a.record(Event::FunctionMiss);a.record(Event::ModuleLoaded);
        a.record(Event::Submitted);
        r.handle("native","op","entry","digest",61).record(Event::Submitted);
        r.handle("native","op","fallback","",0).record(Event::FallbackSelected);
        let s=r.snapshot_json();
        assert!(s.contains("\"submitted\":2"));
        assert!(s.contains("\"resolved\":1"));
        assert!(s.contains("\"fallback_selected\":1"));
        assert!(s.contains("\"adapter_ok\":0"));
    }
    #[test] fn concurrent_handles_and_bound(){
        let r=Recorder::default();let a=r.handle("typed","op","symbol","hash",61);
        std::thread::scope(|s|for _ in 0..4{let a=a.clone();s.spawn(move||for _ in 0..100{a.record(Event::AdapterOk)});});
        assert!(r.snapshot_json().contains("\"adapter_ok\":400"));
        for i in 0..MAX_ROWS{r.handle("a",&i.to_string(),"","",61);}
        assert!(r.snapshot_json().contains("\"dropped_rows\":1"));
    }
}
