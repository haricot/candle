//! Dependency-neutral DKR transport, logical registry and bounded plan cache.
//! No tensor framework, CUDA driver, environment selection or launch ABI here.
//! Adapters remain responsible for semantic manifests and actual launch arguments.
use sha2::{Digest,Sha256};
use std::collections::{HashMap,VecDeque};
use std::hash::Hash;

pub const NATIVE_ABI: &str = "candle.native.ptx.v1";
pub const TYPED_ABI: &str = "dkr.typed.v1";
pub const MAX_PTX_BYTES: usize = 32*1024*1024;

#[derive(Clone,Debug,PartialEq,Eq,Hash)]
pub struct RegistryKey {
    pub abi:String,
    pub operation:String,
    pub implementation:String,
    pub target_cc:u32,
}
impl RegistryKey {
    pub fn new(abi:&str, operation:&str, implementation:&str, target_cc:u32)->Self {
        Self{abi:abi.into(),operation:operation.into(),implementation:implementation.into(),target_cc}
    }
}

pub struct Registry<T> { entries: HashMap<RegistryKey,T> }
impl<T> Default for Registry<T> {
    fn default()->Self { Self{entries:HashMap::new()} }
}
impl<T> Registry<T> {
    pub fn insert(&mut self,key:RegistryKey,value:T)->Result<(),String> {
        use std::collections::hash_map::Entry;
        match self.entries.entry(key) {
            Entry::Vacant(e)=>{e.insert(value);Ok(())},
            Entry::Occupied(e)=>Err(format!("duplicate DKR registry key {:?}",e.key())),
        }
    }
    pub fn get(&self,key:&RegistryKey)->Option<&T>{self.entries.get(key)}
    pub fn values(&self)->impl Iterator<Item=&T>{self.entries.values()}
    pub fn len(&self)->usize{self.entries.len()}
    pub fn is_empty(&self)->bool{self.entries.is_empty()}
    /// Select the highest eligible target, never silently resolve equal-target ties.
    pub fn select(&self,abi:&str,operation:&str,implementation:Option<&str>,cc:u32,
                  admit:impl Fn(&T)->bool)->Result<&T,String> {
        let mut best:Option<(&RegistryKey,&T)>=None;
        let mut ambiguous=false;
        for (key,value) in &self.entries {
            if key.abi!=abi || key.operation!=operation || key.target_cc>cc ||
                implementation.is_some_and(|id|key.implementation!=id) || !admit(value) {continue}
            match best {
                Some((prior,_)) if prior.target_cc>key.target_cc=>{},
                Some((prior,_)) if prior.target_cc==key.target_cc=>{ambiguous=true;},
                _=>{best=Some((key,value));ambiguous=false;},
            }
        }
        if ambiguous{return Err("ambiguous dynamic kernel variants at the same architecture level".into())}
        best.map(|(_,v)|v).ok_or_else(||"no compatible dynamic kernel package".into())
    }
}

#[derive(Clone,Debug,PartialEq,Eq,Hash)]
pub struct ArtifactKey { target_cc:u32, sha256:String }
impl ArtifactKey {
    pub fn target_cc(&self)->u32{self.target_cc}
    pub fn sha256(&self)->&str{&self.sha256}
    pub fn module_key(&self)->String{format!("dkr.ptx.v1:sm_{}:{}",self.target_cc,self.sha256)}
}

/// Immutable verified PTX. A physical cache key is content-based, not a logical
/// operation name. Different ABI adapters may share identical bytes only after
/// independently enforcing their own semantic contracts.
#[derive(Debug)]
pub struct VerifiedArtifact { key:ArtifactKey, ptx:String, entries:std::collections::BTreeSet<String> }
impl VerifiedArtifact {
    pub fn new(ptx:String,target_cc:u32,expected_sha:&str)->Result<Self,String>{
        if ptx.len()>MAX_PTX_BYTES{return Err("PTX artifact budget exceeded".into())}
        let sha=format!("{:x}",Sha256::digest(ptx.as_bytes()));
        if !sha.eq_ignore_ascii_case(expected_sha){return Err("PTX artifact digest mismatch".into())}
        // Strip comments before examining directives/symbols. Preserve line breaks.
        let mut clean=String::with_capacity(ptx.len());
        let mut chars=ptx.chars().peekable();
        while let Some(ch)=chars.next(){
            if ch=='/' && chars.peek()==Some(&'/'){
                chars.next();
                for next in chars.by_ref(){if next=='\n'{clean.push('\n');break}}
            }else if ch=='/' && chars.peek()==Some(&'*'){
                chars.next();let mut closed=false;let mut prev=' ';
                for next in chars.by_ref(){if prev=='*' && next=='/'{closed=true;break}if next=='\n'{clean.push('\n')}prev=next;}
                if !closed{return Err("unterminated PTX comment".into())}
                clean.push(' ');
            }else{clean.push(ch)}
        }
        let mut target=None;let mut width=None;
        let mut entries=std::collections::BTreeSet::new();
        let mut words=clean.split_whitespace();
        while let Some(word)=words.next(){
            match word {
                ".target"=>{
                    let t=words.next().ok_or("missing PTX target")?;
                    if target.replace(t.trim_end_matches(',')).is_some(){return Err("multiple PTX targets".into())}
                }
                ".address_size"=>{
                    let v=words.next().ok_or("missing PTX address size")?;
                    if width.replace(v).is_some(){return Err("multiple PTX address sizes".into())}
                }
                ".entry"=>{
                    let name=words.next().ok_or("missing PTX entry")?.split('(').next().unwrap_or("");
                    if name.is_empty() || !entries.insert(name.to_owned()){return Err("duplicate or empty PTX entry".into())}
                }
                _=>{}
            }
        }
        let wanted=format!("sm_{target_cc}");
        // NVCC may append an architecture modifier after the comma.
        if target!=Some(wanted.as_str()) || width!=Some("64") || entries.is_empty(){
            return Err("PTX target, address width or entry inventory mismatch".into())
        }
        Ok(Self{key:ArtifactKey{target_cc,sha256:sha},ptx,entries})
    }
    pub fn key(&self)->&ArtifactKey{&self.key}
    pub fn ptx(&self)->&str{&self.ptx}
    pub fn has_entry(&self,name:&str)->bool{self.entries.contains(name)}
}

/// Bounded, deterministic LRU. No device assumption or work on a cache hit.
/// Keys must carry device identity and the complete ABI/shape signature.
pub struct PlanCache<K,P,const N:usize>{entries:VecDeque<(K,P)>}
impl<K:Eq,P,const N:usize> PlanCache<K,P,N>{
    pub const fn new()->Self{Self{entries:VecDeque::new()}}
    pub fn len(&self)->usize{self.entries.len()}
    pub fn with_plan<T,E>(&mut self,key:K,prepare:impl FnOnce()->Result<P,E>,
                         launch:impl FnOnce(&P)->Result<T,E>)->Result<T,E>{
        assert!(N>0,"zero DKR plan cache capacity");
        if self.entries.front().is_none_or(|(k,_)|k!=&key){
            let entry=if let Some(i)=self.entries.iter().position(|(k,_)|k==&key){
                self.entries.remove(i).expect("existing plan")
            }else{(key,prepare()?)};
            self.entries.push_front(entry);
            if self.entries.len()>N{self.entries.pop_back();}
        }
        launch(&self.entries.front().expect("resident plan").1)
    }
}

#[cfg(test)]
mod tests{
    use super::*;
    #[test] fn registry_namespaces_and_ambiguity(){
        let mut r=Registry::default();
        r.insert(RegistryKey::new(NATIVE_ABI,"linear","a",61),1).unwrap();
        r.insert(RegistryKey::new(TYPED_ABI,"linear","a",61),2).unwrap();
        r.insert(RegistryKey::new(TYPED_ABI,"linear","b",61),3).unwrap();
        assert!(r.insert(RegistryKey::new(TYPED_ABI,"linear","a",61),9).is_err());
        assert_eq!(r.select(NATIVE_ABI,"linear",None,61,|_|true).unwrap(),&1);
        assert!(r.select(TYPED_ABI,"linear",None,61,|_|true).is_err());
        assert_eq!(r.select(TYPED_ABI,"linear",Some("a"),61,|_|true).unwrap(),&2);
        r.insert(RegistryKey::new(TYPED_ABI,"linear","c",75),4).unwrap();
        assert_eq!(r.select(TYPED_ABI,"linear",None,75,|_|true).unwrap(),&4);
        assert!(r.select(TYPED_ABI,"linear",None,50,|_|true).is_err());
    }
    #[test] fn verifies_content_and_directives(){
        let p=".version 8.0\n.target sm_61\n.address_size 64\n.visible .entry foo() { ret; }";
        let sha=format!("{:x}",Sha256::digest(p.as_bytes()));
        let a=VerifiedArtifact::new(p.into(),61,&sha).unwrap();
        assert!(a.has_entry("foo"));
        assert!(VerifiedArtifact::new(p.into(),75,&sha).is_err());
        assert!(VerifiedArtifact::new(p.into(),61,&"0".repeat(64)).is_err());
        let comment="// .target sm_61\n.address_size 64\n.entry foo() {}";
        assert!(VerifiedArtifact::new(comment.into(),61,&format!("{:x}",Sha256::digest(comment))).is_err());
    }
    #[test] fn lru_preserves_recent_and_failed_prepare(){
        let mut c=PlanCache::<u32,u32,2>::new();
        for k in [1,2,1,3]{
            c.with_plan(k,||Ok::<_,()>(k*10),|p|Ok(*p)).unwrap();
        }
        assert_eq!(c.len(),2);
        assert_eq!(c.with_plan(1,||Err::<u32,_>(()),|p|Ok(*p)),Ok(10));
        assert!(c.with_plan(2,||Err::<u32,_>(()),|p|Ok(*p)).is_err());
        assert_eq!(c.len(),2);
    }
}
