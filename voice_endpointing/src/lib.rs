//! Two independent native inference modules with a shared audio/controller layer.
//! Conversion and reference comparison live in tools/ and are never imported here.
pub mod endpoint;
pub mod firered;
pub mod frontend;
pub mod runner;
pub mod smart_turn;
pub mod weights;
