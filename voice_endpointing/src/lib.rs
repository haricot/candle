//! Audio/controller SDK using the two independent candle-transformers models.
//! Conversion and reference comparison live in tools/ and are never imported here.
pub use candle_transformers::models::{
    fireredvad as firered_model, smart_turn as smart_turn_model,
};
pub mod endpoint;
pub mod firered;
pub mod frontend;
pub mod runner;
pub mod smart_turn;
pub mod weights;
