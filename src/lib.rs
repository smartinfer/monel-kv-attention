#![forbid(unsafe_code)]

mod kv_cache;
mod attention;

pub use attention::{attend, CausalAttention, AttentionError, MAX_ABS_INPUT};
pub use kv_cache::KvCache;

#[cfg(test)]
mod tests;
