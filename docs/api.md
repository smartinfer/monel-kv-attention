# Public API Contract — rust-kv-attention-v1 (frozen)

Crate `kv_attention`. Everything below MUST be reachable from the crate root
with exactly these names and signatures. Behavior is defined in `spec.md`.

```rust
pub const MAX_ABS_INPUT: f64 = 1.0e100;

#[derive(Debug, Clone, PartialEq)]
pub enum AttentionError {
    InvalidConfig,
    DimensionMismatch { expected: usize, found: usize },
    NonFiniteInput,
    ValueOutOfRange,
    CapacityExceeded { capacity: usize },
    EmptyCache,
}

impl std::fmt::Display for AttentionError { /* any human-readable message */ }
impl std::error::Error for AttentionError {}

pub struct KvCache { /* private */ }

impl KvCache {
    pub fn new(capacity: usize, dim: usize) -> Result<KvCache, AttentionError>;
    pub fn capacity(&self) -> usize;
    pub fn dim(&self) -> usize;
    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
    pub fn is_full(&self) -> bool;
    pub fn append(&mut self, key: &[f64], value: &[f64]) -> Result<(), AttentionError>;
    pub fn key(&self, index: usize) -> Option<&[f64]>;
    pub fn value(&self, index: usize) -> Option<&[f64]>;
    pub fn reset(&mut self);
}

pub fn attend(query: &[f64], cache: &KvCache) -> Result<Vec<f64>, AttentionError>;

pub struct CausalAttention { /* private */ }

impl CausalAttention {
    pub fn new(capacity: usize, dim: usize) -> Result<CausalAttention, AttentionError>;
    pub fn step(&mut self, query: &[f64], key: &[f64], value: &[f64])
        -> Result<Vec<f64>, AttentionError>;
    pub fn cache(&self) -> &KvCache;
    pub fn reset(&mut self);
}
```

Placement: `KvCache` is defined in `src/kv_cache.rs`; `attend`,
`CausalAttention`, `AttentionError`, and `MAX_ABS_INPUT` may be defined in
`src/attention.rs` or `src/lib.rs`; `src/lib.rs` re-exports all public items.
