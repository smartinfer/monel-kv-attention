//! Public smoke tests for rust-kv-attention-v1. Visible to the builder and run
//! by the evaluator as an integration test. They check API shape and a few
//! exact cases only; the evaluator's held-out suite is broader.

use kv_attention::{attend, AttentionError, CausalAttention, KvCache, MAX_ABS_INPUT};

#[test]
fn construction_rejects_zero_sizes() {
    assert_eq!(KvCache::new(0, 2).err(), Some(AttentionError::InvalidConfig));
    assert_eq!(KvCache::new(2, 0).err(), Some(AttentionError::InvalidConfig));
    assert!(CausalAttention::new(1, 1).is_ok());
}

#[test]
fn append_and_read_back() {
    let mut cache = KvCache::new(2, 2).unwrap();
    assert!(cache.is_empty());
    cache.append(&[1.0, 2.0], &[3.0, 4.0]).unwrap();
    assert_eq!(cache.len(), 1);
    assert_eq!(cache.key(0), Some(&[1.0, 2.0][..]));
    assert_eq!(cache.value(0), Some(&[3.0, 4.0][..]));
    assert_eq!(cache.key(1), None);
}

#[test]
fn first_step_returns_its_value_exactly() {
    let mut attn = CausalAttention::new(4, 3).unwrap();
    let out = attn.step(&[0.5, -1.0, 2.0], &[1.0, 0.0, -1.0], &[7.0, -8.5, 0.25]).unwrap();
    assert_eq!(out, vec![7.0, -8.5, 0.25]);
}

#[test]
fn equal_scores_average_values() {
    let mut attn = CausalAttention::new(2, 1).unwrap();
    attn.step(&[0.0], &[1.0], &[2.0]).unwrap();
    let out = attn.step(&[0.0], &[5.0], &[4.0]).unwrap();
    assert!((out[0] - 3.0).abs() < 1e-12);
}

#[test]
fn full_cache_is_rejected_without_change() {
    let mut cache = KvCache::new(1, 1).unwrap();
    cache.append(&[1.0], &[1.0]).unwrap();
    assert_eq!(cache.append(&[2.0], &[2.0]), Err(AttentionError::CapacityExceeded { capacity: 1 }));
    assert_eq!(cache.len(), 1);
    assert_eq!(cache.key(0), Some(&[1.0][..]));
}

#[test]
fn dimension_and_value_errors() {
    let mut cache = KvCache::new(2, 2).unwrap();
    assert_eq!(cache.append(&[1.0], &[1.0, 1.0]), Err(AttentionError::DimensionMismatch { expected: 2, found: 1 }));
    assert_eq!(cache.append(&[f64::NAN, 0.0], &[1.0, 1.0]), Err(AttentionError::NonFiniteInput));
    assert_eq!(cache.append(&[2.0 * MAX_ABS_INPUT, 0.0], &[1.0, 1.0]), Err(AttentionError::ValueOutOfRange));
    assert!(cache.is_empty());
    assert_eq!(attend(&[1.0, 1.0], &cache), Err(AttentionError::EmptyCache));
}

#[test]
fn reset_empties_the_cache() {
    let mut attn = CausalAttention::new(1, 1).unwrap();
    attn.step(&[1.0], &[1.0], &[1.0]).unwrap();
    attn.reset();
    assert!(attn.cache().is_empty());
    assert_eq!(attn.step(&[1.0], &[1.0], &[9.0]).unwrap(), vec![9.0]);
}
