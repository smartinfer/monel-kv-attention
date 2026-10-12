use crate::{attend, AttentionError, CausalAttention, KvCache, MAX_ABS_INPUT};

#[test]
fn single_entry_exact_value() {
    let mut cache = KvCache::new(4, 3).unwrap();
    let value = [7.0, -8.5, 0.25];
    cache.append(&[1.0, 0.0, -1.0], &value).unwrap();
    let out = attend(&[0.5, -1.0, 2.0], &cache).unwrap();
    assert_eq!(out, value);
}

#[test]
fn empty_cache_error() {
    let cache = KvCache::new(2, 2).unwrap();
    assert_eq!(attend(&[1.0, 1.0], &cache), Err(AttentionError::EmptyCache));
}

#[test]
fn step_uses_cache_length() {
    let mut attn = CausalAttention::new(3, 1).unwrap();
    let out1 = attn.step(&[0.0], &[1.0], &[2.0]).unwrap();
    assert_eq!(out1, vec![2.0]);
    let out2 = attn.step(&[0.0], &[3.0], &[4.0]).unwrap();
    assert!((out2[0] - 3.0).abs() < 1e-12);
}

#[test]
fn value_validation_on_append() {
    let mut cache = KvCache::new(1, 1).unwrap();
    let too_big = 2.0 * MAX_ABS_INPUT;
    assert_eq!(cache.append(&[too_big], &[1.0]), Err(AttentionError::ValueOutOfRange));
}
