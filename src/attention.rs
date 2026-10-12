use std::error::Error;
use std::f64;
use std::fmt;

use crate::KvCache;

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

impl fmt::Display for AttentionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AttentionError::InvalidConfig => write!(f, "invalid configuration"),
            AttentionError::DimensionMismatch { expected, found } => {
                write!(f, "dimension mismatch: expected {}, found {}", expected, found)
            }
            AttentionError::NonFiniteInput => write!(f, "non-finite input"),
            AttentionError::ValueOutOfRange => write!(f, "value out of range"),
            AttentionError::CapacityExceeded { capacity } => {
                write!(f, "capacity exceeded: {}", capacity)
            }
            AttentionError::EmptyCache => write!(f, "empty cache"),
        }
    }
}

impl Error for AttentionError {}

pub fn attend(query: &[f64], cache: &KvCache) -> Result<Vec<f64>, AttentionError> {
    let dim = cache.dim();
    if query.len() != dim {
        return Err(AttentionError::DimensionMismatch { expected: dim, found: query.len() });
    }
    for &x in query.iter() {
        if !x.is_finite() {
            return Err(AttentionError::NonFiniteInput);
        }
    }
    for &x in query.iter() {
        if x.abs() > MAX_ABS_INPUT {
            return Err(AttentionError::ValueOutOfRange);
        }
    }
    if cache.is_empty() {
        return Err(AttentionError::EmptyCache);
    }

    let n = cache.len();
    let d = dim as f64;
    let scale = d.sqrt();

    let mut scores = Vec::with_capacity(n);
    let mut max_score = f64::NEG_INFINITY;
    for i in 0..n {
        let k = cache.key(i).expect("index < len");
        let mut dot = 0.0;
        for j in 0..dim {
            dot += query[j] * k[j];
        }
        let s = dot / scale;
        if s > max_score {
            max_score = s;
        }
        scores.push(s);
    }

    let mut exp_scores = Vec::with_capacity(n);
    let mut sum_exp = 0.0;
    for s in scores.iter() {
        let e = (s - max_score).exp();
        exp_scores.push(e);
        sum_exp += e;
    }

    let mut weights = Vec::with_capacity(n);
    for e in exp_scores.iter() {
        weights.push(e / sum_exp);
    }

    let mut output = vec![0.0; dim];
    for i in 0..n {
        let v = cache.value(i).expect("index < len");
        let w = weights[i];
        for j in 0..dim {
            output[j] += w * v[j];
        }
    }

    Ok(output)
}

#[derive(Debug, Clone)]
pub struct CausalAttention {
    cache: KvCache,
}

impl CausalAttention {
    pub fn new(capacity: usize, dim: usize) -> Result<CausalAttention, AttentionError> {
        Ok(CausalAttention { cache: KvCache::new(capacity, dim)? })
    }

    pub fn step(
        &mut self,
        query: &[f64],
        key: &[f64],
        value: &[f64],
    ) -> Result<Vec<f64>, AttentionError> {
        let dim = self.cache.dim();
        if query.len() != dim {
            return Err(AttentionError::DimensionMismatch { expected: dim, found: query.len() });
        }
        if key.len() != dim {
            return Err(AttentionError::DimensionMismatch { expected: dim, found: key.len() });
        }
        if value.len() != dim {
            return Err(AttentionError::DimensionMismatch { expected: dim, found: value.len() });
        }

        for &x in query.iter() {
            if !x.is_finite() {
                return Err(AttentionError::NonFiniteInput);
            }
        }
        for &x in key.iter() {
            if !x.is_finite() {
                return Err(AttentionError::NonFiniteInput);
            }
        }
        for &x in value.iter() {
            if !x.is_finite() {
                return Err(AttentionError::NonFiniteInput);
            }
        }

        for &x in query.iter() {
            if x.abs() > MAX_ABS_INPUT {
                return Err(AttentionError::ValueOutOfRange);
            }
        }
        for &x in key.iter() {
            if x.abs() > MAX_ABS_INPUT {
                return Err(AttentionError::ValueOutOfRange);
            }
        }
        for &x in value.iter() {
            if x.abs() > MAX_ABS_INPUT {
                return Err(AttentionError::ValueOutOfRange);
            }
        }

        if self.cache.is_full() {
            return Err(AttentionError::CapacityExceeded { capacity: self.cache.capacity() });
        }

        self.cache.append(key, value)?;
        attend(query, &self.cache)
    }

    pub fn cache(&self) -> &KvCache {
        &self.cache
    }

    pub fn reset(&mut self) {
        self.cache.reset();
    }
}
