use crate::attention::AttentionError;
use crate::MAX_ABS_INPUT;

#[derive(Debug, Clone)]
pub struct KvCache {
    capacity: usize,
    dim: usize,
    len: usize,
    keys: Vec<f64>,
    values: Vec<f64>,
}

impl KvCache {
    pub fn new(capacity: usize, dim: usize) -> Result<KvCache, AttentionError> {
        if capacity == 0 || dim == 0 {
            return Err(AttentionError::InvalidConfig);
        }
        let size = capacity.checked_mul(dim).ok_or(AttentionError::InvalidConfig)?;
        Ok(KvCache {
            capacity,
            dim,
            len: 0,
            keys: vec![0.0; size],
            values: vec![0.0; size],
        })
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn is_full(&self) -> bool {
        self.len == self.capacity
    }

    pub fn append(&mut self, key: &[f64], value: &[f64]) -> Result<(), AttentionError> {
        if key.len() != self.dim {
            return Err(AttentionError::DimensionMismatch { expected: self.dim, found: key.len() });
        }
        if value.len() != self.dim {
            return Err(AttentionError::DimensionMismatch { expected: self.dim, found: value.len() });
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

        if self.is_full() {
            return Err(AttentionError::CapacityExceeded { capacity: self.capacity });
        }

        let idx = self.len * self.dim;
        self.keys[idx..idx + self.dim].copy_from_slice(key);
        self.values[idx..idx + self.dim].copy_from_slice(value);
        self.len += 1;
        Ok(())
    }

    pub fn key(&self, index: usize) -> Option<&[f64]> {
        if index >= self.len {
            return None;
        }
        let start = index * self.dim;
        let end = start + self.dim;
        Some(&self.keys[start..end])
    }

    pub fn value(&self, index: usize) -> Option<&[f64]> {
        if index >= self.len {
            return None;
        }
        let start = index * self.dim;
        let end = start + self.dim;
        Some(&self.values[start..end])
    }

    pub fn reset(&mut self) {
        self.len = 0;
    }
}
