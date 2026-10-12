# Technical Specification — rust-kv-attention-v1 (frozen)

Version 1. Normative; the words MUST / MUST NOT are requirements. The API
signatures are in `api.md`.

## 1. Scope

- Rust (edition 2021), CPU only, one attention head, `f64` arithmetic.
- The caller supplies query, key, and value vectors; there are no learned
  projections, batching, multiple heads, positional encodings, masks over
  padding, eviction, GPU code, quantization, or `unsafe` code.
- Crate name `kv_attention`, library only, **no dependencies** (none in
  `[dependencies]`, `[dev-dependencies]`, or `[build-dependencies]`), no build
  script.

## 2. Dimensions and storage

- `dim` (d) is the length of every query, key, and value vector, fixed at
  construction, `dim >= 1`.
- `capacity` (C) is the maximum number of cached entries, fixed at
  construction, `capacity >= 1`.
- The cache stores accepted (key, value) pairs in insertion order. Entry `i`
  (0-based) is the `i`-th successfully appended pair since construction or the
  last reset. `key(i)` and `value(i)` return slices of length `dim` that are
  **bitwise identical** to the slices passed to the append that created entry
  `i`, for every `i < len()`, and `None` for `i >= len()`. The internal layout
  is free (e.g. row-major contiguous buffers), provided these accessors hold.
- Successful appends never modify existing entries.

## 3. Attention formula

For a query `q` and the `n >= 1` cached entries `(k_i, v_i)`, `i = 0..n-1`:

```
s_i = (sum_j q_j * k_i[j]) / sqrt(d)        // scaled dot-product score
m   = max_i s_i
e_i = exp(s_i - m)                            // numerically stable softmax
Z   = sum_i e_i
w_i = e_i / Z
out[j] = sum_i w_i * v_i[j]                   // output, length d
```

- The softmax MUST be computed in the max-subtracted (or an equivalently
  overflow-free) form. Implementations that exponentiate raw scores fail on
  documented valid inputs.
- Summation order is free; agreement is judged with the tolerances in §8.
- With exactly one cached entry, the output MUST equal that entry's value
  exactly (`==` per element, so `+0.0` and `-0.0` compare equal), as the
  formula gives `w_0 = 1`. The sign of a zero output is not specified.

## 4. Incremental causal decoding

`CausalAttention::step(query, key, value)`:

1. validates the inputs (§6);
2. appends `(key, value)` to its cache — **the current token is appended
   before attention**, so it attends to itself and to every earlier token;
3. returns `attend(query, cache)` over all `n = len()` entries.

Token `t` (0-based) therefore attends to tokens `0..=t`. This MUST be computed
from the cached keys and values; it MUST NOT recompute outputs for earlier
tokens. Per-step work SHOULD be O(n·d) time and O(n + d) extra memory; an
n×n attention matrix MUST NOT be materialized.

## 5. Capacity, reset

- A successful append when `len() < capacity` increments `len()` by one.
- When `len() == capacity`, append and step return
  `Err(AttentionError::CapacityExceeded { capacity })` and change nothing.
  There is no growth, eviction, wrap-around, or overwrite.
- `reset()` sets `len()` to 0; afterwards `key(i)`/`value(i)` return `None`
  for every `i`; `capacity` and `dim` are unchanged; the cache can be filled to
  `capacity` again. Reset of an empty cache is a no-op.

## 6. Validation and error policy

Constant: `MAX_ABS_INPUT = 1.0e100`. A vector element `x` is **valid** when it
is finite and `|x| <= MAX_ABS_INPUT`. This bound keeps every score, softmax
weight, and output finite for all valid inputs.

Checks are applied in this exact order; the first failing check determines the
error:

| operation | 1. dimension | 2. non-finite | 3. range | 4. state |
|---|---|---|---|---|
| `KvCache::new` / `CausalAttention::new` | `capacity == 0` or `dim == 0` → `InvalidConfig` | — | — | — |
| `append(key, value)` | key, then value length ≠ dim → `DimensionMismatch { expected: dim, found: len }` | any NaN/±inf in key, then value → `NonFiniteInput` | any `|x| > MAX_ABS_INPUT` in key, then value → `ValueOutOfRange` | full → `CapacityExceeded { capacity }` |
| `attend(query, cache)` | query length ≠ dim → `DimensionMismatch` | query non-finite → `NonFiniteInput` | query out of range → `ValueOutOfRange` | cache empty → `EmptyCache` |
| `step(query, key, value)` | query, key, value lengths in that order | query, key, value | query, key, value | full → `CapacityExceeded` |

- **Failure atomicity:** every operation that returns `Err` MUST leave the
  cache unchanged: same `len()`, same entries (bitwise), same capacity and dim.
- Valid inputs MUST never produce NaN or infinite outputs, and no operation may
  panic on any input (including empty slices and extreme valid values).
- Subnormal and signed-zero inputs are valid and must be handled normally.

## 7. Determinism

Repeating the same sequence of calls on a fresh instance MUST produce bitwise
identical outputs and errors.

## 8. Numerical agreement (how results are judged)

The evaluator compares each step's output `c` with an independent dense
full-prefix recomputation `o` (computed with compensated summation). For output
element `j` at a step with cached values `v_i`:

```
M      = max over cached entries i and elements j' of |v_i[j']|
D      = max over cached entries i of (sum_j |q_j * k_i[j]|) / sqrt(d)
u      = 2^-53
tol_j  = M * (1e-12 + 4 * (d + 2) * u * D) + 1e-10 * |o_j|
pass_j = c_j is finite  and  |c_j - o_j| <= tol_j
```

- A non-finite `c_j` always fails, regardless of tolerance.
- `M = 0` (all cached values zero) requires the exact output 0.
- The single-entry case (§3) requires exact equality (`==`).
- Bitwise comparison is used only for cache accessors (stored inputs, §2) and
  for repeated runs of the same implementation (§7).
- Agreement is graded on the evaluator's documented well-conditioned
  generators; inputs whose score differences fall below floating-point
  resolution are graded only for finiteness, determinism, and no panics.

## 9. Code constraints

- Exactly these source files: `src/lib.rs`, `src/kv_cache.rs`,
  `src/attention.rs`, `src/tests.rs` (unit tests, compiled under
  `#[cfg(test)]`). Plus `Cargo.toml`.
- At most **1,000 physical lines** across the four `.rs` files, comments and
  tests included. Production (`lib.rs`, `kv_cache.rs`, `attention.rs`) and test
  (`tests.rs`) lines are reported separately.
- `src/lib.rs` MUST contain `#![forbid(unsafe_code)]`.
- Write the smallest clear implementation that satisfies this contract:
  ordinary `rustfmt` formatting, meaningful names, no unnecessary abstractions
  or generics, no code golfing, and no removal of checks to reduce line count.
- The public API is exactly `api.md`; additional private items are allowed.
