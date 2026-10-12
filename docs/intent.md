# Incremental Causal Attention with a KV Cache: User Intent

Build a small Rust library for single-head, CPU-only causal attention that is
evaluated one token at a time.

A caller decodes a sequence token by token. For each new token the caller
supplies three vectors of the same dimension: a query, a key, and a value. The
library remembers every key and value it has accepted so far in a bounded,
append-only cache, adds the new token's key and value to that cache, and then
returns the attention output for the new token: a weighted average of all
remembered values (including the new one), where each value's weight reflects
how well its key matches the new query.

The cache has a fixed capacity chosen when it is created. When it is full, the
library refuses further tokens instead of growing, dropping old entries, or
overwriting anything. The caller can reset the cache to empty and start a new
sequence.

The library must be predictable when given bad input. Vectors of the wrong
length, non-numeric values (NaN, infinities), values outside the documented
magnitude range, or a full cache are reported as errors, and a failed call
must leave the cache exactly as it was.

Results must agree with a straightforward recomputation of causal attention
over the whole prefix, within documented floating-point tolerances, while the
library itself reuses the cached keys and values instead of recomputing
earlier tokens.

The precise contract, API, numeric policy, and constraints are in `spec.md`
and `api.md`. Keep the implementation small, clear, and dependency-free.
