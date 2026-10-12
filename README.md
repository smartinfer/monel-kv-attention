# kv_attention

> **MONEL STAMP**: **Developed by Monel. Verified and assured by Monel.**
> Verdict: **ASSURED** (22/22 graded tests + 5/5 bounded Kani proofs).
> Run `r01`, 2026-10-11. Model gpt-5.1, run by the Monel orchestrator.
> Generation cost **$0.0722** (5 model calls, 37 s).
> Machine-readable stamp: [`MONEL.json`](MONEL.json).

Monel has two parts:
- The **Monel orchestrator** ran the build. It prepared an isolated workspace,
  launched the coding agent in a sandbox, then independently verified and assured
  the result.
- The **Monel coding agent** wrote this crate, using gpt-5.1.

`kv_attention` is a small, dependency-free Rust library for single-head, CPU-only,
`f64` **incremental causal attention over a fixed-capacity key/value cache**. A
caller decodes one token at a time:

1. For each token it supplies a query, a key and a value.
2. The library appends the key and value to the cache.
3. It returns the attention output for the new token over every cached entry.

The cache never grows. When it is full, further tokens are refused. Bad input is
reported as an error and leaves the cache exactly as it was: wrong lengths, NaN or
infinity, magnitudes above `1e100`, or a full cache.

The behavior is specified in [`docs/intent.md`](docs/intent.md),
[`docs/spec.md`](docs/spec.md) and [`docs/api.md`](docs/api.md).

## Usage

```rust
use kv_attention::CausalAttention;

let mut attention = CausalAttention::new(16, 4)?;   // capacity 16, dimension 4
let out = attention.step(&q, &k, &v)?;            // append (k, v), attend with q
```

```sh
cargo build
cargo test        # unit tests + public smoke tests
```

- **No dependencies.** It uses the standard library only.
- **No unsafe code.** The crate sets `#![forbid(unsafe_code)]`.
- **Size:** 327 lines across `src/`, plus 32 lines of unit tests.

## How it was built: the Monel orchestrator running the Monel coding agent

| | |
| --- | --- |
| Orchestrator | **Monel orchestrator**, revision `87e2ea7` |
| Coding agent | **Monel coding agent**, revision `d456b45`; it builds and tests its own code before finishing |
| Model | **gpt-5.1** (OpenAI) |
| Backend / API | OpenAI **Responses API** (`/v1/responses`) through Monel's native model transport (v0.4.4) |
| Model settings | reasoning effort `none`, max 4,096 output tokens per call, explicit prompt-cache key (cache policy `stable_prefix`) |
| Isolation | macOS sandbox: reads limited to the workspace, system paths and the Rust toolchain; offline Cargo; API key never visible to the agent's commands |
| Toolchain | Rust stable (cargo 1.97.1), aarch64-apple-darwin |
| Task | `rust-kv-attention-v1` from the Monel evaluation suite, task revision `b5de22e` |
| Input to the agent | public intent, spec, API, builder prompt and 7 smoke tests only; grader, held-out tests, reference and proofs were never visible |
| Attempts | one generation, no repair, no human intervention |

What the session did:

| Call | Action |
| --- | --- |
| 1 | Read the spec, API, intent, manifest, starter files and smoke tests |
| 2 | Wrote `src/lib.rs`, `src/kv_cache.rs`, `src/attention.rs` and `src/tests.rs` |
| 3 | Asked to run `cargo build` and `cargo test`; a tool-argument error was rejected as recoverable |
| 4 | Ran `cargo build --offline` and `cargo test --offline` (all tests pass) |
| 5 | Finished, citing the passing test run as verification evidence |

## Cost measurement

Token counts are as reported by the provider. Costs use OpenAI's gpt-5.1 list
prices: $1.25 per million input tokens, $0.125 per million cached input tokens,
and $10.00 per million output tokens (snapshot 2026-10-10). They are computed
estimates, not invoiced amounts.

| Call | Input tokens | of which cached | Output tokens | Latency | Cost (USD) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 7,253 | 0 | 656 | 7.0 s | 0.015626 |
| 2 | 13,360 | 7,168 | 2,992 | 20.0 s | 0.038556 |
| 3 | 16,800 | 13,184 | 193 | 3.0 s | 0.008098 |
| 4 | 17,441 | 16,896 | 193 | 2.5 s | 0.004723 |
| 5 | 18,482 | 17,536 | 187 | 3.0 s | 0.005245 |
| **Total** | **73,336** | **54,784 (75%)** | **4,221** | **35.4 s** | **0.072248** |

| Stage | Wall time | Model cost |
| --- | ---: | ---: |
| Generation (coding-agent session) | 36.9 s | $0.0722 |
| Independent grading (22 tests, structure) | 4.2 s | $0 |
| Bounded proofs (Kani K1–K5) | 56.6 s | $0 |

## How it was verified

Monel's verifier is independent of the agent.

| Check | Result |
| --- | --- |
| Behavior: 22 graded tests (7 public smoke, 1 exact-API conformity, 14 held-out laws) | **PASS 22/22** |
| Held-out laws: agreement with a dense oracle (Neumaier summation), stable softmax at 1e200-scale scores, error order and atomicity, reset/refill, no panics over a pathological input grid, determinism, `attend` purity | included above |
| Structure: four files, line budget, no dependencies, no `unsafe` | no issues |
| Bounded proofs (Kani 0.68.0): K1 construction, K2 append transitions and atomicity, K3 dimension-mismatch errors, K4 accessor bounds, K5 reset | **5/5 VERIFICATION SUCCESSFUL** |

**Assured** means both of the following hold:
- the behavior verdict is PASS on all 22 tests;
- all five Kani harnesses verify.

The record in [`verification/VERIFICATION.json`](verification/VERIFICATION.json)
includes the SHA-256 of every source file that was graded and proved.

### What "assured" does not mean

- The Kani proofs are **bounded**: capacity 3, dimension 2. No inductive argument
  extends them to larger sizes.
- Attention arithmetic is **tested**, not proved. It is checked against the oracle
  within the tolerance in `docs/spec.md` §8 on seeded and adversarial inputs. There
  is no general floating-point error bound.
- Atomicity of `CausalAttention::step` beyond the tested cases, and performance on
  other machines, are not established.
- One run shows this works; it is not a success rate.

## Provenance of the task

The task is derived from the KV-cache and attention behavior in Candle
(`c68b2499`, MIT OR Apache-2.0), with deliberate changes:
- no growth;
- failed calls change nothing (atomic);
- NaN and infinity are rejected;
- `f64` throughout.

The code here was written by the agent from the specification. The agent did not
see Candle or the private reference.
