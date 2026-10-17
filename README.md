# almide/nnlib

Neural-network kernels for [Almide](https://github.com/almide/almide):
attention, rotary position embedding, row norms, activations, quantised
Q1_0 / Q8_0 packed weights and fused Qwen3 decoder layers.

These 25 functions were part of the stdlib `matrix` module until dialect
epoch 14, which moved them here unchanged: same names, same parameters, same
results, bit for bit (see [Verification](#verification)). The package is pure
Almide over the stdlib's generic `Matrix` operations (`matrix.mul`,
`matrix.linear_row`, `matrix.append_rows`, ...), so the same code runs natively
and with `--target wasm`.

The name follows Julia's NNlib.jl, the library of neural-network primitives
(activations, softmax, attention) that model code builds on.

## Install

```bash
almide add almide/nnlib@v0.1.0
```

which adds to `almide.toml`:

```toml
[dependencies]
nnlib = { git = "https://github.com/almide/nnlib", tag = "v0.1.0" }
```

## Usage

```almide
import nnlib

effect fn main() -> Unit = {
  let x = matrix.from_lists([[0.5, -1.0, 2.0, 0.25], [1.0, 0.0, -0.5, 3.0]])
  let h = nnlib.rms_norm_rows(x, [1.0, 1.0, 1.0, 1.0], 0.00001)
  let r = nnlib.rope_rotate_at(h, 2, 2, 10000.0, 7)
  let a = nnlib.masked_multi_head_attention(r, r, r, 2)
  println("${matrix.to_lists(nnlib.gelu(a))}")
}
```

`examples/llama_block.almd` is one Llama-style decoder layer.

## Functions

| Function | |
|---|---|
| `gelu(m)` | Elementwise GELU, tanh approximation. |
| `silu_mul(a, b)` | Elementwise `x * sigmoid(x)` of `a`, times `b`. |
| `swiglu_gate(x, w_gate, w_up)` | `silu(x Wg^T) * (x Wu^T)`; empty input gives 0 x 0. |
| `softmax_rows(m)` | Row-wise softmax, max-subtracted; a row holding NaN or +inf, or all -inf, is all NaN. |
| `linear_row_gelu(x, weight, bias)` | `gelu(matrix.linear_row(x, weight, bias))`. |
| `fused_gemm_bias_scale_gelu(a, b, bias, alpha)` | `gelu(matrix.scale(matrix.add(matrix.mul(a, b), bias), alpha))`. |
| `rms_norm_rows(m, gamma, eps)` | Row-wise RMS norm; columns cut to `gamma`. |
| `layer_norm_rows(m, gamma, beta, eps)` | Row-wise layer norm; columns cut to `gamma` / `beta`. |
| `pre_norm_linear(x, gamma, beta, eps, weight, bias)` | `matrix.linear_row(layer_norm_rows(x, ...), weight, bias)`. |
| `causal_mask_add(m, mask_val)` | `mask_val` added where column > row. |
| `attention_weights(q, kt, scale)` | `softmax_rows(matrix.scale(matrix.mul(q, kt), scale))`. |
| `scaled_dot_product_attention(q, kt, v, scale)` | `matrix.mul(attention_weights(q, kt, scale), v)`. |
| `multi_head_attention(q, k, v, n_heads)` | Attention over `n_heads`; `n_heads < 1` aborts; a NaN or +inf score row is NaN. |
| `masked_multi_head_attention(q, k, v, n_heads)` | Causal multi-head attention; `n_heads < 1` aborts. |
| `rope_rotate(x, n_heads, head_dim, theta_base)` | Interleaved RoPE, position = row; bad head sizes abort. |
| `rope_rotate_at(x, n_heads, head_dim, theta_base, start_pos)` | RoPE with row `p` at `start_pos + p`; a negative start is 0. |
| `rope_rotate_neox_at(x, n_heads, head_dim, theta_base, start_pos)` | NeoX RoPE: pairs `(j, j + head_dim / 2)`. |
| `from_q1_0_bytes(data, offset, rows, cols)` | Q1_0 1-bit decode; blocks past the data are 0.0. |
| `select_rows_q1_0(data, offset, cols, row_ids)` | Q1_0 rows at `row_ids`; past-data rows are 0. |
| `linear_q1_0_row_no_bias(x, w_bytes, w_offset, w_rows, w_cols)` | `x` times Q1_0-packed `W^T`. |
| `select_rows_q8_0_dq(data, offset, cols, row_ids)` | Q8_0 rows dequantized; 0 past the data or if `cols % 32 > 0`. |
| `linear_q8_0_row_no_bias(x, w_bytes, w_offset, w_rows, w_cols)` | `x` times Q8_0 `W^T`; all 0 if `w_cols % 32 > 0`. |
| `qwen3_block_q1_0_kv(h, k_cache, v_cache, w, gamma_offs, weight_offs, start_pos, n_q_heads, n_kv_heads, head_dim, ffn_hidden, rope_theta, eps)` | One Qwen3 decoder layer over Q1_0 weights; returns the new `(h, k_cache, v_cache)`. |
| `qwen3_block_f32_kv(...)` (same parameters) | One Qwen3 decoder layer over f32 weights, NeoX RoPE. |
| `qwen3_block_q8_0_kv(...)` (same parameters) | One Qwen3 decoder layer over Q8_0 weights (norm gammas f32), NeoX RoPE. |

Q1_0 is 18 bytes per 128 weights (fp16 scale, then 16 bytes of sign bits,
LSB first: bit 1 is `+scale`, bit 0 is `-scale`). Q8_0 is 34 bytes per 32
weights (fp16 scale, then 32 signed int8 quants). The Qwen3 blocks read the
weights straight from the model's byte buffer: `gamma_offs` holds the byte
offsets of the `[attn_norm, q_norm, k_norm, ffn_norm]` f32 gammas and
`weight_offs` those of the `[q, k, v, o, gate, up, down]` weights.

## Migrating from the stdlib (dialect epoch 14)

Add the dependency, `import nnlib`, and change the module prefix. Nothing else
changes: parameters, results and the bits of every result are the same.

```almide
// before (epoch 13)
let y = matrix.gelu(x)

// after (epoch 14)
import nnlib
let y = nnlib.gelu(x)
```

| Before (stdlib) | After (`import nnlib`) |
|---|---|
| `matrix.qwen3_block_f32_kv(...)` | `nnlib.qwen3_block_f32_kv(...)` |
| `matrix.qwen3_block_q1_0_kv(...)` | `nnlib.qwen3_block_q1_0_kv(...)` |
| `matrix.qwen3_block_q8_0_kv(...)` | `nnlib.qwen3_block_q8_0_kv(...)` |
| `matrix.from_q1_0_bytes(...)` | `nnlib.from_q1_0_bytes(...)` |
| `matrix.linear_q1_0_row_no_bias(...)` | `nnlib.linear_q1_0_row_no_bias(...)` |
| `matrix.select_rows_q1_0(...)` | `nnlib.select_rows_q1_0(...)` |
| `matrix.linear_q8_0_row_no_bias(...)` | `nnlib.linear_q8_0_row_no_bias(...)` |
| `matrix.select_rows_q8_0_dq(...)` | `nnlib.select_rows_q8_0_dq(...)` |
| `matrix.attention_weights(...)` | `nnlib.attention_weights(...)` |
| `matrix.scaled_dot_product_attention(...)` | `nnlib.scaled_dot_product_attention(...)` |
| `matrix.multi_head_attention(...)` | `nnlib.multi_head_attention(...)` |
| `matrix.masked_multi_head_attention(...)` | `nnlib.masked_multi_head_attention(...)` |
| `matrix.rope_rotate(...)` | `nnlib.rope_rotate(...)` |
| `matrix.rope_rotate_at(...)` | `nnlib.rope_rotate_at(...)` |
| `matrix.rope_rotate_neox_at(...)` | `nnlib.rope_rotate_neox_at(...)` |
| `matrix.gelu(...)` | `nnlib.gelu(...)` |
| `matrix.silu_mul(...)` | `nnlib.silu_mul(...)` |
| `matrix.swiglu_gate(...)` | `nnlib.swiglu_gate(...)` |
| `matrix.linear_row_gelu(...)` | `nnlib.linear_row_gelu(...)` |
| `matrix.fused_gemm_bias_scale_gelu(...)` | `nnlib.fused_gemm_bias_scale_gelu(...)` |
| `matrix.rms_norm_rows(...)` | `nnlib.rms_norm_rows(...)` |
| `matrix.layer_norm_rows(...)` | `nnlib.layer_norm_rows(...)` |
| `matrix.pre_norm_linear(...)` | `nnlib.pre_norm_linear(...)` |
| `matrix.causal_mask_add(...)` | `nnlib.causal_mask_add(...)` |
| `matrix.softmax_rows(...)` | `nnlib.softmax_rows(...)` |

The generic matrix operations (`matrix.mul`, `transpose`, `add`, `scale`,
`map`, `from_lists`, `to_lists`, `get`, `rows`, `cols`, `linear_row`,
`linear_row_no_bias`, `linear_f32_row_no_bias`, `select_rows_f32`,
`from_bytes_f32_le`, `append_rows`, ...) stay in the stdlib.

Two things a migration can notice:

- The stdlib rewrote a few unfused chains into these kernels on its own
  (`matrix.gelu(matrix.linear_row(x, w, b))` became `linear_row_gelu`, and so
  on). The fused and unfused spellings always gave the same bits, so writing
  `nnlib.gelu(matrix.linear_row(x, w, b))` or `nnlib.linear_row_gelu(x, w, b)` is a
  choice of style, not of result.
- An out-of-domain head count or head geometry still aborts with exit code 1
  and the same message, but a package has no public way to print the stdlib's
  bare `Error: ...` line, so stderr reads `PANIC: Error: head count must be
  positive` (and `PANIC: Error: head geometry exceeds row width`), without a
  trailing newline: the language's user-panic line (contract C-219), the same
  on native and wasm. Every other abort (`Error: matrix shape mismatch`,
  `Error: matrix dimensions too large`, `Error: index out of bounds`) is the
  stdlib's own line, raised by the stdlib operation that owns it.

## Verification

- `spec/`: the stdlib's own `test` blocks for these kernels
  (`matrix_{transformer,fused_chain,linear_row_gelu,attention_weights,sdpa,rms_swiglu,pre_norm_linear,llama_block,kernel_e2e}_test`)
  plus `pinned_test.almd`. Run them on wasm with
  `almide test spec/ --target wasm` and on native with
  `bash scripts/test-native.sh` (plain `almide test` prefers the wasm leg).
- `tests/cross/` and `tests/fail/`: the stdlib's cross-target fixtures for
  these kernels, with the stdout, stderr and exit code the stdlib produced
  before epoch 14 as the expectation. `bash scripts/test-golden.sh` runs
  each on native and on wasm.
- A differential harness (kept outside this repository, since it calls the
  `matrix.X` spellings epoch 14 removed) ran 4588 generated cases — random
  and edge-value inputs (NaN, +-inf, -0.0, f64::MAX, the -708/708 exp clamp,
  -1e9 masks, empty and r x 0 shapes, ragged pairs, out-of-buffer and
  negative offsets, wrapped row ids) — through `matrix.X` and `nnlib.X` on both
  targets and compared the result bits. The port agrees with itself across
  native and wasm on every case and with the stdlib wherever the stdlib was
  correct; the exceptions are under Known issues.

## Known issues

- **RoPE on native, against the stdlib built before the fix.** The stdlib's
  native RoPE used the host's `sin_cos`, whose last ULP depends on the
  platform libm; its wasm leg used the vendored `sin` / `cos`. This package
  uses the vendored `math.sin` / `math.cos` on both targets, which is the
  stdlib's corrected behaviour: it matches the stdlib's wasm leg on every
  case, and differs by 1 ULP from an older native stdlib in about a quarter
  of the cases at larger positions.
- **Q1_0 packed GEMV on aarch64, against an older stdlib build.** The
  stdlib's native `linear_q1_0_row_no_bias` in the 0.67.1 development build
  this port was checked against returned every output negated on Apple
  Silicon (its own `from_q1_0_bytes` decode disagreed with it; the stdlib's
  fixture `q1_packed_is_decoded` printed `false`), and so did
  `qwen3_block_q1_0_kv`. This package follows the documented convention
  (bit 1 is `+x`) and agrees with the stdlib's wasm implementation and with
  `matrix.linear_row_no_bias` over the decoded weights.
- **Q8_0 on x86 AVX2.** The stdlib's native x86 AVX2 Q8_0 dot folded the
  activation's sign into the weight; a `-128` weight quant cannot be negated
  in an i8, so it flipped sign against a negative activation (exact answer
  `4095.999984741211` for the pinned case; the stdlib printed it negated).
  This package computes the exact integer dot in Almide, so it has no such
  lane and gives `4095.999984741211` on every target. The divergence the
  stdlib showed between its native and wasm `qwen3_block_q8_0_kv` on x86 is
  consistent with this bug; that has not yet been confirmed on a real AVX2
  machine.

## Performance

The stdlib ran these as native intrinsics (SIMD fast-exp, BLAS where it
applied); this package runs them as Almide list code. Measured on an Apple
M4 Pro, `almide build --release`, median of 5 runs, setup time subtracted:

| Workload | stdlib intrinsic | `nnlib` | ratio |
|---|---|---|---|
| `gelu`, 256 x 1024 matrix | 0.9 - 1.1 ms | 6.8 - 9.4 ms | 7 - 11x slower |
| `qwen3_block_f32_kv`, hidden 256, 4 q / 2 kv heads of 64, FFN 512, 16-row prefill | 3.0 - 3.6 ms | 4.3 - 5.0 ms | 1.2 - 1.7x slower |

The elementwise kernels pay the most; the Qwen3 block pays less because
its projections (`matrix.linear_f32_row_no_bias`) are still stdlib
intrinsics. The package uses no `native/` fast path: every function is pure
Almide, so the wasm build needs nothing beyond the package source.

## License

Dual-licensed under either of [MIT](LICENSE-MIT) or
[Apache-2.0](LICENSE-APACHE), at your option.
