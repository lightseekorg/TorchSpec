# Capturing Kimi-K3 Hidden States under AttnRes

This is the background you need before touching Kimi-K3 auxiliary hidden-state
capture, and the record of two bugs we fixed in `vllm_k3.patch`. It is written
in the order that makes the problem make sense rather than the order we found
things in.

## The thing that makes K3 different

In an ordinary transformer, "the hidden state after layer K-1" is a tensor. It
is threaded from one layer to the next, and every reader sees the same value.

Under AttnRes, K3 threads three things instead:

- `prefix_sum` — the running sum within the current block
- `block_residual`, the *bank* — the blocks committed so far, `(T, num_blocks, H)`
- a pending `delta`, the previous layer's MLP output, not yet folded in

and no reader consumes them directly. Every consumer — the next layer's
attention front-end, and the model's own output-side aggregation — first runs a
softmax mixture over the bank and the prefix. From the kernel:

```
sources = bank[0..num_blocks-1] + [prefix]         # num_blocks + 1 of them
logit(v) = (v · (norm_w ⊙ proj_w)) * rsqrt(mean(v²) + eps)
mixed    = Σ softmax(logits)_i · v_i               # a convex combination
```

The query vector `norm_w ⊙ proj_w` comes from the *consumer's* own
`self_attention_res_norm.weight` and `self_attention_res_proj.weight`.

**So "the hidden state after layer K-1" is not a property of the residual
stream. It is defined by whichever layer reads it.** Every consequence below
follows from that sentence.

## Three quantities people confuse

| Quantity | What it is | Who reads it |
|---|---|---|
| `prefix_sum + pending_mlp_out` | the running prefix on its own | **nobody** |
| `mixed` | the softmax mixture, using layer K's score weights | the tap, and the drafter |
| `mixed` normed by layer K's `input_layernorm` | layer K's attention input | only layer K |

The drafter is trained against the second row. The third row is layer K's
private input preprocessing — the drafter has its own `fc_norm`
(`"fc_norm": true` in `configs/draft_models/kimi_k3_dspark_mla.json`), so
feeding it a value already normed by the target model stacks two unrelated
normalisations.

The first row is what the pre-tap code recorded, and what a PP boundary used to
fall back to. It is not an approximation of the second row; it is one of
`num_blocks + 1` sources.

## What a "tap" is

A tap is a probe on the residual stream, in the plumbing sense: it reads the
signal without interrupting it. Concretely it is the same `attn_res` call the
consumer would make, with the output norm turned off:

```python
attn_res(prefix, None, bank, score_norm, score_proj, None, num_blocks=...,
         block_write_idx=-1, ...)
#                      ^^^^ output_norm_weight = None -> returns `mixed`
```

Three things make it a pure read, and all three matter:

- `delta=None` — the kernel's `if HAS_DELTA: tl.store(prefix_ptr, ...)` does not fire
- `block_write_idx=-1` — `if WRITE_BLOCK: tl.store(blocks_ptr, ...)` does not fire
- the `prefix` passed in is `prefix_sum + pending_mlp_out`, freshly computed, not
  the live `prefix_sum`

Verified directly (`.tmp/attnres-pp/verify_tap_is_pure_read.py`): after the call,
`prefix` and the bank are bitwise unchanged, the output aliases neither, and
running it twice gives the same answer.

This is why the tap cannot reuse the consumer's call. The consumer's call has
side effects — it folds the delta back into the prefix and commits a block into
the bank — and it returns the *normed* value. `tests/models/kimi_k3/test_eagle3.py`
pins both conditions (`delta is None`, `block_write_idx == -1`), because an
end-to-end PP comparison cannot catch a regression here: both sides would do the
same corrupting tap and the damage would cancel.

## Why the tap is a second computation at all

`attn_res` has exactly one output pointer:

```
149:  output = mixed
154:  if APPLY_OUTPUT_NORM:
162:      output = mixed * rstd * output_norm_weight
164:  tl.store(output_ptr + ..., output, ...)
```

It writes either the mixture or the normed value, never both — even though at
line 162 both are live in registers. That single-exit interface, not the maths,
is why a tapped layer computes the mixture twice.

Note the shape of the fix that is *not* available: unfusing the norm into a
separate kernel would force `mixed` out to memory and back. These kernels are
entirely memory-bound, so that costs two extra passes over `(T, H)` on all 93
layers × 2 calls:

| | passes per forward | traffic at T=32768, H=7168, bf16 |
|---|---|---|
| fused (today) | 1536 | 672 GiB |
| norm as a separate kernel | 1908 | 835 GiB (+24%) |

The cheap version is a second optional output pointer, used only on tapped
layers: `+5` passes to write `mixed`, `-40` passes because the separate tap call
disappears, net **-15.3 GiB**. That needs both the Triton kernel and
`csrc/libtorch_stable/kimi_k3/attn_res_kernel.cu` changed — changing only Triton
would force tapped layers off the native path and make the model's arithmetic
depend on the capture config, which is the bug in the next section. So it means
recompiling vLLM, which our patch stack is built to avoid. Not done.

## Bug 1: the tap at a pipeline boundary

If capture id `K` is exactly a non-last stage's `end_layer`, layer `K` lives on
the next rank, so its score weights are not here.

The upstream behaviour ([vLLM #50487](https://github.com/vllm-project/vllm/pull/50487),
still open) is to fall back to the running prefix. In the language above, that
forces the softmax one-hot onto a single source and discards the rest. For a
stage ending at layer 48 with `attn_res_block_size=8` that is 1 of 7 sources. In
a harness driving the real kernel with synthetic weights, cosine similarity
between the fallback and the real mixture ran **0.05 to 0.60** — a different
feature, not a nearby one.

The damage is that the exported feature becomes a function of the pipeline
partition. The same capture id yields the mixture at pp=1 and the raw prefix at a
pp that happens to cut there, while the drafter at serving time always consumes
the mixture. It is silent: right shape, right dtype, no NaNs.

**Fix: keep a copy of the two vectors.** Everything else the tap needs is already
local, including one detail that makes it exact — this stage's
`num_attn_res_blocks` is `cdiv(end_layer, block_size)`, which is precisely layer
`end_layer`'s `prev_valid_blocks`, so the bank is already the right length rather
than merely present. What is missing is two `hidden_size` vectors, unsharded and
unquantised, about **28KB** together.

They are redirected out of the checkpoint by `_maybe_boundary_attn_res_name`
before the PP filter drops them as belonging to a layer this rank does not own,
and initialised to NaN so a checkpoint that never supplies them fails loudly
instead of exporting a mixture over uninitialised memory.

Rejecting boundary ids was our first attempt. It does not survive mostly-PP
sharding: for 93 layers with capture ids at {8, 32, 48, 64, 88} under the default
`get_pp_indices`, the first collision is at pp=10 (id 64), pp=12 has none, pp=16
has two, and from pp=40 on all five are stage ends.

## Bug 2: the kernel dispatch at a pipeline boundary

Found while measuring bug 1, and unrelated to capture — it affects plain serving.

The native fused op is only eligible when, among other conditions,
`delta is not None` and `block_write_idx < 0`. The handoff folds the delta into
the prefix and the receiving stage started its layers with no delta, so **the
first layer of every non-first stage ran the Triton fallback** while the same
layer at pp=1 ran the native op. They agree to within bf16 rounding and then
compound, so the arithmetic — every tap after that point, not just the boundary
one — depended on where the pipeline was cut.

**Fix: hand that layer an explicit zero delta.** Block-write layers never reach
the native path from either side, so the filler is skipped for them and a
block-aligned partition allocates nothing.

## What was verified

On GB300 against the real `attn_res` kernel, comparing a two-stage run against a
single-stage one at every possible split point:

| stack | splits | boundary tap | all taps | final hidden states |
|---|---|---|---|---|
| 24 layers / block 4 | 23 | 23 | 23 | 23 |
| 32 layers / block 8 | 31 | 31 | 31 | 31 |

Bit-identical, not close. Before the bug 2 fix, end-to-end identity held at 4 of
31 splits, and those four were exactly the block-write-aligned cuts — which is
also the independent confirmation of the dispatch mechanism.

Caveat on provenance: the harness reproduces K3's AttnRes bookkeeping and drives
the real kernel, but with synthetic weights. Bit-identity is a structural claim
and holds regardless; the cosine range above is illustrative.

Test-suite effect, against the pinned image with `--noconftest` and
`test_latent_moe_tail.py` excluded (it fails to collect on pristine too):
`tests/models/kimi_k3` goes from 59 failed / 26 passed to 59 failed / 29 passed,
with the failing set identical line for line. Those 59 are GPU tests that cannot
run on a CPU-only box — `current_platform.device_type` is `''` there, so
`torch.device('')` raises.

## What capture costs

Per tapped layer, one extra `attn_res` reading `num_blocks + 1` sources. For the
93-layer stage-1 recipe:

| capture id | 8 | 32 | 48 | 64 | 88 | total |
|---|---|---|---|---|---|---|
| source reads | 2 | 5 | 7 | 9 | 12 | 35 |

With the write of each result that is 40 passes over `(T, H)`, against the 1536
the model's own AttnRes calls already do — about **2.6%** of AttnRes traffic,
17.5 GiB, and a smaller fraction of the whole forward. It grows with the depth
of the tapped layer, so where you tap matters more than how many times.

Worth keeping in proportion: K3's PP handoff carries the bank, so it is
`1 + cdiv(start_layer, block_size)` tensors of `(T, H)` rather than one. At
T=32768 and H=7168 one such tensor is 0.44 GiB:

| cut at | tensors on the wire | per handoff |
|---|---|---|
| 24 | 4 | 1.75 GiB |
| 47 / 48 | 7 | 3.06 GiB |
| 88 | 12 | 5.25 GiB |

pp=2 moves 3.1 GiB per step, pp=4 moves 9.2 GiB, pp=8 moves 22.3 GiB — growing
superlinearly, since deeper cuts also carry more blocks. That is over NVLink/IB
rather than HBM, so any optimisation effort belongs there long before it belongs
in capture's 17.5 GiB of local traffic.

## Choosing where to cut

Two independent reasons to prefer PP boundaries at multiples of
`attn_res_block_size`:

- the first layer of the stage is then a block-write layer, which takes the
  Triton path from either side — this was the zero-code workaround for bug 2 and
  is still the cheapest configuration, since the filler delta is skipped
- shallower cuts move less bank

Boundaries coinciding with capture ids are fine now. That is what bug 1's fix
buys, and it matters because the MLA layer pattern puts the natural capture ids
on the same multiples of 8 that make good block-aligned cuts.

`aux_hidden_states_layers` in the YAML is **pre-shift**: the engine adds one, so
`7` means "the residual stream after layer 7 ran" and becomes capture id 8.

## Where things live

| | |
|---|---|
| model changes | `patches/vllm/nightly-7794b1e08.../vllm_k3.patch` |
| tests | `patches/vllm/nightly-7794b1e08.../tests/vllm_k3_tests.patch` (not in `series`) |
| equivalence harness | `.tmp/attnres-pp/verify_boundary_tap.py` |
| purity check | `.tmp/attnres-pp/verify_tap_is_pure_read.py` |
| dispatch probe | `.tmp/attnres-pp/verify_native_vs_triton.py` |

Neither fix went upstream. Bug 2 would have made a clean standalone vLLM PR, and
bug 1 belongs in #50487, which currently ships the fallback this replaces along
with a test asserting it. See `UPSTREAM_EXPORT_PLAN.md` (M13) for that decision.
