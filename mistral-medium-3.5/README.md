# Mistral-Medium-3.5-128B — Architecture Deep-Dive

Per-op / per-module analysis of Mistral's dense 128B, 256k-context, multimodal, hybrid-reasoning
flagship (open weights, Modified MIT license). Companion artifacts:
**[interactive deck](https://mstojkovictt.github.io/kernel-tutorials/mistral-medium-3.5/)** ·
**[one-page design sheet](https://mstojkovictt.github.io/kernel-tutorials/mistral-medium-3.5/design-sheet.html)**.

**Verification method — nothing below is quoted from memory.** Every hyperparameter comes from
the repo's `config.json` / `params.json`; every tensor shape was read out of the
**safetensors shard headers** of the actual checkpoint (HTTP range requests against
`model-0000{1,2,3}-of-00003.safetensors`, 2,465 tensors); execution-order semantics come from the
reference implementation in `transformers` v5.12.1 (`models/ministral3`, `models/mistral3`,
`models/pixtral`, `modeling_rope_utils.py`). Parameter totals and the checkpoint's byte count are
closed exactly, not approximately.

> **Config-fix note (use the corrected config).** The initially released `config.json` carried a
> YaRN bug: `text_config.rope_parameters.mscale_all_dim` was `1.0`. In
> `_compute_yarn_parameters` (transformers), the cos/sin attention factor is
> `attention_factor = get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dim)` with
> `get_mscale(s, m) = 0.1·m·ln(s) + 1`. With `mscale = mscale_all_dim = 1.0` the ratio is exactly
> `1.0` — YaRN's attention scaling was silently disabled, degrading long-context quality. Commit
> `c4be198` (**"Fixing Transformers config"**, 2026-05-01, juliendenize) set
> `mscale_all_dim: 0.0`; being falsy, that routes to the default branch
> `attention_factor = get_mscale(64) = 0.1·ln 64 + 1 ≈ 1.41589`. Everything below uses the
> corrected config at `main`.

---

## 0. Notation

| Symbol | Meaning | Value here |
|---|---|---|
| `B` | batch size | 1 (reference workload) |
| `S` | sequence length | 8192 prefill / decode @ 32k ctx (reference) |
| `d_model` | decoder hidden size | 12288 |
| `L` | decoder layer count | 88 |
| `H` | query heads | 96 |
| `H_kv` | KV heads (GQA) | 8 |
| `d_head` | head dimension | 128 |
| `d_ff` | MLP intermediate size | 28672 |
| `V` | vocabulary size | 131072 |

Sanity identity: `H · d_head = 96 · 128 = 12288 = d_model` (o_proj is square);
`H_kv · d_head = 1024`; `d_ff / d_model = 2.33`.

---

## 1. Source grounding

All fields from `mistralai/Mistral-Medium-3.5-128B` @ `main` (post-fix). `config.json` nests the
decoder under `text_config` (`model_type: "ministral3"`, wrapped by `model_type: "mistral3"`,
architecture `Mistral3ForConditionalGeneration`); `params.json` is the Mistral-native equivalent.

### Decoder (`config.json → text_config`, cross-checked vs `params.json`)

| Item | Config field | Value |
|---|---|---|
| hidden size | `hidden_size` (`dim`) | 12288 |
| layers | `num_hidden_layers` (`n_layers`) | 88 |
| query heads | `num_attention_heads` (`n_heads`) | 96 |
| KV heads | `num_key_value_heads` (`n_kv_heads`) | 8 |
| head dim | `head_dim` | 128 |
| MLP intermediate | `intermediate_size` (`hidden_dim`) | 28672 |
| vocab | `vocab_size` | 131072 |
| activation | `hidden_act` | `silu` (gated: SwiGLU) |
| norm | RMSNorm, `rms_norm_eps` (`norm_eps`) | 1e-05 |
| tied embeddings | `tie_word_embeddings` (`tied_embeddings`) | false |
| max context | `max_position_embeddings` | 262144 |
| sliding window | `sliding_window` | null (full attention, no SWA) |
| attention dropout | `attention_dropout` | 0.0 |
| biases | — (checkpoint) | none anywhere (no bias tensors in any shard) |
| QK-norm | — (checkpoint) | absent (no `q_norm`/`k_norm` tensors) |

### Positional encoding (`text_config.rope_parameters`, `params.json → yarn`)

| Item | Field | Value |
|---|---|---|
| RoPE base | `rope_theta` | 1,000,000.0 |
| scaling type | `rope_type` | `yarn` |
| factor | `factor` | 64.0 |
| pretrain window | `original_max_position_embeddings` | 4096 (× 64 = 262144 ✓) |
| ramp bounds | `beta_fast` / `beta_slow` (`beta`/`alpha` in params.json) | 4.0 / 1.0 (non-default: HF default is 32/1) |
| attention factor | `mscale` = 1.0, `mscale_all_dim` = **0.0** (the fix) | → cos/sin scaled by 0.1·ln 64 + 1 ≈ **1.41589** |
| llama-4 pos scaling | `llama_4_scaling_beta` | 0 (`params.json`: `llama_4_scaling: null`) — query scaling `1 + β·log(1+⌊pos/4096⌋)` present in `Ministral3Attention` but **disabled** |

### Vision + projector (`vision_config`, top-level fields; `params.json → vision_encoder`)

| Item | Field | Value |
|---|---|---|
| vision arch | `vision_config.model_type` | `pixtral` (trained from scratch per model card; variable image sizes/aspect ratios) |
| hidden / layers / heads | `hidden_size`, `num_hidden_layers`, `num_attention_heads`, `head_dim` | 1664 / 48 / 16 / 104 (16·104 = 1664 ✓) |
| vision MLP | `intermediate_size`, `hidden_act` | 8192, `silu` (gated, like the decoder) |
| patching | `patch_size`, `image_size`, `num_channels` | 14, 1540 (max), 3 → up to 110×110 = 12100 patches/image |
| vision RoPE | `rope_parameters.rope_theta` | 10000.0, 2-D (meshgrid) rope, `default` type |
| feature tap | `vision_feature_layer` | −1 (last layer) |
| projector | `spatial_merge_size`, `projector_hidden_act`, `multimodal_projector_bias` | 2 (merges 2×2 patches → ÷4 tokens), `gelu`, false |
| image token | `image_token_index` | 10 |

### Released quantization (this is how the weights ship)

`config.json → quantization_config`: `quant_method: "fp8"` (E4M3, `params.json`:
`fp8_e4m3`), static per-tensor activation scales, `weight_block_size: null` (per-tensor weight
scale). `modules_to_not_convert = [vision_tower, multi_modal_projector, lm_head]` stay bf16.
Checkpoint dtype census confirms: 121,802,588,160 params in `F8_E4M3` + 5,901,623,248 in `BF16`
(embed_tokens is also bf16), and the shards close **byte-exactly**:
121,802,588,160·1 + 5,901,623,248·2 + 2,464 scale-tensor bytes = **133,605,834,656 B** = the
index's `metadata.total_size`. ✓

---

## 2. Module & layer inventory

Exact tree, from the checkpoint's tensor names (`model.safetensors.index.json` + shard headers).
Shapes are `[out, in]` as stored. Note the real paths: the decoder lives under
`model.language_model.*` (not `model.*`), the vision MLP is `feed_forward` with
`attention_norm`/`ffn_norm`, and each quantized linear carries two scalar scale tensors
(`weight_scale_inv`, `activation_scale`) — quantization metadata, not parameters.

```
Mistral3ForConditionalGeneration
├── model.language_model
│   ├── embed_tokens                                   [V, d_model] = [131072, 12288]   bf16
│   ├── layers[0..87]                                  (Ministral3DecoderLayer)
│   │   ├── input_layernorm        (RMSNorm, no bias)  [12288]
│   │   ├── self_attn                                  (no biases, no QK-norm)
│   │   │   ├── q_proj                                 [H·d_head,  d_model] = [12288, 12288]  fp8
│   │   │   ├── k_proj                                 [H_kv·d_head, d_model] = [1024, 12288] fp8
│   │   │   ├── v_proj                                 [1024, 12288]                          fp8
│   │   │   └── o_proj                                 [d_model, H·d_head] = [12288, 12288]   fp8
│   │   ├── post_attention_layernorm                   [12288]
│   │   └── mlp                                        (SwiGLU)
│   │       ├── gate_proj                              [d_ff, d_model] = [28672, 12288]       fp8
│   │       ├── up_proj                                [28672, 12288]                         fp8
│   │       └── down_proj                              [d_model, d_ff] = [12288, 28672]       fp8
│   └── norm                       (final RMSNorm)     [12288]
├── model.vision_tower             (PixtralVisionModel, all bf16, no biases)
│   ├── patch_conv                 (Conv2d 14×14 s14)  [1664, 3, 14, 14]
│   ├── ln_pre                     (RMSNorm)           [1664]
│   └── transformer.layers[0..47]
│       ├── attention_norm         (RMSNorm)           [1664]
│       ├── attention.{q,k,v,o}_proj                   [1664, 1664] each (16 heads × 104, 2-D RoPE)
│       ├── ffn_norm               (RMSNorm)           [1664]
│       └── feed_forward.{gate,up}_proj / down_proj    [8192, 1664] ×2 / [1664, 8192] (SiLU-gated)
├── model.multi_modal_projector    (bf16, no biases)
│   ├── norm                       (RMSNorm over vision dim) [1664]
│   ├── patch_merger.merging_layer (2×2 unfold → linear) [1664, 6656]   (6656 = 1664·2²)
│   ├── linear_1                                       [12288, 1664]  → GELU
│   └── linear_2                                       [12288, 12288]
└── lm_head                        (untied, bf16)      [V, d_model] = [131072, 12288]
```

### Parameter count per module family (formula → count)

| Family | Formula | Substitution | Count |
|---|---|---|---:|
| embed_tokens | `V·d_model` | 131072·12288 | 1,610,612,736 |
| attention ×88 | `L·(2·d²  + 2·d·H_kv·d_head)` | 88·(2·12288² + 2·12288·1024) = 88·327,155,712 | 28,789,702,656 |
| MLP ×88 | `L·3·d·d_ff` | 88·3·12288·28672 = 88·1,056,964,608 | 93,012,885,504 |
| decoder norms | `L·2·d + d` | 88·24576 + 12288 | 2,174,976 |
| lm_head | `V·d_model` | 131072·12288 | 1,610,612,736 |
| vision tower | `48·(4·1664² + 3·1664·8192 + 2·1664) + 1664·3·14² + 1664` | 48·51,973,376 + 978,432 + 1,664 | 2,495,702,144 |
| projector | `1664 + 1664²·4 + 1664·12288 + 12288²` | 1,664 + 11,075,584 + 20,447,232 + 150,994,944 | 182,519,424 |
| **Total** | | | **127,704,210,176** |

**Closure: 127,704,210,176 = 127.70B ≈ 128B.** ✓ (The marketing number rounds up ~0.23%.)
Per decoder layer: 327,155,712 attn + 1,056,964,608 MLP + 24,576 norms = 1,384,144,896;
×88 = 121,804,750,848 — the decoder trunk is 95.4% of the model. Verified against the shard
headers tensor-by-tensor: the sum over all 2,465 tensors (minus the 1,232 scalar quant scales)
matches exactly.

---

## 3. Per-op analysis — one decoder layer

Reference workload: `B = 1`; **prefill** column at `S = 8192`; **decode** column for 1 token at
context 32768. FLOPs use the `2mnk` matmul convention; attention rows are the *naive* full causal
square (a causal-aware kernel does ~half the score FLOPs; FlashAttention never materializes
`[H, S, S]` in HBM — both noted below). Bytes = weights + activations in/out + KV traffic at
**bf16** (2 B/elem); shipped fp8 weights halve the weight-byte terms. AI = FLOPs / bytes.

| Op | Weight shape | Input → output shape (prefill) | FLOPs prefill | FLOPs decode | Bytes prefill | Bytes decode | AI pre | AI dec |
|---|---|---|---|---|---|---|---:|---:|
| input_layernorm | `[d]` = [12288] | `[B,S,d]` → same = [1,8192,12288] | 0.40 GF | 49 kF | 384 MiB | 72 KiB | ~1 | ~0.7 |
| q_proj | `[H·d_h, d]` = [12288,12288] | `[B,S,d]` → `[B,S,H·d_h]` | 2.474 TF | 302 MF | 672 MiB | 288 MiB | 3511 | 1.0 |
| k_proj | `[H_kv·d_h, d]` = [1024,12288] | `[B,S,d]` → `[B,S,1024]` | 206 GF | 25 MF | 232 MiB | 24 MiB | 847 | 1.0 |
| v_proj | [1024,12288] | `[B,S,d]` → `[B,S,1024]` | 206 GF | 25 MF | 232 MiB | 24 MiB | 847 | 1.0 |
| RoPE(q,k) | cos/sin (yarn ×1.41589) | q,k in place | 0.33 GF | 40 kF | 416 MiB | 52 KiB | ~1 | ~0.8 |
| KV-cache append | — | `[B,S,H_kv,d_h]`×2 → cache | 0 | 0 | 64 MiB (write) | 8 KiB | 0 | 0 |
| QKᵀ | — | `[B,H,S,d_h]·[B,H_kv,d_h,S]` → `[B,H,S,S]` | 1.649 TF | 805 MF | 12.2 GiB | 70 MiB | 126 | 11.0 |
| softmax | — | `[B,H,S,S]` → same | 32 GF | 16 MF | 24 GiB | 12 MiB | ~1 | ~1.2 |
| attn·V | — | `[B,H,S,S]·[B,H_kv,S,d_h]` → `[B,H,S,d_h]` | 1.649 TF | 805 MF | 12.2 GiB | 70 MiB | 126 | 11.0 |
| o_proj | `[d, H·d_h]` = [12288,12288] | `[B,S,H·d_h]` → `[B,S,d]` | 2.474 TF | 302 MF | 672 MiB | 288 MiB | 3511 | 1.0 |
| residual add | — | 2×`[B,S,d]` → `[B,S,d]` | 0.10 GF | 12 kF | 576 MiB | 72 KiB | ~0.2 | ~0.2 |
| post_attention_layernorm | [12288] | `[B,S,d]` → same | 0.40 GF | 49 kF | 384 MiB | 72 KiB | ~1 | ~0.7 |
| gate_proj | `[d_ff, d]` = [28672,12288] | `[B,S,d]` → `[B,S,d_ff]` | 5.772 TF | 705 MF | 1.28 GiB | 672 MiB | 4196 | 1.0 |
| up_proj | [28672,12288] | `[B,S,d]` → `[B,S,d_ff]` | 5.772 TF | 705 MF | 1.28 GiB | 672 MiB | 4196 | 1.0 |
| SiLU ⊙ | — | 2×`[B,S,d_ff]` → `[B,S,d_ff]` | 1.2 GF | 143 kF | 1.31 GiB | 168 KiB | ~1 | ~0.8 |
| down_proj | `[d, d_ff]` = [12288,28672] | `[B,S,d_ff]` → `[B,S,d]` | 5.772 TF | 705 MF | 1.28 GiB | 672 MiB | 4196 | 1.0 |
| residual add | — | 2×`[B,S,d]` → `[B,S,d]` | 0.10 GF | 12 kF | 576 MiB | 72 KiB | ~0.2 | ~0.2 |

**Layer totals:** prefill **26.01 TF** (24.33 TF causal-aware), decode **4.40 GF**.
**Model totals** (×88 + lm_head `2·S·d·V`): prefill 8192 tokens ≈ **2315 TF**
(cross-check: dense-approx `2·N_LM·S = 2·125.0e9·8192 ≈ 2048 TF` + 264 TF attention-quadratic ✓);
decode ≈ **390 GF/token** at 32k (≈ `2·N_LM` = 250 GF weight term + 141 GF attention term).

### GQA specifics

`G = H / H_kv = 96 / 8 = 12` query heads share each KV head. That's why
`k_proj`/`v_proj` are `[1024, 12288]` — 12288/1024 = **12× narrower** than `q_proj` — and the KV
cache is **12× smaller than MHA** (96/8). Against MQA (`H_kv = 1`) it is 8× larger.
`repeat_kv` broadcasts each KV head to its 12 queries at attention time
(`Ministral3Attention`, `num_key_value_groups = 12`).

### KV cache

Per token per layer: `2 · H_kv · d_head · bytes = 2·8·128·2 = 4096 B` (bf16; 2048 B fp8).
Full model: `× L=88` → **352 KiB/token bf16, 176 KiB fp8**. At `B = 1`:

| Context | bf16 | fp8 | vs shipped weights (124.4 GiB) |
|---|---|---|---|
| 8,192 | 2.75 GiB | 1.38 GiB | 2.2% |
| 32,768 | 11.0 GiB | 5.5 GiB | 8.8% |
| 131,072 | 44.0 GiB | 22.0 GiB | 35% |
| 262,144 | **88.0 GiB** | **44.0 GiB** | 71% |

KV scales with `B`: at `B = 8`, a full 256k bf16 cache is 704 GiB — more than any weight format.
Without GQA (MHA, `H_kv = 96`), 256k bf16 KV would be 1056 GiB.

### Memory footprint (weights + reference-workload KV)

| Format | Weights | + KV bf16 @32k | Fits |
|---|---|---|---|
| bf16 everything | 2·127.704e9 = 255.4 GB (237.9 GiB) | 248.9 GiB | 4×H100-80 (TP4), comfortably 8×H100 (model card recommends TP8) |
| fp8 as shipped (LM fp8, vision/proj/head bf16) | 133.6 GB (124.4 GiB) | 135.4 GiB | 2×H100-80 (tight, no headroom at long ctx), 1×B200-192 up to ~128k bf16 KV |
| int4 LM + bf16 rest (hypothetical) | ≈63.2 GiB + scales | 74.2 GiB | 1×H100-80 barely at short ctx; 1×H200-141 to 256k fp8 KV |

### Prefill vs decode (roofline)

On an H100-SXM-class roofline (990 TFLOP/s bf16, 3.35 TB/s HBM ⇒ ridge ≈ 295 FLOP/B):

- **Prefill** — all six weight GEMMs sit at AI 847–4196, far right of the ridge: **compute-bound**.
  Naive attention (AI ≈ 126) would be memory-bound because it writes/reads the 12.2 GiB
  `[H,S,S]` score tensor; FlashAttention fuses QKᵀ→softmax→·V so scores never leave SRAM,
  pushing effective AI well past the ridge — prefill attention is compute-bound in practice.
- **Decode** — every weight GEMM collapses to GEMV: AI ≈ 1.0 (read 150–352 MB of weights to do
  2 FLOPs/param). Attention reads the whole KV cache per token: AI ≈ 11. Everything is far left
  of the ridge: **decode is memory-bandwidth-bound**, and the floor is
  `weights-bytes + KV-bytes per token ≈ 133.6 GB + 5.8 GB ≈ 139 GB/token-step` (fp8 weights,
  32k bf16 KV) ⇒ ~24 ms/token minimum on one H100's 3.35 TB/s if unsharded (hence TP8: ~3 ms).
  Batching raises GEMM AI ∝ B; KV reads do not amortize across a batch.

### Speculative decoding (released draft)

`mistralai/Mistral-Medium-3.5-128B-EAGLE` is an EAGLE draft head for vLLM/SGLang: its
`params.json` is **the identical trunk config with `n_layers: 2`** — same `dim` 12288, 96 Q / 8 KV
heads, `head_dim` 128, `hidden_dim` 28672, vocab 131072, fp8, no vision encoder. Two decoder
layers ≈ 2·1.384e9 ≈ 2.77B trunk params (< 2.3% of the target). It plugs in at the dataflow seam
between the final norm and `lm_head`: EAGLE conditions on the target's last hidden state
`[B, 1, 12288]` plus sampled token embeddings, drafts several tokens with its 2 layers, and the
full 88-layer model verifies them in a single (cheap, prefill-like) batched forward — converting
memory-bound decode steps into compute-dense verification steps.

---

## Flagged unknowns (not derivable from config/checkpoint/card)

- Training data, token count, compute, and post-training recipe (the card documents reasoning
  modes `none`/`high` via chat template + system prompt, but not how they were trained).
- Vision encoder pretraining details beyond "trained from scratch, variable image sizes/aspect ratios".
- Whether 4096 (`original_max_position_embeddings`) was the true pretrain window or an
  intermediate stage (Ministral-3 report describes 16k→262k for the small models; not stated for
  Medium 3.5).
- Exact attention kernel used in production serving (config carries no implementation hint;
  `sliding_window: null` and no attention sinks in config).
- EAGLE head's exact fusion/embedding-sharing details (its repo ships `consolidated.safetensors`
  + `params.json` only; not inspected tensor-by-tensor here).

## Sources

- `mistralai/Mistral-Medium-3.5-128B` @ `main`: `config.json`, `params.json`,
  `model.safetensors.index.json` (`metadata.total_size = 133,605,834,656`), safetensors shard
  headers (all tensor shapes/dtypes), `README.md` model card, `LICENSE`.
- Config fix: commit `c4be198` "Fixing Transformers config", 2026-05-01 (`mscale_all_dim` 1.0 → 0.0).
- Reference implementation: `transformers` v5.12.1 — `models/ministral3/modeling_ministral3.py`
  (attention/MLP/norm order, `repeat_kv`, llama-4 scaling hook), `models/mistral3/modeling_mistral3.py`
  (projector, patch merger), `models/pixtral/modeling_pixtral.py` (vision tower, 2-D RoPE),
  `modeling_rope_utils.py::_compute_yarn_parameters` (YaRN + mscale semantics).
- `mistralai/Mistral-Medium-3.5-128B-EAGLE`: `params.json`.

*Analysis date: 2026-08-19. Config state: post-`c4be198` (2026-05-01).*
