# kvcalc

A small CLI that tells you how many concurrent sequences an LLM can serve on a GPU, computed from the model's Hugging Face `config.json`.

It reports:

- **Weight memory**: parameter count × bytes per parameter
- **KV cache per token**: what every processed token costs to keep around
- **KV cache per sequence**: per-token cost × context length
- **Max concurrent sequences**: `(GPU memory × utilization − weights) / KV per sequence`

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python kvcalc.py NousResearch/Meta-Llama-3-8B --gpu-gib 80
```

```
Model:            NousResearch/Meta-Llama-3-8B
Params:           8.03B
Weights:          14.96 GiB  (fp16)
KV per token:     128.00 KiB  (fp16)
KV per sequence:  1.00 GiB  (8192 tokens)
Max concurrent:   57  (on 80.0 GiB @ 90%)
```

| Flag | Default | Meaning |
|---|---|---|
| `model_id` | required | Hugging Face model id |
| `--gpu-gib` | required | GPU memory in GiB |
| `--dtype` | `fp16` | Weight dtype: `fp32`, `fp16`, `bf16`, `fp8`, `int8` |
| `--kv-dtype` | same as `--dtype` | KV cache dtype (e.g. bf16 weights + fp8 KV, like vLLM's `--kv-cache-dtype fp8`) |
| `--context` | model's `max_position_embeddings` | Tokens per sequence |
| `--utilization` | `0.9` | Fraction of GPU memory usable, leaving room for activations and CUDA overhead (matches vLLM's default) |

Gated models (e.g. Mixtral) need `huggingface-cli login` and license acceptance on the model page.

## Supported architectures

| Model | Attention | KV / token (fp16) | Max seqs (80 GiB, 8k ctx) |
|---|---|---|---|
| Llama 3 8B | GQA (8 KV heads) | 128 KiB | 57 |
| Mixtral 8x7B | GQA + MoE MLP | 128 KiB | 0 (87 GiB of weights doesn't fit) |
| DeepSeek-V2-Lite | MLA | 30.38 KiB | 180 |

**Standard attention (MHA / GQA):**

```
KV bytes per token = 2 × layers × kv_heads × head_dim × bytes_per_element
```

Sanity check: Llama 3 8B in fp16 gives 2 × 32 × 8 × 128 × 2 = 128 KiB per token, so 1 GiB per 8k-token sequence.

**Multi-head Latent Attention (DeepSeek V2/V3):** caches one compressed latent plus one shared RoPE key per layer, instead of K and V per head:

```
KV bytes per token = layers × (kv_lora_rank + qk_rope_head_dim) × bytes_per_element
```

**Mixture of experts:** weight memory counts every expert, because all of them must be resident in VRAM. Only the KV cache follows attention, so MoE doesn't change it.

## Validation against llama.cpp

Compared kvcalc with the KV cache size `llama-server` logs at startup.

Setup: `bartowski/Meta-Llama-3-8B-Instruct-GGUF:Q4_K_M`, `-c 8192`, Apple M5 (Metal), log level `-lv 4`. kvcalc used `NousResearch/Meta-Llama-3-8B`, which has the same architecture config.

```bash
llama-server -hf bartowski/Meta-Llama-3-8B-Instruct-GGUF:Q4_K_M -c 8192 -np 1 -lv 4 2>&1 | tee run1.log
grep -E "llama_kv_cache: size|memory_breakdown" run1.log
```

Startup log (run 1):

```
llama_kv_cache: size = 1024.00 MiB (  8192 cells,  32 layers,  1/1 seqs), K (f16):  512.00 MiB, V (f16):  512.00 MiB
memory breakdown [MiB] | self   model   context   compute
  - MTL0 (Apple M5)    | 5551 = 4403 +    1024 +     124
```

| Quantity | kvcalc | llama.cpp | Result |
|---|---|---|---|
| KV cache, 8192 tokens, f16 | 1.00 GiB | 1024.00 MiB | Exact match |
| K / V split | 512 + 512 MiB | 512 + 512 MiB | Exact match |
| Layers in cache | 32 | 32 | Exact match |
| Weights | 14.96 GiB (fp16) | 4403 MiB (Q4_K_M) | Expected gap, see below |
| Compute buffer | not modeled | 124 MiB | Expected gap, see below |

**The KV formula is exact.** `2 × 32 layers × 8 KV heads × 128 head_dim × 2 bytes × 8192 tokens` gives exactly 1024 MiB, the number llama.cpp allocates.

### Explaining the gaps

**1. Weight quantization does not quantize the KV cache.** The weights are 4-bit (Q4_K_M), but the log shows `K (f16)` and `V (f16)`. llama.cpp keeps the cache in f16 unless you pass `-ctk` and `-ctv`. To model this in kvcalc, set the two dtypes separately with `--kv-dtype fp16`.

**2. kvcalc can't predict Q4 weight size.** 4403 MiB for 8.03B parameters works out to about 4.6 bits per weight, not 4. K-quant formats store per-block scales and minimums next to the 4-bit values, and some tensors (such as the embeddings and the output layer) are kept at higher precision. kvcalc only supports uniform dtypes (`fp32`, `fp16`, `bf16`, `fp8`, `int8`), so it has no way to represent a mixed-precision GGUF. Read weight memory for a GGUF from its file size instead.

**3. The compute buffer is outside the formula.** llama.cpp reserves 124 MiB for activations and scratch tensors (for a 512-token micro-batch). This is the overhead kvcalc's `--utilization 0.9` reserve exists to absorb. It isn't KV, so it isn't in the KV number.

**4. `-c` is a total budget in llama.cpp, but per-sequence in kvcalc.** Run 2 (`-np 4`) logged:

```
llama_kv_cache: size = 1024.00 MiB (  2048 cells,  32 layers,  4/4 seqs), K (f16):  512.00 MiB, V (f16):  512.00 MiB
```

llama.cpp divided the 8192 tokens among the four slots instead of giving each slot 8192 tokens. With `kv_unified = false`, each slot gets its own 2048-cell section of the cache, and the log reports that per-slot cell count. The total is unchanged: 4 × 2048 cells = 1024 MiB, the same as run 1. Adding slots split the cache; it didn't grow it. kvcalc's `--context` is the length *each* sequence gets, so the llama.cpp setup that matches kvcalc's "N sequences × 8192 tokens" is `-np N -c (N × 8192)`.

**5. Quantized KV types aren't exactly 1 byte per element.** Run 3 (`-ctk q8_0 -ctv q8_0 -fa on`) logged:

```
llama_kv_cache: size =  544.00 MiB (  8192 cells,  32 layers,  1/1 seqs), K (q8_0):  272.00 MiB, V (q8_0):  272.00 MiB
```

kvcalc's `--kv-dtype fp8` treats 8-bit KV as exactly 1 byte, which predicts 512 MiB. llama.cpp's `q8_0` stores a 2-byte fp16 scale with every block of 32 int8 values, so each value costs 34 / 32 = 1.0625 bytes. 512 MiB × 1.0625 = 544 MiB, the measured value. kvcalc underestimates q8_0 KV by 6.25%.

### All three runs

| Run | Flags | kvcalc | llama.cpp | Gap |
|---|---|---|---|---|
| 1 | `-np 1` | 1024 MiB | 1024.00 MiB | 0 |
| 2 | `-np 4` | 1024 MiB total (4 × 2048 tokens) | 1024.00 MiB (2048 cells × 4 seqs) | 0, once `-c` is read as a total |
| 3 | `-ctk q8_0 -ctv q8_0` | 512 MiB (`--kv-dtype fp8`) | 544.00 MiB | +6.25%, from block scales |

### Tip: finding the KV line

Recent `llama-server` builds hide model-loading details at the default log level (3). Pass `-lv 4` to show the `llama_kv_cache: size = ...` line and the memory breakdown table.

## Limitations

- Assumes every sequence uses the full context length. With paged KV allocation (vLLM, SGLang), real concurrency on shorter requests is higher; this is the worst-case floor.
- Treats multiple GPUs as one pool. Tensor parallelism adds per-card overhead, so multi-GPU numbers are slightly optimistic.
- Ignores small terms such as attention biases (e.g. Qwen QKV bias, ~0.002% of params).
- MLA numbers assume the serving engine stores the compressed latent, as vLLM and SGLang do.