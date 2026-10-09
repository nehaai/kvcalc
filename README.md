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

## Limitations

- Assumes every sequence uses the full context length. With paged KV allocation (vLLM, SGLang), real concurrency on shorter requests is higher; this is the worst-case floor.
- Treats multiple GPUs as one pool. Tensor parallelism adds per-card overhead, so multi-GPU numbers are slightly optimistic.
- Ignores small terms such as attention biases (e.g. Qwen QKV bias, ~0.002% of params).
- MLA numbers assume the serving engine stores the compressed latent, as vLLM and SGLang do.
