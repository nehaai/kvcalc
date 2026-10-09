import json
from huggingface_hub import hf_hub_download
import argparse

DTYPE_BYTES = {"fp32": 4, "fp16": 2, "bf16": 2, "fp8": 1, "int8": 1}


def load_config(model_id: str) -> dict:
    path = hf_hub_download(repo_id=model_id, filename="config.json")
    with open(path) as f:
        cfg = json.load(f)

    # Multimodal models (e.g. Llama 3.2 Vision) nest the LLM config here
    if "text_config" in cfg:
        cfg = cfg["text_config"]
    return cfg


def attn_dims(cfg: dict) -> tuple[int, int, int]:
    attn_heads = cfg["num_attention_heads"]
    kv_heads = cfg.get("num_key_value_heads", attn_heads)
    head_dim = cfg.get("head_dim") or cfg["hidden_size"] // attn_heads
    return attn_heads, kv_heads, head_dim


def kv_bytes_per_token(cfg: dict, bytes_per_elem: int) -> int:
    if is_mla(cfg):
        return mla_kv_bytes_per_token(cfg, bytes_per_elem)
    return standard_kv_bytes_per_token(cfg, bytes_per_elem)


def param_count(cfg: dict) -> int:
    if is_mla(cfg):
        return mla_param_count(cfg)
    return standard_param_count(cfg)


def is_mla(cfg: dict) -> bool:
    return bool(cfg.get("kv_lora_rank"))


def standard_kv_bytes_per_token(cfg: dict, bytes_per_elem: int) -> int:
    layers = cfg["num_hidden_layers"]
    attn_heads = cfg["num_attention_heads"]
    kv_heads = cfg.get("num_key_value_heads", attn_heads)
    head_dim = cfg.get("head_dim") or cfg["hidden_size"] // attn_heads
    return 2 * layers * kv_heads * head_dim * bytes_per_elem


def kv_bytes_per_seq(cfg: dict, bytes_per_elem: int, context_len: int) -> int:
    return kv_bytes_per_token(cfg, bytes_per_elem) * context_len


def standard_param_count(cfg: dict) -> int:
    h = cfg["hidden_size"]
    layers = cfg["num_hidden_layers"]
    attn_heads, kv_heads, head_dim = attn_dims(cfg)
    inter = cfg["intermediate_size"]
    vocab = cfg["vocab_size"]

    attn = h * (attn_heads * head_dim)        # Q
    attn += 2 * h * (kv_heads * head_dim)     # K, V
    attn += (attn_heads * head_dim) * h       # O
    experts = cfg.get("num_local_experts", 1)
    mlp = experts * 3 * h * inter
    router = h * experts if experts > 1 else 0
    norms = 2 * h
    per_layer = attn + mlp + router + norms

    embed = vocab * h
    lm_head = 0 if cfg.get("tie_word_embeddings", False) else vocab * h
    return embed + layers * per_layer + h + lm_head


def mla_kv_bytes_per_token(cfg: dict, bytes_per_elem: int) -> int:
    # one compressed latent + one RoPE key per layer; no 2x, no x heads
    per_layer = cfg["kv_lora_rank"] + cfg["qk_rope_head_dim"]
    return cfg["num_hidden_layers"] * per_layer * bytes_per_elem


def mla_param_count(cfg: dict) -> int:
    h = cfg["hidden_size"]
    heads = cfg["num_attention_heads"]
    kv_rank = cfg["kv_lora_rank"]
    q_rank = cfg.get("q_lora_rank")
    nope, rope = cfg["qk_nope_head_dim"], cfg["qk_rope_head_dim"]
    v_dim = cfg["v_head_dim"]
    q_dim = heads * (nope + rope)

    # Attention
    if q_rank:                                   # V2/V3 also compress Q
        attn = h * q_rank + q_rank + q_rank * q_dim
    else:                                        # V2-Lite: plain Q
        attn = h * q_dim
    attn += h * (kv_rank + rope)                 # compress -> latent + rope key
    attn += kv_rank                              # norm on the latent
    attn += kv_rank * heads * (nope + v_dim)     # decompress -> K, V
    attn += heads * v_dim * h                    # output projection

    # MLP: first layer dense, rest MoE (routed + shared experts)
    dense_mlp = 3 * h * cfg["intermediate_size"]
    expert = 3 * h * cfg.get("moe_intermediate_size", 0)
    n_routed = cfg.get("n_routed_experts") or 0
    n_shared = cfg.get("n_shared_experts") or 0
    moe_mlp = (n_routed + n_shared) * expert + n_routed * h   # + router

    total = 0
    for i in range(cfg["num_hidden_layers"]):
        moe_layer = (n_routed > 0
                     and i >= cfg.get("first_k_dense_replace", 0)
                     and i % cfg.get("moe_layer_freq", 1) == 0)
        total += attn + (moe_mlp if moe_layer else dense_mlp) + 2 * h

    vocab = cfg["vocab_size"]
    lm_head = 0 if cfg.get("tie_word_embeddings", False) else vocab * h
    return total + vocab * h + h + lm_head


def weight_bytes(cfg, bytes_per_elem) -> int:
    return param_count(cfg) * bytes_per_elem


def max_concurrent_seqs(cfg: dict, w_bpe: int, kv_bpe: int, context_len: int,
                        gpu_gib: float, utilization: float = 0.9) -> int:
    usable = gpu_gib * 1024**3 * utilization
    free = usable - weight_bytes(cfg, w_bpe)
    if free <= 0:
        return 0
    return int(free // kv_bytes_per_seq(cfg, kv_bpe, context_len))


def fmt(n: float) -> str:
    for unit in ["B", "KiB", "MiB", "GiB"]:
        if n < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} TiB"


def main():
    p = argparse.ArgumentParser(description="KV cache & GPU memory calculator")
    p.add_argument("model_id", help="Hugging Face model id")
    p.add_argument("--dtype", choices=DTYPE_BYTES, default="fp16")
    p.add_argument("--kv-dtype", choices=DTYPE_BYTES, default=None,
                help="KV cache dtype (default: same as --dtype)")
    p.add_argument("--context", type=int, default=None,
                   help="tokens per sequence (default: model's max)")
    p.add_argument("--gpu-gib", type=float, required=True)
    p.add_argument("--utilization", type=float, default=0.9)
    args = p.parse_args()

    cfg = load_config(args.model_id)
    w_bpe = DTYPE_BYTES[args.dtype]
    kv_bpe = DTYPE_BYTES[args.kv_dtype or args.dtype]
    ctx = args.context or cfg["max_position_embeddings"]

    print(f"Model:            {args.model_id}")
    print(f"Params:           {param_count(cfg) / 1e9:.2f}B")
    print(f"Weights:          {fmt(weight_bytes(cfg, w_bpe))}  ({args.dtype})")
    print(f"KV per token:     {fmt(kv_bytes_per_token(cfg, kv_bpe))}  ({args.kv_dtype or args.dtype})")
    print(f"KV per sequence:  {fmt(kv_bytes_per_seq(cfg, kv_bpe, ctx))}  ({ctx} tokens)")
    print(f"Max concurrent:   {max_concurrent_seqs(cfg, w_bpe, kv_bpe, ctx, args.gpu_gib, args.utilization)}"
          f"  (on {args.gpu_gib} GiB @ {args.utilization:.0%})")


if __name__ == "__main__":
    main()


# if __name__ == "__main__":
#     cfg = load_config("NousResearch/Meta-Llama-3-8B")
    
#     for key in ["num_hidden_layers", "num_attention_heads",
#                 "num_key_value_heads", "hidden_size", "head_dim"]:
#         print(f"{key}: {cfg.get(key)}")

#     b = kv_bytes_per_token(cfg, 2)   # fp16 = 2 bytes
#     print("b ",b, b / 1024, "KiB")

#     s = kv_bytes_per_seq(cfg, 2, 8192)
#     print("s ",s / 1024**3, "GiB")

#     w = weight_bytes(cfg, 2)
#     print("w ", param_count(cfg), w / 1024**3, "GiB")

#     print("max", max_concurrent_seqs(cfg, 2, 8192, 80))