"""Prefix-reuse FLOPs for Qwen3.5 (hybrid Gated DeltaNet + gated full attention).

Per call (CLM, arXiv 2609.37725, Eq. 8-9), with P prompt tokens, R of them served from the prefix
cache, U = P - R and G generated tokens:

    F = C_token * (U + G) + C_head * G + C_attn * (P^2 - R^2 + 2 G P + G^2) / 2

C_token counts 2 FLOPs per multiply-add over the per-token linear layers (MLP, full-attention
q/k/v/o, DeltaNet in/out projections), without embeddings or the output head. C_head is the output
head, charged on generated tokens only (CLM omits it; it matters for the small models). C_attn is
4 * L_full * h_q * d_h per query-key pair in the full-attention layers. Omitted, as in CLM: norms,
activations, softmax, the DeltaNet recurrent state update. With R = 0 the formula gives the
no-cache upper bound.
"""

from dataclasses import dataclass

# Text-model dimensions from each model's config.json (Qwen/Qwen3.5-<size>): hidden_size,
# intermediate_size, num_hidden_layers, full_attention_interval, num_attention_heads,
# num_key_value_heads, head_dim, linear_num_key_heads, linear_key_head_dim,
# linear_num_value_heads, linear_value_head_dim.
FIELDS = ("hidden", "inter", "layers", "full_every", "hq", "hkv", "dh", "lk", "lkd", "lv", "lvd")
QWEN35 = {
    "0.8B": (1024, 3584, 24, 4, 8, 2, 256, 16, 128, 16, 128),
    "2B": (2048, 6144, 24, 4, 8, 2, 256, 16, 128, 16, 128),
    "4B": (2560, 9216, 32, 4, 16, 4, 256, 16, 128, 32, 128),
    "9B": (4096, 12288, 32, 4, 16, 4, 256, 16, 128, 32, 128),
    "27B": (5120, 17408, 64, 4, 24, 4, 256, 16, 128, 48, 128),
}
VOCAB = 248_320


@dataclass(frozen=True)
class FlopConstants:
    c_token: float
    c_head: float
    c_attn: float


def constants(hidden, inter, layers, full_every, hq, hkv, dh, lk, lkd, lv, lvd, vocab=VOCAB):
    n_full = layers // full_every
    # Gated attention: q_proj also produces the output gate (2 * hq * dh outputs).
    full = hidden * hq * dh * 2 + 2 * hidden * hkv * dh + hq * dh * hidden
    # DeltaNet: q, k (key dims), v, z (value dims), b and a gates, output projection.
    linear = hidden * (2 * lk * lkd + 2 * lv * lvd) + hidden * 2 * lv + lv * lvd * hidden
    params = layers * 3 * hidden * inter + n_full * full + (layers - n_full) * linear
    return FlopConstants(2.0 * params, 2.0 * hidden * vocab, 4.0 * n_full * hq * dh)


CONSTANTS = {size: constants(**dict(zip(FIELDS, dims))) for size, dims in QWEN35.items()}


def size_of(model: str) -> str:
    """'Qwen/Qwen3.5-27B' or 'Qwen/Qwen3.5-27B-FP8' -> '27B'."""
    return model.rsplit("Qwen3.5-", 1)[-1].removesuffix("-FP8")


def call_flops(k: FlopConstants, prompt: int, cached: int, completion: int) -> float:
    p, r, g = prompt, cached, completion
    return (
        k.c_token * (p - r + g) + k.c_head * g + k.c_attn * (p * p - r * r + 2 * g * p + g * g) / 2
    )
