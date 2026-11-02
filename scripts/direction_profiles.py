"""GLM-4.7-Flash tensor mapping."""

PROFILES = ("glm47",)
GLM_KINDS = ("attn_out", "ffn_dense", "ffn_shexp", "ffn_exps", "token_embd")


def glm_targets(config, layers, kinds):
    n = config["num_hidden_layers"]
    dense = config["first_k_dense_replace"]
    experts = config["n_routed_experts"]
    freq = config.get("moe_layer_freq", 1)
    if n != 47 or config["hidden_size"] != 2048 or experts != 64:
        raise ValueError(
            "GLM-4.7-Flash profile requires 47 main layers, width 2048, 64 experts"
        )
    if not 0 <= dense <= n or not isinstance(freq, int) or freq < 1:
        raise ValueError("invalid dense/MoE config")
    if set(kinds) - set(GLM_KINDS):
        raise ValueError("unsupported GLM kind")
    selected = []
    for layer in layers:
        if not 0 <= layer < n:
            raise ValueError("GLM main layers must be in 0..46; MTP is excluded")
        moe = layer >= dense and layer % freq == 0
        for kind in dict.fromkeys(kinds):
            if kind == "token_embd":
                continue
            suffix = {
                "attn_out": ("attn_output", "self_attn.o_proj.weight"),
                "ffn_dense": ("ffn_down", "mlp.down_proj.weight"),
                "ffn_shexp": ("ffn_down_shexp", "mlp.shared_experts.down_proj.weight"),
                "ffn_exps": ("ffn_down_exps", None),
            }[kind]
            if (kind == "ffn_dense" and moe) or (
                kind in ("ffn_shexp", "ffn_exps") and not moe
            ):
                continue
            names = (
                [
                    f"model.layers.{layer}.mlp.experts.{e}.down_proj.weight"
                    for e in range(experts)
                ]
                if kind == "ffn_exps"
                else [f"model.layers.{layer}.{suffix[1]}"]
            )
            selected.append((layer, kind, f"blk.{layer}.{suffix[0]}.weight", names))
    if "token_embd" in kinds:
        selected.append(
            (None, "token_embd", "token_embd.weight", ["model.embed_tokens.weight"])
        )
    if not selected:
        raise ValueError("no GLM tensors selected for the configured dense/MoE layers")
    return selected


def resolve_index(targets, index):
    missing = [
        name for _, _, _, names in targets for name in names if name not in index
    ]
    if missing:
        raise ValueError(f"missing {len(missing)} required HF tensors: {missing[:10]}")
    return [
        dict(
            gguf=gg, kind=kind, layer=layer, hf=names, shards=[index[n] for n in names]
        )
        for layer, kind, gg, names in targets
    ]
