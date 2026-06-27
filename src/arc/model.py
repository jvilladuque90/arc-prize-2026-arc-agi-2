"""Carga de Qwen2.5 + LoRA y generación restringida a grids.

Las dependencias pesadas (torch/transformers/peft) se importan dentro de las funciones
para que el resto del paquete pueda importarse sin tenerlas instaladas (p.ej. dev local).

Uso típico (en notebook con GPU):
    from arc.model import load_model, build_lora, generate_grid
    model, tok = load_model("Qwen/Qwen2.5-0.5B-Instruct", load_4bit=True)
    model = build_lora(model)
    grid = generate_grid(model, tok, prompt)
"""
from __future__ import annotations

from typing import List, Optional

Grid = List[List[int]]

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def load_model(
    model_name: str = DEFAULT_MODEL,
    load_4bit: bool = False,
    dtype: str = "bfloat16",
    device_map: str = "auto",
):
    """Carga modelo + tokenizer. `load_4bit` usa bitsandbytes (no en Windows)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    kwargs = {"device_map": device_map}
    if load_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=getattr(torch, dtype),
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    else:
        kwargs["torch_dtype"] = getattr(torch, dtype)

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    return model, tok


def build_lora(
    model,
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
):
    """Envuelve el modelo con un adaptador LoRA entrenable (PEFT)."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False):
        model = prepare_model_for_kbit_training(model)

    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"]

    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    return get_peft_model(model, cfg)


def allowed_grid_token_ids(tokenizer) -> List[int]:
    """IDs de tokens válidos para un grid: dígitos 0-9, salto de línea y EOS."""
    ids = set()
    for ch in list("0123456789") + ["\n"]:
        for t in tokenizer.encode(ch, add_special_tokens=False):
            ids.add(t)
    if tokenizer.eos_token_id is not None:
        ids.add(tokenizer.eos_token_id)
    return sorted(ids)


def make_grid_logits_processor(tokenizer):
    """LogitsProcessor que enmascara todo lo que no sea dígito/newline/EOS.
    Reduce alucinaciones de formato durante la generación del grid."""
    import torch
    from transformers import LogitsProcessor

    allowed = set(allowed_grid_token_ids(tokenizer))

    class _GridProcessor(LogitsProcessor):
        def __call__(self, input_ids, scores):
            mask = torch.full_like(scores, float("-inf"))
            idx = torch.tensor(sorted(allowed), device=scores.device)
            mask[:, idx] = scores[:, idx]
            return mask

    return _GridProcessor()


def generate_grid(
    model,
    tokenizer,
    prompt: str,
    sep: str = "",
    max_new_tokens: int = 2048,
    constrained: bool = True,
    do_sample: bool = False,
    temperature: float = 1.0,
) -> Optional[Grid]:
    """Genera y parsea un grid a partir de un prompt. Devuelve None si no parsea."""
    import torch
    from transformers import LogitsProcessorList

    from .grid import text_to_grid

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature
    if constrained:
        gen_kwargs["logits_processor"] = LogitsProcessorList(
            [make_grid_logits_processor(tokenizer)]
        )

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    try:
        grid = text_to_grid(text, sep)
        # validación mínima: filas no vacías y rectangular
        if grid and all(len(r) == len(grid[0]) and r for r in grid):
            return grid
    except Exception:
        pass
    return None
