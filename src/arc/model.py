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
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    else:
        # transformers renombró torch_dtype -> dtype; las versiones nuevas IGNORAN
        # torch_dtype y cargan en fp32 (un 3B en fp32 = 12.4GB: OOM en T4).
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name, dtype=getattr(torch, dtype), **kwargs)
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=getattr(torch, dtype), **kwargs)
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


def _parse_grid(text: str, sep: str = "") -> Optional[Grid]:
    """Parsea texto a grid; None si no es un grid rectangular no vacío."""
    from .grid import text_to_grid

    try:
        grid = text_to_grid(text, sep)
        if grid and all(len(r) == len(grid[0]) and r for r in grid):
            return grid
    except Exception:
        pass
    return None


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
    return _parse_grid(text, sep)


def generate_grid_candidates(
    model,
    tokenizer,
    prompt: str,
    sep: str = "",
    n_candidates: int = 3,
    temperature: float = 0.7,
    max_new_tokens: int = 2048,
    constrained: bool = True,
) -> List[Grid]:
    """Genera candidatos diversos para un prompt: 1 greedy + (n-1) por sampling.

    Devuelve grids únicos (dedup por contenido), en orden greedy-primero. Puede
    devolver menos de `n_candidates` si hay colisiones o salidas no parseables.
    """
    import torch
    from transformers import LogitsProcessorList

    from .grid import grid_to_text

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]
    gen_common = dict(max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id)
    if constrained:
        gen_common["logits_processor"] = LogitsProcessorList(
            [make_grid_logits_processor(tokenizer)]
        )

    cands: List[Grid] = []
    seen = set()

    def _collect(out) -> None:
        texts = tokenizer.batch_decode(out[:, prompt_len:], skip_special_tokens=True)
        for t in texts:
            g = _parse_grid(t, sep)
            if g is None:
                continue
            key = grid_to_text(g)
            if key not in seen:
                seen.add(key)
                cands.append(g)

    with torch.no_grad():
        _collect(model.generate(**inputs, do_sample=False, **gen_common))
        if n_candidates > 1:
            _collect(model.generate(
                **inputs, do_sample=True, temperature=temperature,
                num_return_sequences=n_candidates - 1, **gen_common,
            ))
    return cands


def score_completions(
    model,
    tokenizer,
    prompt: str,
    completions: List[str],
    batch_size: int = 1,  # firma retro-compatible; se procesa de a 1 (ver nota de memoria)
) -> List[float]:
    """log P(completion | prompt) para cada completion (suma de log-probs + EOS).

    Forward passes sin generación: es la señal de *ranking* de candidatos (técnica de
    los top ARC 2024/2025: puntuar bajo vistas augmentadas en lugar de solo contar votos).

    Nota de memoria: los logits completos son [L, vocab≈152k] — a L~3k eso son ~0.9GB en
    bf16 y el doble en float32. Por eso se procesa UNA secuencia a la vez y el log_softmax
    float32 se aplica SOLO a las posiciones de la completion (no al prompt); de lo
    contrario un batch revienta una T4 de 14GB (OOM observado en Kaggle).
    """
    import torch

    p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    eos = tokenizer.eos_token or ""
    scores: List[float] = []

    for comp in completions:
        c_ids = tokenizer(comp + eos, add_special_tokens=False)["input_ids"]
        ids = torch.tensor([p_ids + c_ids], device=model.device)
        with torch.no_grad():
            logits = model(input_ids=ids).logits
        # logprob del token t sale de la posición t-1: solo necesitamos las posiciones
        # que predicen la completion -> [len(c_ids), vocab]
        lo, hi = len(p_ids) - 1, len(p_ids) + len(c_ids) - 1
        comp_logits = logits[0, lo:hi].float()
        del logits
        logprobs = torch.log_softmax(comp_logits, dim=-1)
        targets = torch.tensor(c_ids, device=model.device)
        scores.append(logprobs.gather(-1, targets.unsqueeze(-1)).sum().item())
        del comp_logits, logprobs
    return scores
