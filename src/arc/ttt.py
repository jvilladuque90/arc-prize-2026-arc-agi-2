"""Test-Time Training (TTT) por tarea.

En inferencia, antes de predecir CADA tarea de test, se adapta el adaptador LoRA con unos
pocos pasos de fine-tune sobre los propios pares `train` de la tarea (augmentados,
leave-one-out). Luego se predice (idealmente con AIRV) y se RESTAURA el adaptador para que
la siguiente tarea empiece desde el mismo punto. Fue la técnica decisiva de los ganadores.

Flujo:
    ttt = TTT(model, tok, steps=24, lr=1e-4, n_aug=4)   # snapshot del LoRA actual
    results = predict_with_ttt(task, model, tok, ttt, n_aug_infer=8)

`torch` se importa de forma perezosa: el módulo se puede importar sin GPU/torch.
"""
from __future__ import annotations

import random
from typing import List, Optional, Tuple

from .augment import augment_task, color_permutation
from .data import Task
from .grid import format_task_prompt, grid_to_text


def task_ttt_samples(task: Task, n_aug: int = 4, sep: str = "",
                     max_cells: int = 900) -> List[Tuple[str, str]]:
    """Pares (prompt, completion) a partir de los `train` de la tarea (leave-one-out
    sobre cada par, con `n_aug` vistas augmentadas D4+color). NO usa los `test`
    (cuyas soluciones se desconocen en inferencia)."""
    specs = [(0, list(range(10)))]
    for a in range(1, n_aug):
        specs.append((a % 8, color_permutation(seed=(hash((task.task_id, a)) & 0xFFFFFFFF),
                                               keep_zero=True)))
    samples: List[Tuple[str, str]] = []
    for d4, cmap in specs:
        at = augment_task(task, d4, cmap)
        n = len(at.train)
        for i in range(n):
            ctx = at.train[:i] + at.train[i + 1:]
            if not ctx:
                continue
            q = at.train[i]
            in_cells = len(q.input) * len(q.input[0])
            out_cells = len(q.output) * len(q.output[0])
            if max_cells and max(in_cells, out_cells) > max_cells:
                continue
            prompt = format_task_prompt(ctx, q.input, sep)
            completion = "\n" + grid_to_text(q.output, sep) + "\n"
            samples.append((prompt, completion))
    return samples


class TTT:
    """Adaptador test-time: guarda un snapshot de los pesos entrenables (LoRA) y los
    restaura entre tareas para evitar fuga de información de una tarea a otra."""

    def __init__(self, model, tokenizer, steps: int = 24, lr: float = 1e-4,
                 n_aug: int = 4, max_seq_len: int = 2048, batch_size: int = 1,
                 seed: int = 0):
        self.model = model
        self.tok = tokenizer
        self.steps = steps
        self.lr = lr
        self.n_aug = n_aug
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.seed = seed
        # snapshot de los parámetros entrenables (el adaptador LoRA)
        self._snap = {n: p.detach().clone()
                      for n, p in model.named_parameters() if p.requires_grad}

    def restore(self) -> None:
        import torch

        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if n in self._snap:
                    p.copy_(self._snap[n])

    def _encode(self, prompt: str, completion: str):
        eos = self.tok.eos_token or ""
        p_ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        c_ids = self.tok(completion + eos, add_special_tokens=False)["input_ids"]
        ids = (p_ids + c_ids)[: self.max_seq_len]
        labels = ([-100] * len(p_ids) + c_ids)[: self.max_seq_len]
        return ids, labels

    def _batch(self, enc):
        import torch

        pad = self.tok.pad_token_id or 0
        maxlen = max(len(ids) for ids, _ in enc)
        input_ids, labels, attn = [], [], []
        for ids, lb in enc:
            n = maxlen - len(ids)
            input_ids.append(ids + [pad] * n)
            labels.append(lb + [-100] * n)
            attn.append([1] * len(ids) + [0] * n)
        dev = self.model.device
        return (torch.tensor(input_ids, device=dev),
                torch.tensor(labels, device=dev),
                torch.tensor(attn, device=dev))

    def adapt(self, task: Task) -> int:
        """Restaura el snapshot y adapta el LoRA a `task`. Devuelve nº de muestras usadas."""
        import torch

        self.restore()
        samples = task_ttt_samples(task, self.n_aug)
        if not samples:
            return 0

        opt = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad], lr=self.lr
        )
        rng = random.Random(self.seed)
        self.model.train()
        for _ in range(self.steps):
            batch = [samples[rng.randrange(len(samples))] for _ in range(self.batch_size)]
            input_ids, labels, attn = self._batch([self._encode(p, c) for p, c in batch])
            out = self.model(input_ids=input_ids, attention_mask=attn, labels=labels)
            out.loss.backward()
            opt.step()
            opt.zero_grad()
        self.model.eval()
        return len(samples)


def predict_with_ttt(task: Task, model, tokenizer, ttt: TTT,
                     n_aug_infer: int = 8, sep: str = "", **gen_kwargs) -> List[dict]:
    """Adapta (TTT) → predice con AIRV → restaura. Devuelve [{attempt_1, attempt_2}, ...]."""
    from .infer import predict_task
    from .model import generate_grid

    ttt.adapt(task)
    predict_fn = lambda prompt: generate_grid(model, tokenizer, prompt, sep=sep, **gen_kwargs)
    try:
        results = predict_task(task, predict_fn, n_aug=n_aug_infer, sep=sep)
    finally:
        ttt.restore()
    return results
