# Plan de open-source para elegibilidad de premios

> Regla oficial (arcprize.org/competitions/2026/arc-agi-2, verificada 2026-07-30):
> **"All code and methods must be open sourced to be eligible for prizes"** — y te eliminan de
> la competencia si no lo haces. Para el Grand Prize: *"All artifacts should be open sourced and
> attached to an official competition Solution Writeup within seven days of the competition's
> submission deadline"*. El writeup se evalúa en 6 criterios (0-5 cada uno).

## Política durante la competencia

**Mantener todo PRIVADO hasta el cierre.** Es legal y estratégico: nuestros injertos medidos
(cheap-first, etc.) serían copiables por el pelotón que corre el mismo notebook público. La
obligación de publicar aplica **tras el deadline** (ventana de 7 días), no antes.

## Checklist de publicación (ejecutar en los 7 días tras el deadline)

1. **Repo GitHub → público** (ya MIT, ver `LICENSE`). Contiene todo el código, notebooks y docs.
2. **Kernels de Kaggle → públicos** (`is_private: false` en `push_kernels.py` y re-push, o flip
   en la UI): la submission final y los kernels de los que depende (`barc-download` si se usa
   el modelo de inducción, `barc-datagen`, `barc-sft-c`…).
3. **Artefactos derivados → públicos**: la data compositiva generada (jsonl) y los adapters
   entrenados. Todos son reproducibles desde el repo (los kernels que los generan están
   versionados), pero publicarlos ahorra el paso de reproducción al jurado.
4. **Solution Writeup oficial en Kaggle**: los working notes bilingües
   ([working_note_es.md](working_note_es.md) / [working_note_en.md](working_note_en.md)) +
   [TACTICAS.md](TACTICAS.md) son la base — adaptarlos al formato del foro de Kaggle y adjuntar
   los enlaces a repo/kernels/datasets. Incluir la tabla de contribución por componente y los
   resultados negativos (los criterios del jurado premian exploración y análisis honesto).

## Compatibilidad de licencias de lo que usamos (verificado)

| Asset | Licencia | ¿OK? |
|---|---|---|
| Checkpoint NVARC (`sorokin/qwen3_4b_grids15_sft139`) | Apache 2.0 | ✅ |
| Pipeline LB33 (notebook público de Kaggle) | público, Apache (kernels Kaggle) | ✅ |
| Modelo BARC inducción (`barc0/...-8B`, HF) | Llama 3.1 Community License | ✅ uso permitido; los ganadores 2024/2025 usaron modelos Llama abiertos |
| Seeds + `common.py` de BARC (repo `xu3kev/BARC`) | pública (repo abierto) | ✅ vendorizado con atribución en los docs |
| Data `barc0/induction_heavy_100k_jsonl` (HF) | pública | ✅ |
| Nuestro repo | MIT | ✅ |

Nota: nuestros kernels/datasets privados NO violan reglas de "external data" — todo lo externo
que usamos es público y gratuito para cualquier participante; lo privado es solo **nuestro
trabajo derivado**, que se abrirá al cierre.

## Recordatorio operativo

- El deadline de la competencia define la ventana; **agendar la publicación el mismo día del
  cierre** para no comerse los 7 días.
- `push_kernels.py` genera `is_private: true` por defecto — al publicar, cambiar a `false` y
  re-pushear la versión final de cada kernel (o flip en la UI, más rápido).
