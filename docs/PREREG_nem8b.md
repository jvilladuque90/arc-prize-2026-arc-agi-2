# PREREG `nem8b` (2026-10-23) — ¿aporta un generador alternativo (Qwen3-8B-Nemotron77K, sin TTT)?

**Pregunta (U6/U7 de PANORAMA.md):** entre las tareas que el campeón FALLA, ¿cuántas resuelve este 8B,
que razona (`<think>`) y emite la rejilla en `\boxed{}`? Única vía con techo alto, pero con prior bajo:
no existe ningún resultado publicado de este modelo en AGI-2 (búsqueda del 22-oct).

**Diseño:** las 18 tareas de `strat` (producción exacta; **15 de ellas el campeón NO tiene la GT en su
pool**, 3 sí). 4 GPUs, `generate` de HF, muestreo T=0,6, **un intento por tarea**, tope 16.000 tokens
nuevos, no se lanzan tareas nuevas pasados 45 min. Formato del prompt elegido por **NLL** de un target
real del dataset sobre 4 variantes (chat/raw × con/sin `<think>`), no a ojo; la plantilla del prompt
reproduce 73/73 prompts reales del dataset.

**Limitación declarada de antemano:** las trazas medianas son ~18K tokens (p90 ~37K) y aquí se cortan a
16K, así que parte de los «0» serán truncados. Por eso se cuentan aparte: *resuelta*, *respondió pero
mal*, *sin respuesta (truncada/sin cierre)* y *no lanzada por reloj*.

**U** = tareas que el campeón falla y el 8B resuelve.

| resultado | lectura | siguiente paso |
|---|---|---|
| **U ≥ 3** | hay señal complementaria real | ingeniería: vLLM+FP8, presupuesto de reloj, sitio en el pipeline |
| **U = 1–2** | marginal (con 15 tareas, ±1 es ruido) | no invertir; anotar |
| **U = 0** y (sin respuesta + no lanzadas) ≤ 50% de las 18 | línea muerta | cerrar U6/U7 |
| **U = 0** y (sin respuesta + no lanzadas) > 50% | **INCONCLUSA**, no muerta | solo reabrir si hay un motor rápido (vLLM) |

**Sanidad:** el NLL por formato debe separar claramente al mejor (>0,3 nats de diferencia con el peor);
si no, el formato no está identificado y no se lee nada.

**Coste previsto:** ~60–75 min de pared ≈ **2–2,5 h-cuota** (peor caso ~76 min). Un solo banco.

---

## RESULTADO (23-oct): BANCO NO VÁLIDO — el modelo ya no existe en Kaggle

Los 4 workers abortaron en 2 min con `assert MDIR` (no se montó ningún modelo). Causa: Kaggle descartó
en silencio `model_sources` (metadata committeada `[]`, el fallo que avisa TACTICAS.md) y, al
investigar, `GetModel`/`GetModelInstance` devuelven **403**: `nisshokuitsuki/qwen3-8b-nemotron77k-grid-direct`
era accesible el 29-sep y **ya no lo es** (retirado o vuelto privado). Probadas ambas capitalizaciones
del framework; no hay copia en Hugging Face ni otro espejo en Kaggle. **No hay lectura de U**: ni a favor
ni en contra. Coste: ~0,1 h-cuota (2 min en L4×4 + un intento en T4).

**Dos errores míos, dichos claro:** (1) no comprobé la metadata committeada tras el push (lo hice solo
después de ver el fallo); (2) el 29-sep dejé la apuesta C «para después» sin copiar el modelo a un
dataset propio, y un artefacto de un tercero puede desaparecer. Regla: si una decisión depende de un
artefacto ajeno, **copiarlo a algo nuestro el día que se decide que importa**.
