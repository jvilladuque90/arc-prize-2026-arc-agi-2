# ARC Prize 2026 — reglas operativas (NO negociables, del usuario)

## GPU / cuota Kaggle — leer ANTES de lanzar cualquier kernel

1. **La L4×4 quema cuota a TARIFA 2× del wall-clock** (1 h de pared = 2 h-quota).
   El ledger se lleva SIEMPRE en horas-quota.
2. **Tope semanal de este proyecto: 10 h-quota** (el pool de 30 h se comparte con agi-3).
   Experimentos hasta ~8 h-quota; las últimas ~2 h reservadas al commit de UNA submission.
   A 4 h-quota restantes en el pool global: parar todo salvo commits de submission.
3. **La L4×4 son 4 GPUs independientes (4×24 GB) y se paga la máquina COMPLETA: todo kernel
   de banco DEBE shardear en las 4 GPUs.** Un banco mono-GPU paga 4× de más. Patrón
   obligatorio en `scripts/plantilla_kernel_4gpu.py` (el del kernel campeón: main spawnea 4
   workers con `CUDA_VISIBLE_DEVICES=i`, shard `KEYS[i::4]`, main funde los JSON).
   `push_kernels.py` avisa si el notebook parece mono-GPU.
4. Un kernel = varias preguntas (consolidar mediciones); todo lo evaluable offline, offline.
   El rerun de scoring de una submission NO gasta cuota; solo el commit ("Save & Run").
   Máximo 1 submission/día. NUNCA sesiones interactivas con GPU.
5. `push_kernels.py` SIEMPRE con `PYTHONUTF8=1` y, para kernels del pipeline campeón o
   bancos 4B, `--accelerator NvidiaL4`.

## La naturaleza del sistema bajo prueba (regla del usuario, NO negociable)

- **Estamos midiendo un LLM. NO es determinista, y eso es normal, no un defecto.** La varianza
  entre corridas es parte de la naturaleza del objeto (coma flotante en GPU, orden de reducción
  en los kernels, temporización). **NO buscar determinismo donde no puede haberlo.**
- **El objetivo es SUBIR EL SCORE AL MÁXIMO y hacerlo lo más CONSISTENTE posible** — que es
  literalmente lo que premia la competencia, porque el veredicto sale de una sola corrida sobre
  el oculto. Nivel alto + varianza baja, en ese orden.
- Corolario operativo: **no gastar slots ni cuota en perseguir reproducibilidad bit a bit.** Se
  reduce varianza solo cuando cae como *subproducto* de una mejora real (ejemplo bueno: quitar
  el tope de reloj por tarea subió el nivel +4 tareas Y estrechó σ de 1,27 a 0,415; nadie pagó
  por la σ). Un injerto cuyo único efecto sea "más reproducible" no vale un slot.
- Corolario metodológico: la respuesta al ruido es **estadística adecuada**, no exactitud
  binaria — comparación pareada cuando se pueda, y **mediana y test de signo junto a la media**
  (en el barrido de rango la media decía "sin cambio" mientras 10/12 tareas empeoraban).

## PRESUPUESTO DE RELOJ — el campeón corre al 95% de su deadline (medido 7-sep)

- **Ningún injerto puede añadir más de ~4,8% de coste por tarea.** Inferido del LB: `r512` añadió
  +11,9% y perdió 15,3 tareas (6,4% de 240) ⇒ `deadline/coste_base = 1,048`.
- **Ahorrar tiempo NO da puntos** (ya completamos el 100%): `r64` fue 11% más rápido y perdió 2,3
  tareas por capacidad. **Gastarlo es catastrófico**: `r512` cayó del acantilado, −15,3 tareas.
- Presupuestar el coste **ANTES** de diseñar, no después. Quedan descartados por reloj:
  **thinking** (×2-5 en decode), **decode 16→24 vistas** (+28,6%), más pasos de TTT.
  Único ahorro identificado sin pérdida de capacidad conocida: **scoring 8→4 augs (−4%)**.
- **El cuello NO es la memoria, es el tiempo**: cero OOM en toda la campaña. `default_weights`
  (`arc_solver.py:317`) son 1,04 GB de adaptador duplicado en GPU y pasarlos a RAM es gratis,
  pero compra robustez, no puntos.
- ⚠️ **El smoke público de 4 tareas NO acota los extremos del oculto (240).** Tres veces ha
  costado caro presupuestar recursos sobre él. Usar la distribución completa de tamaños, offline.

## Campaña de saltos grandes (decisión del usuario, 9-sep) — NO negociable

- **Estamos a 30,69 contra un líder en 57%: un +1 o +2 no sirve, solo consume calendario.**
  Regla: *"no pienses en tiempo, si abre la posibilidad de saltos de score grandes vale la pena"*.
  **Descartado todo lo que viva en la banda de ruido**: modelo pequeño por velocidad (vale 0, no
  hay comprador para el reloj liberado), ajuste de rango, ajuste de vistas.
- **Plan A** = SFT contra el déficit de escala medido. **Plan B** = inducción aprendida a un DSL
  compacto + verificación exacta contra las demos. **NO son excluyentes**: B se traga el mecanismo
  de A (un programa es indiferente a la escala) y son complementarias en cobertura. Detalle
  completo en `docs/PLAN_AB.md`.
- **El leaderboard es el instrumento más barato y potente que tenemos**: n=240 por ~2 h-cuota del
  commit, frente a bancos que cuestan 1,6 h-cuota para 2 observaciones informativas (`rbuyer`).
  **Gastar cuota en comprar CAPACIDAD, no en comprar mediciones.** Sigue vigente: un injerto por
  brazo (el LB no atribuye cambios agrupados) y 1 slot/día.
- **Antes de cualquier gasto de GPU, la puerta gratis.** La Fase 0 enterró un diseño entero por
  0 h-cuota. Si la pregunta se contesta con aritmética sobre los JSON, se contesta ahí.
- ⚠️ **Calcular cuántos pares INFORMATIVOS dará un banco, no cuántas tareas.** `GT-en-pool` solo
  dispara en ~30%, así que la mayoría sale 0-0: hacen falta n≈40-60 pares ≈ 8-10 h-cuota, más que
  el tope semanal. Hay preguntas **inasequibles**, y eso NO es lo mismo que un resultado negativo.

## Bancos de medida — lecciones pagadas con puntos de LB

- **El vehículo de medida debe coincidir con producción en modelo, formato y población.**
  El 2B (sft141) memoriza train Y eval públicos → INVÁLIDO para veredictos sobre el 4B.
  Bancos de TTT: 4B + eval + augmentaciones (la vista única quedó invalidada).
- Tokens SIEMPRE construidos a mano CON marcadores de rol (ids 11/12); el tokenizer de
  Colab los descarta y eso costó una familia entera de experimentos (D1/D2/D4).
- **El TTT es EL MOTOR: +20,6 pt (nottt=9,58 vs campeón 30,14) por 34,8% del reloj.**
  No quitarlo; amplificarlo. `ttt8` en cuarentena (luz verde salida de banco contaminado).
- Verificar afirmaciones sobre producción CONTRA producción (los logs del campeón son gratis).
- Predicciones pre-registradas antes de cada corrida; poblaciones de calibración = las del
  test (lección lb33-cap, −2,5 pt).

## Infra

- Colab CLI: verificar el ADC activo antes de lanzar (cuentas 1-3 de este proyecto);
  corridas >1 h NUNCA con `colab run` (usar new+upload+exec desacoplado); persistir
  parciales a Kaggle tras cada etapa.
- Los detalles y la historia viven en `docs/TTT_PLAN.md`, `docs/SCALE_PLAN.md`,
  `docs/SPEED_PLAN.md` y la memoria persistente.
- **Modelos fuera de la imagen de Kaggle** (p.ej. Qwen3.5): exigen `transformers>=5.2` vía dataset
  de wheels (`juliancamilovilla/tf52-wheels`, pip `--no-index --no-deps --find-links` en el
  lanzador). La imagen actual NO trae bitsandbytes. **CLI en Windows**: `kaggle datasets create`
  falla con rutas largas → ejecutar **desde dentro del dir con `-p .`**; no encadenar con `| tail`
  (oculta el exit code).
