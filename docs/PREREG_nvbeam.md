# PREREG `nvbeam` (2026-10-28) — haz-8 como respaldo en la pasada B de NVARC+ para los pools vacíos

## Replanteo (petición del usuario: seguir buscando por encima de 30,97)

Dos mediciones que se hicieron aisladas y nunca juntas:

1. **El haz-8 encuentra la GT en ~1/3 de los pools vacíos** (`haz`: 2 de 6 vacías de `strat`; `haz2`: 2 de 6,
   conjunto distinto ⇒ efecto real sobre un conjunto fronterizo). Murió por reloj: +29,1% **dentro de una pasada a
   carga completa que ya usa el 95%**.
2. **NVARC+ a media carga deja ~5 h libres** y las gasta en una pasada B con semillas nuevas que medimos en **+0**
   (`ttt-2hip`: Jaccard 0,26, unión +0), y además repite las vacías con las mismas semillas (bug corregido en
   `nvplus_a1`, nunca ejecutado). NVARC+ = 29,44 · 30,97 (media 30,2).

`nvbeam` = pasada A de NVARC+ (64 seq, 8 vistas, tope 800 s) → marcador de procesada → **pasada B: las tareas con
pool VACÍO reciben TTT con semillas nuevas + haz-8 sobre 2 vistas (sin umbral p>0,2)**, sus candidatos pasan por el
mismo re-scoring de 8 augmentaciones y `kgmon`; el tiempo sobrante va a semillas nuevas para el resto.

## Aritmética de reloj (medida, no supuesta)

- Pools vacíos: ~33% (`autopsia`, carga completa; a media carga serán más) ⇒ ~80–100 tareas.
- Haz-8 vectorizado: 51,6 s/tarea de media (`haz2`), tope 240 s por vista; 2 vistas ⇒ ≤ 8 min/tarea, típico ~2 min.
- TTT a media carga (64 seq): ~2–3 min/tarea. Re-scoring: ≤ 8 candidatos × 2 vistas.
- ⇒ ~5–6 min/tarea × 90 tareas / 4 GPUs ≈ **2–2,3 h** de las ~5 h de la pasada B. Cabe, y sobra para `seedB`.

## Lo que puede salir mal (dicho antes)

- La tasa 1/3 se midió con TTT a carga completa (128 seq); con 64 seq el adaptador es más débil.
- El haz devuelve 8 candidatos aunque sean malos: la selección `kgmon` sobre pools antes vacíos no tiene
  competencia, así que **no puede empeorar** una tarea que valía 0 (solo gasta reloj de B).
- Cache de unsloth: `_reordena` lanza si no reconoce el tipo (no hay no-op silencioso); el smoke lo ejercita.

## Lecturas (fijadas antes; contra `lb33-det` 30,69 ± 0,415 y NVARC+ 30,2 ± 1,1)

| LB | lectura | siguiente |
|---|---|---|
| **≥ 32,5** | el haz convierte pools vacíos en aciertos a escala; nueva base | segunda lectura; luego ancho 16 / 4 vistas |
| 31,0 – 32,5 | probable mejora, dentro de 2σ de NVARC+ | segunda lectura antes de decidir |
| 29,5 – 31,0 | nada | cerrar el haz en B; probar B = umbral 0,1 en vacías |
| < 29,5 | el haz en B rompe algo (reloj de A o selección) | revisar `passB_plan.json` y logs del smoke |

**Coste:** commit = smoke de 2 tareas con haz forzado (~25–30 min ≈ 0,9 h-cuota); el resto en el rerun (0 cuota).
