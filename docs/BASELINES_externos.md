# Baselines externos — búsqueda y auditoría (26-oct-2026)

Pregunta del usuario: *los líderes rondan 90; busca fuera de Kaggle (GitHub, web) el baseline público de
mayor score, audítalo, mira qué tiene que no tengamos.* Búsqueda hecha en Kaggle (API: leaderboard,
~190 kernels corridos desde el 1-oct, página de modelos), GitHub (topic `arc-agi-2`, repos citados) y web.

## 1. Veredicto en una línea

**No existe ningún baseline público con más score que el nuestro.** Lo mejor público verificable sigue
siendo la familia NVARC/`sft139` en **33,89** (nosotros 30,69 con el mismo código). Los equipos de 70–88
**no han publicado nada** y no están obligados hasta 7 días después del 2-nov. Lo que sí apareció es
**una vía nueva y abierta con techo alto** (§4) que no habíamos considerado: Qwen3.8-27B como agente
de síntesis de programas, verificado por ARC Prize en **42,4%** sobre AGI-2.

## 2. Estado del leaderboard (26-oct)

| # | equipo | score | fecha |
|---|---|---|---|
| 1 | Tufa Labs | **88,06** | 22-oct |
| 2 | Yi-Chia Chen @ Fields Model | 84,72 | 24-oct |
| 3 | Domdolus Tolus | 82,08 | 25-oct |
| 4 | nvbanana | 81,39 | 24-oct |
| 5 | rabbithole | 80,83 | 25-oct |
| 6–18 | Majkel1337 … Sakayori Lab | 76,8 → 68,1 | 24/25-oct |

3.375 equipos. **Dieciocho por encima de 68** y muchos nuevos en octubre: el método se está propagando,
pero por canales que no son notebooks públicos (ninguno de los 190 kernels revisados declara >33,89).
Yi-Chia Chen escribió en el foro de AGI-3 que **decidió no abrir su solución** del segundo hito.

## 3. Lo auditado y por qué NO sirve como baseline

| candidato | dónde | qué dice | por qué no |
|---|---|---|---|
| `dastin359/athanor` (GitHub, 9★) | GitHub | **95,7% en eval público** a $3,12/tarea | usa **APIs** de Anthropic/OpenAI/Gemini: no corre offline. Además eval *público* (AGI-2 lo cita como contaminado para casi todo) |
| `epang080516/arc_agi` (162★) | GitHub | 26,0% semi-privado, DreamCoder con LLM | llama a **Grok-4 por API** |
| `Hisernberg/arc-agi-2-solver` | GitHub | título «High-Scoring»; banda **27,36–30,97**, «PB 32,64» | es NVARC + DSL de 15 primitivas + fusión de selección: **nuestro mismo nivel** |
| `alnasserlegalservices/merenvay…` | GitHub | «candidato 2026» | «NOT YET SUBMITTED / NOT YET VERIFIED», sin número |
| `RaphaelKhalid/arcagi2kaggle` | GitHub | NVARC + predictores simbólicos de forma/paleta | su propia web: **«Score unproven»**; 0 ablaciones |
| `davidlueng986-alt/arc-agi-2-kaggle` | GitHub | rastrea los notebooks públicos | confirma: mejor público nombrado = **33,89**; #1 «no publicado» |
| `yukailiu/…champion-solution`, `woguoat/…triple-pass`, `rabari9999/…silver-sota` | Kaggle | NVARC + lotes de 8 vistas + pasada B + fallback DSL | sin score declarado; el propio código dice que la v8 «con boosts» dio **26,39** |
| `yusuketogashi/arc-baseline-rebuild` (107 votos) | Kaggle | 33,89 + semilla crc32 + presupuesto tras el TTT + **retry con umbral 0,1** en pools vacíos | mismo modelo; la única idea nueva (retry a 0,1) la medimos nosotros: abrir el umbral compra +1/12 por +75% de tiempo |
| `evgendvorkin/arc-2-2026-3-4` (56 votos) | Kaggle | Boyko × NVARC+ dual-pass, cita «Triple-Pass NVARC+ SOTA **35,9** LB» de `mayankraaz` | el kernel de mayankraaz **ya no está descargable**; el 35,9 queda sin verificar y está dentro del rango del pelotón (34–39) |
| `dmitriigluzdov/…six-added-layers` | Kaggle | sft139 + 6 capas nuevas entrenadas sobre 342K sintéticos | **29,17** vs 30,83 de la base en sus corridas: negativo, coherente con nuestros cierres |

Lo que **todos** los públicos de la familia NVARC añaden y nosotros ya probamos o descartamos: pasadas
múltiples con semillas (medido: +0), fallback DSL (0% en AGI-2), retry con umbral abierto (+1/12 por
+75%), más capas/SFT sintético (negativo). **No hay nada en esa familia que no tengamos.**

## 4. Lo que SÍ es nuevo: Qwen3.8-27B como agente de síntesis de programas (offline)

**Hecho verificado por ARC Prize** (arcprize.org/results/alibaba-qwen3-8-27b): Qwen3.8-27B, modelo
abierto, con `reasoning_effort=xhigh`: **ARC-AGI-2 semi-privado 42,4%** ($0,45/tarea), AGI-1 87,5%.
Con esfuerzo *low* 22,8%, *medium* 13,2%, sin razonamiento 1,5%. **El score depende casi por completo
del presupuesto de razonamiento.**

**Existe infraestructura pública para correrlo en Kaggle offline**: checkpoint FP8 repacado
(`foysalemonshanto/qwen3-8-27b-fp8-repacked-v1`, `mikedan7/qwen3-8-27b-fp8-official`), wheels de vLLM
(`mirzamilanfarabi/arc3-vllm-h100-wheelhouse-v3`, `keithtyser/taaf-duck-qwen38-serving-v1`) y tres
notebooks de agente en 4×L4: `madarshbb/arc-qwen-3-8-27b-agent` (41 votos), `delongyang/…within-4-hours`,
`khoinguyennguyen/…agentic-multi-turn-reasoning`.

**Cómo funciona el agente de `madarshbb`** (leído del código): vLLM sirve el 27B FP8 con KV en FP8 y
decodificación especulativa MTP; el modelo trabaja en un **REPL de Python**: escribe un programa, lo
ejecuta contra las demos, corrige y hace `submit` solo cuando el programa **reproduce todas las demos**;
2 muestras por tarea = los dos intentos; `thinking_token_budget` 40.960, `max_tokens` 49.152,
`concurrency` 10, reserva de 35 min.

**Su score en el LB: 21,53** (reportado por terceros; el notebook no lo declara). **Por qué 21,5 y no 42**:
aritmética de reloj con el *propio* supuesto del notebook (140 tok/s agregados):
140 × 11,4 h ≈ **5,7 M tokens por corrida** → 240 tareas × 2 muestras ≈ **12 K tokens por intento**,
frente a los **~49 K** que pide: cabe **un cuarto** del razonamiento. ARC Prize mide *xhigh* sin tope de
tiempo; en L4×4 el agente corre en régimen «low/medium» y eso vale 13–23%. **El cuello es throughput.**

Para 1 intento × 45 K tokens en 240 tareas harían falta **~260 tok/s**, casi el doble. Dónde puede salir:
(a) un solo intento en vez de 2 (los dos intentos valen menos que doblar el razonamiento del primero);
(b) speculative MTP bien configurado (el notebook ya lo intenta en su «server_ladder»);
(c) ordenar tareas por tamaño y dar más tokens a las que las demos sugieren resolubles; (d) `max_num_seqs`
y concurrencia afinadas para la L4 (el notebook está tuneado a ojo para H100).

**Qué tiene que nosotros no tenemos:** inducción de programas por un modelo que razona, con verificación
exacta contra las demos. Es *exactamente* lo que nuestras mediciones dicen que falta
(`rearcvac`: el déficit es inducir una regla **nueva** en test; los DSL a mano murieron por expresividad,
no por la idea). **Qué tenemos que ellos no:** nada que les sirva — el TTT/transducción es otro paradigma.

## 5. Lectura honesta de hacia dónde apuntan los 80+

Lo único público que casa con esos scores es el modelo: 42,4% *con un solo modelo sin TTT, sin
ensamblar y sin tuning*. Un equipo con tiempo (desde abril) que combine el 27B con muestreo amplio,
verificación y selección puede estar en 70–88. Nuestro 4B con TTT está **estructuralmente** por debajo:
no es un problema de injertos.

## 6. Qué propongo (y qué no)

1. **No adoptar ningún notebook público como baseline**: ninguno supera lo que tenemos. `lb33-det` sigue
   siendo la submission segura.
2. **Puerta del 27B, barata (~1,5 h-cuota)**: clonar `madarshbb/arc-qwen-3-8-27b-agent` tal cual, con
   el modelo y los wheels **copiados a datasets nuestros el mismo día** (lección `nem8b`), y correrlo
   sobre las **18 tareas de `strat`** con un tope de pared de 40 min. Prerreg: cuenta cuántas de las 15
   que el campeón falla resuelve con verificación exacta, y mide **tok/s reales en L4** (el número que
   decide todo). Con ≥3 y ≥120 tok/s: hay una submission de salto posible antes del 2-nov (quedan 7 días).
3. **Si pasa la puerta**: 1 intento por tarea, orden por coste, y el resto del reloj al razonamiento.
   Se mide en el LB (n=240) con un solo commit.
4. **Si no pasa**: `lb33-det` final + Paper Track.

Riesgos declarados: 7 días de calendario; la infraestructura vLLM/FP8 en L4 es ajena y frágil (el
notebook lleva una «escalera» de 6 configuraciones de arranque porque fallan); los 21,53 públicos son
reporte de terceros.
