"""Empaqueta y publica los notebooks como kernels de Kaggle (GPU) via Kaggle CLI.

Crea una carpeta temporal con el notebook + kernel-metadata.json y ejecuta
`kaggle kernels push`. Ambos kernels van SIN internet por defecto: las maquinas L4x4
(96GB) de esta competencia EXIGEN internet deshabilitado, asi que el modelo base y las
deps deben venir de Kaggle Models/Datasets adjuntos. Usa --internet para forzar red
(solo valido en T4x2/P100, que descargarian el modelo de HF).

Ejemplos:
  # Entrenamiento en GPU L4x4 (offline): adjunta codigo + modelo base de Kaggle Models
  python scripts/push_kernels.py train \
      --code-dataset juliancamilovilla/arc-agi2-code \
      --model qwen-lm/qwen2.5/transformers/0.5b-instruct

  # Inferencia/submission offline: adjunta codigo + base + LoRA entrenado
  python scripts/push_kernels.py submit \
      --code-dataset juliancamilovilla/arc-agi2-code \
      --model qwen-lm/qwen2.5/transformers/0.5b-instruct \
      --dataset juliancamilovilla/arc-agi2-model
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMP = "arc-prize-2026-arc-agi-2"

# Imagen Docker pinneada (la de los runs buenos de arc-agi2-eval). La imagen "latest"
# de Kaggle (jul-2026) trae transformers que exige torchao>=0.16 pero incluye 0.10 ->
# ImportError al cargar el modelo. Pinnear tambien hace reproducible el rerun de submission.
DOCKER_IMAGE = ("gcr.io/kaggle-private-byod/python@sha256:"
                "57e612b484cf3df5026ee4dcc3cb176974b22b2bc0937fb1e16132a8be4cb13c")


def load_env(env_path: Path) -> None:
    # El .env es la fuente de verdad de credenciales: SOBREESCRIBE el entorno del sistema
    # (que puede tener un KAGGLE_API_TOKEN obsoleto que rompería la autenticación).
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip()


def kaggle_username() -> str:
    return (
        os.environ.get("KAGGLE_USERNAME")
        or os.environ.get("kaggle_username")
        or "juliancamilovilla"
    )


KERNELS = {
    "train": {"notebook": "notebooks/train.ipynb", "slug": "arc-agi2-train",
              "title": "arc agi2 train", "gpu": True},
    "submit": {"notebook": "notebooks/submit.ipynb", "slug": "arc-agi2-submit",
               "title": "arc agi2 submit", "gpu": True},
    "eval": {"notebook": "notebooks/evaluate.ipynb", "slug": "arc-agi2-eval",
             "title": "arc agi2 eval", "gpu": True},
    # Baseline LB33/NVARC (Vía A): notebook autocontenido, NO usa src/arc; requiere el
    # kernel de wheels de sorokin (--kernel-source) y el modelo NVARC (--model). Ver LB33_AUDIT.md.
    # Trae su propia imagen BYOD (unsloth) — override del docker por defecto.
    "lb33": {"notebook": "notebooks/lb33_baseline.ipynb", "slug": "arc-agi2-lb33-baseline",
             "title": "arc agi2 lb33 baseline", "gpu": True,
             "docker": ("gcr.io/kaggle-private-byod/python@sha256:"
                        "320043e14c68293f1c946585b9257123385205a58af4b94b17d31868cae4e868"),
             "defaults": {"kernel_source": ["sorokin/pip-install-unsloth-flash-patch"],
                          "model": ["sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"]}},
    # Injerto de cobertura (cheap-first) sobre el baseline; mismo entorno. Ver LB33_AUDIT.md.
    "lb33-graft": {"notebook": "notebooks/lb33_graft.ipynb", "slug": "arc-agi2-lb33-graft",
                   "title": "arc agi2 lb33 graft", "gpu": True,
                   "docker": ("gcr.io/kaggle-private-byod/python@sha256:"
                              "320043e14c68293f1c946585b9257123385205a58af4b94b17d31868cae4e868"),
                   "defaults": {"kernel_source": ["sorokin/pip-install-unsloth-flash-patch"],
                                "model": ["sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"]}},
    # Campeon + TOPE DE LONGITUD POR TAREA (17-ago): max_new_tokens deja de ser 931 fijo y pasa
    # a 4x max(mayor salida de train, entrada de test). El lote de 4 vistas corre hasta la mas
    # lenta, y una vista que divaga lo arrastra entero. GT cabe en el 100% de 1248 subtareas
    # -> monotono. Mismo entorno que lb33-graft; un solo numero cambia.
    "lb33-cap": {"notebook": "notebooks/lb33_cap.ipynb", "slug": "arc-agi2-lb33-cap",
                 "title": "arc agi2 lb33 cap", "gpu": True,
                 "docker": ("gcr.io/kaggle-private-byod/python@sha256:"
                            "320043e14c68293f1c946585b9257123385205a58af4b94b17d31868cae4e868"),
                 "defaults": {"kernel_source": ["sorokin/pip-install-unsloth-flash-patch"],
                              "model": ["sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"]}},
    # Campeon + REPOSICION DE LOS MARCADORES DE ROL (18-ago). El tokenizador (WordLevel, 13
    # entradas, 0 merges) se come 'user'/'assistant', asi que los ids 11/12 nunca llegan al
    # modelo aunque fue entrenado con ellos. A/B en Colab: NLL de la GT 18.6->0.037, GT bajo
    # umbral 0/14->14/14, pool 0->16. Solo cambia la tokenizacion del prompt. LANZAR CON
    # --accelerator NvidiaL4 (el campeon spawnea 4 procesos, uno por GPU).
    "lb33-roles": {"notebook": "notebooks/lb33_roles.ipynb", "slug": "arc-agi2-lb33-roles",
                   "title": "arc agi2 lb33 roles", "gpu": True,
                   "docker": ("gcr.io/kaggle-private-byod/python@sha256:"
                              "320043e14c68293f1c946585b9257123385205a58af4b94b17d31868cae4e868"),
                   "defaults": {"kernel_source": ["sorokin/pip-install-unsloth-flash-patch"],
                                "model": ["sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"]}},
    # EXPERIMENTO: campeon con el TTT DESACTIVADO. Mide cuanto aporta realmente el TTT, que
    # consume el 34.8% del presupuesto. En Colab se midio que sobreajusta (train a NLL ~0, GT
    # del test sin mejorar y en 4/10 EMPEORA). El reload score del smoke se compara con el 2.5
    # historico del campeon. LANZAR CON --accelerator NvidiaL4.
    "lb33-nottt": {"notebook": "notebooks/lb33_nottt.ipynb", "slug": "arc-agi2-lb33-nottt",
                   "title": "arc agi2 lb33 nottt", "gpu": True,
                   "docker": ("gcr.io/kaggle-private-byod/python@sha256:"
                              "320043e14c68293f1c946585b9257123385205a58af4b94b17d31868cae4e868"),
                   "defaults": {"kernel_source": ["sorokin/pip-install-unsloth-flash-patch"],
                                "model": ["sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"]}},
    # CE FOCAL vs ESTANDAR en la epoca unica, banco FIEL a produccion (post-ttt05=26,39):
    # 1 epoca exacta, decode AGREGADO de 8 vistas con des-transformacion (inversion verificada
    # 300/300) y voto probmul-like, 12 tareas estratificadas por el triage, shardeado x4.
    # ~20 min de pared (~0,7 h-quota). Prerreg: efecto modesto (+0 a +2), primero en el grupo
    # "sobreajusta".
    "ttt-focal": {"notebook": "notebooks/ttt_focal.ipynb", "slug": "arc-agi2-ttt-focal",
                  "title": "arc agi2 ttt focal", "gpu": True,
                  "defaults": {"model": ["sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"]}},
    # PAR DETERMINISTA (1-sep). DOS cambios comunes, ambos contra el no-determinismo:
    #  (1) seed de scoring: hash(bk) -- salado por proceso -- pasa a md5 estable.
    #  (2) se elimina el tope de RELOJ por tarea (spend_time > 1200). Medido en el log
    #      del campeon: 981571dc arranco su ultimo lote a 1159,6 s, a 40,4 s del tope
    #      -> un 3,5% de jitter decidia 16 vistas vs 12. Una tarea = 0,8333 pt y la
    #      sigma observada es 1,27 (~1,5 tareas): la granularidad del ruido cuadra.
    #      El deadline GLOBAL se conserva (proteccion de submission). El DFS mantiene
    #      su propio tope de 540 s, que nunca mordio (lote mas lento medido: 203,7 s).
    # lb33-det = CONTROL (campeon determinista). Dos submits: si dan el MISMO score, el
    # determinismo funciona y a partir de ahi los A/B son PAREADOS (§21).
    "lb33-det": {"notebook": "notebooks/lb33_det.ipynb", "slug": "arc-agi2-lb33-det",
                 "title": "arc agi2 lb33 det", "gpu": True,
                 "docker": ("gcr.io/kaggle-private-byod/python@sha256:"
                            "320043e14c68293f1c946585b9257123385205a58af4b94b17d31868cae4e868"),
                 "defaults": {"kernel_source": ["sorokin/pip-install-unsloth-flash-patch"],
                              "model": ["sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"]}},
    # lb33-ttford (10-sep) = campeon lb33-det + UN injerto: en el dataset de TTT la demostracion
    # de SALIDA MAYOR queda siempre la ultima (contexto aleatorio). Cada una de las 128 secuencias
    # pasa a predecir la salida mas grande con solo las pequenas de contexto. Coste de reloj CERO.
    # Mecanismo: 75% de GT-en-pool cuando la salida cae DENTRO de la envolvente de las salidas demo
    # frente a 11% cuando cae fuera, y el 82% del eval esta fuera. El decode (eval_ds) no se toca.
    # PRERREG: >31,52 adoptar | 29,86-31,52 ruido | <29,86 revertir.
    "lb33-ttford": {"notebook": "notebooks/lb33_ttford.ipynb", "slug": "arc-agi2-lb33-ttford",
                    "title": "arc agi2 lb33 ttford", "gpu": True,
                    "docker": ("gcr.io/kaggle-private-byod/python@sha256:"
                               "320043e14c68293f1c946585b9257123385205a58af4b94b17d31868cae4e868"),
                    "defaults": {"kernel_source": ["sorokin/pip-install-unsloth-flash-patch"],
                                 "model": ["sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"]}},
    # lb33-stack = TRATAMIENTO: determinista + los dos injertos de cobertura rehabilitados
    # (cheap-first y tope de longitud por tarea; el "GRAFT 3" de 600s nunca se aplico -- el
    # codigo sigue en 1200). Ambos monotonos y ya validados como no-daninos (§22).
    "lb33-stack": {"notebook": "notebooks/lb33_stack.ipynb", "slug": "arc-agi2-lb33-stack",
                   "title": "arc agi2 lb33 stack", "gpu": True,
                   "docker": ("gcr.io/kaggle-private-byod/python@sha256:"
                              "320043e14c68293f1c946585b9257123385205a58af4b94b17d31868cae4e868"),
                   "defaults": {"kernel_source": ["sorokin/pip-install-unsloth-flash-patch"],
                                "model": ["sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"]}},
    # SFT fuera-de-envolvente (9-sep), L4x4 con DDP en las 4 GPUs. Entrena sft139 a producir
    # salidas cuyas dimensiones/estadistica caen FUERA del rango que mostraron las demos, que es
    # el regimen donde el campeon pasa de 75% a 11% de GT-en-pool y que cubre el 82% del eval.
    # Datos: re-particiones REALES del training set por tamano de salida (803 de 625 tareas), NO
    # sinteticos -- la puerta del DSL probo que los sinteticos que yo generaria no existen en AGI-2.
    # Perdida solo en el cuerpo de la salida objetivo. Val por TAREA. PRERREG: >=10% de mejora.
    "sft-outenv": {"notebook": "notebooks/sft_outenv.ipynb", "slug": "arc-agi2-sft-outenv",
                   "title": "arc agi2 sft outenv", "gpu": True,
                   "docker": ("gcr.io/kaggle-private-byod/python@sha256:"
                              "320043e14c68293f1c946585b9257123385205a58af4b94b17d31868cae4e868"),
                   "defaults": {"kernel_source": ["sorokin/pip-install-unsloth-flash-patch"],
                                "model": ["sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"]}},
    # nemvac (30-sep): detector de vacuidad de los puzzles Nemotron bajo sft139 SIN TTT (un forward
    # por puzzle) con controles memorizado (AGI-1 train) y no visto (eval). Decide si esos puzzles
    # sirven como SFT (Plan A). PREREG en docs/PREREG_nemvac.md. LANZAR CON --accelerator NvidiaL4.
    "nemvac": {"notebook": "notebooks/nemvac.ipynb", "slug": "arc-agi2-nemvac",
               "title": "arc agi2 nemvac", "gpu": True,
               "docker": ("gcr.io/kaggle-private-byod/python@sha256:"
                          "320043e14c68293f1c946585b9257123385205a58af4b94b17d31868cae4e868"),
               "defaults": {"kernel_source": [],
                            "model": ["sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"]}},
    # nvplus-a1 (29-sep): nvplus + pasada B DIRIGIDA. Las tareas con pool vacio no escriben shard y
    # NVARC+ las repetia en el catch-up con las MISMAS semillas; ahora hay marcador de procesada, el
    # catch-up es solo para no alcanzadas y B va vacias -> un candidato -> resto. LANZAR CON --accelerator NvidiaL4.
    "nvplus-a1": {"notebook": "notebooks/nvplus_a1.ipynb", "slug": "arc-agi2-nvplus-a1",
                  "title": "arc agi2 nvplus a1", "gpu": True,
                  "docker": ("gcr.io/kaggle-private-byod/python@sha256:"
                             "320043e14c68293f1c946585b9257123385205a58af4b94b17d31868cae4e868"),
                  "defaults": {"kernel_source": ["sorokin/pip-install-unsloth-flash-patch"],
                               "model": ["sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"]}},
    # nvplus (29-sep): NVARC+ v2 publico (luxluxshan/arc2-nvarc-v1) SIN CAMBIOS, baseline candidato.
    # Mismo sft139 y misma imagen que el campeon; cambia la orquestacion: pasada A (64 seq TTT, 8 vistas,
    # tope 800 s/tarea) + catch-up + pasada B con semillas nuevas (tope 700 s), seleccion kgmon FUSIONADA
    # que nunca pierde el top-1 de A, y fallback identidad en vez de [[0]]. LANZAR CON --accelerator NvidiaL4.
    "nvplus": {"notebook": "notebooks/nvplus.ipynb", "slug": "arc-agi2-nvplus",
               "title": "arc agi2 nvplus", "gpu": True,
               "docker": ("gcr.io/kaggle-private-byod/python@sha256:"
                          "320043e14c68293f1c946585b9257123385205a58af4b94b17d31868cae4e868"),
               "defaults": {"kernel_source": ["sorokin/pip-install-unsloth-flash-patch"],
                            "model": ["sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"]}},
    # haz2 (21-sep): el haz VECTORIZADO. haz midio +2/18 tareas recuperadas pero 59 s/tarea
    # (+33,5% de reloj), artefacto de 96 .item() por paso. Aqui una ordenacion en GPU y dos
    # .tolist() por paso, prefill con lote 1 + expansion de cache. Anchos 8 y 16. Guarda los
    # candidatos del haz con score para re-jugar la seleccion offline. Decide si cabe en el techo.
    "haz2": {"notebook": "notebooks/haz2.ipynb", "slug": "arc-agi2-haz2",
             "title": "arc agi2 haz2", "gpu": True,
             "docker": ("gcr.io/kaggle-private-byod/python@sha256:"
                        "320043e14c68293f1c946585b9257123385205a58af4b94b17d31868cae4e868"),
             "defaults": {"kernel_source": ["sorokin/pip-install-unsloth-flash-patch"],
                          "model": ["sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"]}},
    # haz (21-sep): re-medicion CON TTT de la familia de reglas de aceptacion, cerrada el 22-ago
    # sin TTT y con una sola vista. La auditoria encontro que el 33% de tareas no produce NINGUN
    # candidato aunque el modelo este al 96-98% por celda: impuesto al tamano del umbral absoluto.
    # Instrumento: rango del token correcto con teacher forcing (1 forward) => ancho de haz
    # necesario ~ exp(deficit vs greedy). Mas haz real de ancho 1/8/32. 18 tareas de strat.
    "haz": {"notebook": "notebooks/haz.ipynb", "slug": "arc-agi2-haz",
            "title": "arc agi2 haz", "gpu": True,
            "docker": ("gcr.io/kaggle-private-byod/python@sha256:"
                       "320043e14c68293f1c946585b9257123385205a58af4b94b17d31868cae4e868"),
            "defaults": {"kernel_source": ["sorokin/pip-install-unsloth-flash-patch"],
                         "model": ["sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"]}},
    # rb2 (12-sep): la pregunta del comprador con el proxy validado en strat. r512 vs r256 sin
    # presion de reloj sobre las mismas 19 tareas estratificadas (sin la contaminada), 2 GPUs por
    # brazo, SIN decode: primaria = NLL/celda de la GT post-TTT (informa en el 100% de tareas).
    # PRERREG: ratio mediano <=0,85 y signo p<0,10 => hay comprador (rama Qwen3.5-2B viva);
    # >=1,15 => aleja; lo demas empate = negativo.
    "rb2": {"notebook": "notebooks/rb2.ipynb", "slug": "arc-agi2-rb2",
            "title": "arc agi2 rb2", "gpu": True,
            "docker": ("gcr.io/kaggle-private-byod/python@sha256:"
                       "320043e14c68293f1c946585b9257123385205a58af4b94b17d31868cae4e868"),
            "defaults": {"kernel_source": ["sorokin/pip-install-unsloth-flash-patch"],
                         "model": ["sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"]}},
    # strat (10-sep): produccion exacta, UN brazo, 20 tareas de eval estratificadas p5-p95 por
    # coste, LPT en 4 workers, deadline sobre la cola y dentro de la tarea. Cierra el punto ciego de
    # la auditoria: toda la evidencia por tarea venia del decil mas barato. Registra GT-en-pool,
    # pools con scores y NLL/celda pre/post TTT. No prueba injertos: produce hechos (P1-P4).
    "strat": {"notebook": "notebooks/strat.ipynb", "slug": "arc-agi2-strat",
              "title": "arc agi2 strat", "gpu": True,
              "docker": ("gcr.io/kaggle-private-byod/python@sha256:"
                         "320043e14c68293f1c946585b9257123385205a58af4b94b17d31868cae4e868"),
              "defaults": {"kernel_source": ["sorokin/pip-install-unsloth-flash-patch"],
                           "model": ["sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"]}},
    # r-buyer (9-sep), L4x4: r512 vs r256 SIN presion de reloj, pareado sobre 12 tareas de
    # eval estratificadas por coste. Separa el confundido del acantilado de r512 (-15,3 tareas
    # en LB con +11,9% de coste): reloj o capacidad? Bloquea la rama Qwen3.5-2B (+12,4 pt), que
    # necesita que el reloj liberado tenga comprador. Primaria vistas_con_gt; el EMPATE cuenta
    # como negativo (registrado antes de correr). 2 GPUs por brazo, sin fuga de estado.
    "rbuyer": {"notebook": "notebooks/rbuyer.ipynb", "slug": "arc-agi2-rbuyer",
               "title": "arc agi2 rbuyer", "gpu": True,
               "docker": ("gcr.io/kaggle-private-byod/python@sha256:"
                          "320043e14c68293f1c946585b9257123385205a58af4b94b17d31868cae4e868"),
               "defaults": {"kernel_source": ["sorokin/pip-install-unsloth-flash-patch"],
                            "model": ["sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"]}},
    # gate35 (10-sep), L4x4 shard x4: re-mide Qwen3.5-4B contra el titular con DIAGNOSTICO DE
    # CARGA (output_loading_info: cuantas claves quedan ausentes => tensores aleatorios) y CONTROL
    # ENTRE ESPEJOS (limamateus vs thanhduc1102). En gate-base se cargaron 4.206M de 4.660M params
    # y el 3.5-4B salio peor que el 3.5-2B de su propia familia: hay que descartar artefacto de
    # carga antes de enterrar la rama. PRERREG: carga limpia + ratio <=0,85 + fwd <=1,05x => arranca.
    "gate35": {"notebook": "notebooks/gate35.ipynb", "slug": "arc-agi2-gate35",
               "title": "arc agi2 gate35", "gpu": True, "docker": "",
               "defaults": {"dataset": ["juliancamilovilla/tf52-wheels"],
                            "model": ["qwen-lm/qwen-3/transformers/4b/1",
                                      "limamateus/qwen3-5-4b/pyTorch/default/1",
                                      "thanhduc1102/qwen_35_4b/transformers/default/1"]}},
    # GATE de modelo base a 4 bandas + ablacion de features (9-sep), L4x4 shardeado x4.
    # Prior (NLL/celda de la GT, pareado, 24 tareas eval) del titular Qwen3-4B contra
    # Qwen3.5-4B (mismo tamano), Qwen3.5-2B (con descuento) y Qwen3-1.7B (ancla). Ademas
    # censo de parametros por bloque y forward con lm_head recortado a 16 filas: cuanto del
    # modelo es peso muerto si solo hacemos texto/rejillas. PRERREG: q35_4b vive solo con
    # ratio <= 0,85 Y gana >= 2/3 Y fwd <= 1,05x; q35_2b le basta ratio <= 1,05.
    "gate-base": {"notebook": "notebooks/gate_base.ipynb", "slug": "arc-agi2-gate-base",
                  "title": "arc agi2 gate base", "gpu": True, "docker": "",
                  "defaults": {"dataset": ["juliancamilovilla/tf52-wheels"],
                               "model": ["qwen-lm/qwen-3/transformers/4b/1",
                                         "qwen-lm/qwen-3/transformers/1.7b/1",
                                         "barnobarno/qwen3.5-4b/transformers/unsloth/1",
                                         "akzsh5100/qwen-3-5-2b/transformers/transformers/1"]}},
    # GATE MoE (8-sep): prior de ARC de Qwen3.5-35B-A3B (stock, 4-bit) vs Qwen3-4B (stock, bf16),
    # NLL/celda de la GT con las demos en texto, 24 tareas eval, pareado, shard x4. Mide tambien el
    # tiempo de forward de cada uno (valida la aritmetica de bandwidth de 49 bis). PRERREG: ratio
    # mediano <= 0,80 y gana >= 2/3 => proyecto MoE vivo; si no, muerto por ~1 h-cuota.
    # v1 fallo: la imagen unsloth pinneada no reconoce qwen3_5_moe. v2: docker "" => imagen
    # ACTUAL de Kaggle (trae transformers con Qwen3.5); sin wheels de unsloth (aqui no se usan).
    "gate-moe": {"notebook": "notebooks/gate_moe.ipynb", "slug": "arc-agi2-gate-moe",
                 "title": "arc agi2 gate moe", "gpu": True,
                 "docker": "",
                 "defaults": {"model": ["qwen-lm/qwen-3/transformers/4b/1",
                                        "qwen-lm/qwen-3-5/transformers/qwen3.5-35b-a3b/1"]}},
    # BANCO ttt-2hip (8-sep), L4x4 shardeado. Diagnostico de TTT_PLAN 47: generamos UNA
    # hipotesis y la repetimos 16 veces (las 16 vistas son simetrias; en 6 de 12 tareas con
    # pool proponen una sola forma). Dos preguntas en un kernel:
    #  B) dos TTT con semillas de augmentacion distintas dan pools DIFERENTES? la UNION
    #     contiene la GT mas veces? PRERREG: union - max(A,B) >= 2 de 12 => injerto vivo;
    #     jaccard alto y union ~ A => convergen => idea muerta sin gastar slot.
    #  A) el re-scoring con 4 augs conserva el top-2 que con 8? PRERREG: >=90% => el ahorro
    #     de -4% de reloj es seguro y financia el +6,2% de la doble hipotesis (deja +2,2%).
    # MEMORIA: el snapshot del adaptador (~1 GB) va en RAM de CPU, no en GPU.
    "ttt-2hip": {"notebook": "notebooks/ttt_2hip.ipynb", "slug": "arc-agi2-ttt-2hip",
                 "title": "arc agi2 ttt 2hip", "gpu": True,
                 "docker": ("gcr.io/kaggle-private-byod/python@sha256:"
                            "320043e14c68293f1c946585b9257123385205a58af4b94b17d31868cae4e868"),
                 "defaults": {"kernel_source": ["sorokin/pip-install-unsloth-flash-patch"],
                              "model": ["sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"]}},
    # BANCO ttt-regla (7-sep), L4x4 shardeado. Propuesta del usuario: paso intermedio en
    # espacio de REJILLA (unica variante expresable con 16 tokens de vocabulario) y unica de la
    # familia del razonamiento que entra en presupuesto (+3-5% contra +4,8%). Antes de invertir
    # en un SFT de formato nuevo: ¿el modelo YA sabe aplicar la regla a una entrada nueva?
    # A = se retira el ultimo par de train, el TTT entrena con N-1, y se decodifica la entrada
    # retirada contra su salida conocida. B = la entrada de test contra la solucion del eval.
    # PRERREG: A-B >= 3 de 12 => la regla se aplica y falla la novedad del test => CERRAR la
    # rama; A ~ B => el cuello es aprender la regla => el paso intermedio puede ayudar.
    "ttt-regla": {"notebook": "notebooks/ttt_regla.ipynb", "slug": "arc-agi2-ttt-regla",
                  "title": "arc agi2 ttt regla", "gpu": True,
                  "docker": ("gcr.io/kaggle-private-byod/python@sha256:"
                             "320043e14c68293f1c946585b9257123385205a58af4b94b17d31868cae4e868"),
                  "defaults": {"kernel_source": ["sorokin/pip-install-unsloth-flash-patch"],
                               "model": ["sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"]}},
    # INJERTO r512 (6-sep): el rango del TTT SUBE de 256 a 512 (alpha 32 -> 45 deja la escala
    # rsLoRA en 1,989 contra 2,000, un 0,56%). Motivo: r64 = 29,72 vs 30,69 => menos capacidad
    # RESTA, luego 256 puede no ser el techo. Primera vez que la evidencia apunta a AMPLIFICAR
    # el TTT en vez de recortarlo (nottt 9,58 / ttt05 26,39 / ttt8 / r64: todo lo quitado fallo).
    # MEMORIA (medida, no supuesta): picos de inferencia 15597 MB a r=64 y 17109 a r=256 ->
    # ~19125 MB a r=512, o sea 18,7 GB de 24 con 22% de margen. Cabe, pero justo, y el decode
    # NO tenia proteccion contra OOM: se anaden dos guardas (TTT y decode) para que un pico
    # cueste un lote y no el SHARD ENTERO. Sin OOM el comportamiento es identico.
    # PRERREG: >=31,3 el rango sigue sumando => subir es la direccion y probar r=768;
    # 30,3-31,3 neutro => 256 es el techo, EJE CERRADO; <=30,0 pasarse resta (regularizacion
    # o inestabilidad) => 256 es un optimo interior y el eje queda cerrado por ambos lados.
    "lb33-r512": {"notebook": "notebooks/lb33_r512.ipynb", "slug": "arc-agi2-lb33-r512",
                  "title": "arc agi2 lb33 r512", "gpu": True,
                  "docker": ("gcr.io/kaggle-private-byod/python@sha256:"
                             "320043e14c68293f1c946585b9257123385205a58af4b94b17d31868cae4e868"),
                  "defaults": {"kernel_source": ["sorokin/pip-install-unsloth-flash-patch"],
                               "model": ["sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"]}},
    # INJERTO r64 (5-sep): UN solo cambio conceptual sobre lb33-det = el rango del TTT
    # 256 -> 64 (alpha 32 -> 16 para dejar la escala rsLoRA en 2,0, la de produccion).
    # Evidencia: banco ttt-rank pareado en 12 tareas -> 1,34x mas rapido (25,5% del TTT =
    # 8,9% del pipeline) sin degradacion detectable (media, mediana y test de signo).
    # embed_tokens/lm_head son [16,2560], rango efectivo min(r,16)=16 con 256 y con 64:
    # siguen en rango completo, por eso no hace falta rank_pattern.
    # PRERREG: >=31,3 el deadline muerde y el tiempo liberado se convierte en tareas
    # (adoptar); 30,3-31,3 neutro => el deadline NO muerde y el ahorro hay que GASTARLO
    # en mas vistas de decode, pero r=64 queda validado como financiacion; <=30,0 el
    # rango si importaba en produccion y la loss era mal proxy => revertir.
    "lb33-r64": {"notebook": "notebooks/lb33_r64.ipynb", "slug": "arc-agi2-lb33-r64",
                 "title": "arc agi2 lb33 r64", "gpu": True,
                 "docker": ("gcr.io/kaggle-private-byod/python@sha256:"
                            "320043e14c68293f1c946585b9257123385205a58af4b94b17d31868cae4e868"),
                 "defaults": {"kernel_source": ["sorokin/pip-install-unsloth-flash-patch"],
                              "model": ["sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"]}},
    # BARRIDO DEL RANGO DEL TTT (5-sep), L4x4 SHARDEADO x4. Solo fase de TTT, sin decode.
    # A r=256 el TTT entrena 529,8 M params y el 99,75% esta en las 7 proyecciones; embed/head
    # son 1,3 M y ya son rango completo (matrices [16,2560]). Por eso se barre el rango de las
    # proyecciones (256/64/16/4) con embed/head clavadas en 256. Control: alpha = 2*sqrt(r)
    # mantiene la escala rsLoRA en 2,0, si no se mediria escala en vez de capacidad.
    # PRERREG: candidato si el delta de loss pareado no se distingue de 0 (n=12) y el reloj
    # del TTT baja >=15%. La loss es proxy: verde aqui => confirmar en LB antes de injertar.
    "ttt-rank": {"notebook": "notebooks/ttt_rank.ipynb", "slug": "arc-agi2-ttt-rank",
                 "title": "arc agi2 ttt rank", "gpu": True,
                 "docker": ("gcr.io/kaggle-private-byod/python@sha256:"
                            "320043e14c68293f1c946585b9257123385205a58af4b94b17d31868cae4e868"),
                 "defaults": {"kernel_source": ["sorokin/pip-install-unsloth-flash-patch"],
                              "model": ["sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"]}},
    # BANCO DE DOSIS en el 4B (1-sep), L4x4 SHARDEADO x4. Prerreg: r(productividad@4,
    # beneficio 4->16) <= -0,6 con n=16 => disparador real; >= -0,3 => la rama muere.
    # El observable salio del smoke del 2B (r=-0,924, n=4) que REFUTO la dispersion del voto.
    # Decodifica 16 vistas UNA vez y persiste que produjo cada una: la curva sale offline.
    "augdose-4b": {"notebook": "notebooks/augdose_4b.ipynb", "slug": "arc-agi2-augdose-4b",
                   "title": "arc agi2 augdose 4b", "gpu": True,
                   "docker": ("gcr.io/kaggle-private-byod/python@sha256:"
                              "320043e14c68293f1c946585b9257123385205a58af4b94b17d31868cae4e868"),
                   "defaults": {"kernel_source": ["sorokin/pip-install-unsloth-flash-patch"],
                                "model": ["sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"]}},
    # GRAFT ttt05: campeon con num_train_epochs=0.5 en el TTT (UN numero). Evidencia del banco
    # 4B limpio (22-ago): 48 pasos = peor politica (regret 94,5 vs 37,9 en 12); el sobreajuste
    # del TTT destruye el ajuste a la GT en 8/20; mitad del reloj de TTT -> cobertura (~17% del
    # total). Prerreg: >=30,14 adoptar / 29-30,14 ruido / <29 revertir (la adaptacion reducida
    # duele a escala). Smoke gate: reload >=2,5. LANZAR CON --accelerator NvidiaL4.
    "lb33-ttt05": {"notebook": "notebooks/lb33_ttt05.ipynb", "slug": "arc-agi2-lb33-ttt05",
                   "title": "arc agi2 lb33 ttt05", "gpu": True,
                   "docker": ("gcr.io/kaggle-private-byod/python@sha256:"
                              "320043e14c68293f1c946585b9257123385205a58af4b94b17d31868cae4e868"),
                   "defaults": {"kernel_source": ["sorokin/pip-install-unsloth-flash-patch"],
                                "model": ["sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"]}},
    # LOO EARLY-STOPPING (paso siguiente del triage): ¿la NLL de una demo APARTADA (observable)
    # predice el punto optimo de parada del TTT (oraculo)? 20 tareas, curvas cada 6 pasos,
    # PRIMER kernel shardeado en 4 GPUs (patron plantilla_kernel_4gpu). ~15 min de pared.
    "ttt-loo": {"notebook": "notebooks/ttt_loo.ipynb", "slug": "arc-agi2-ttt-loo",
                "title": "arc agi2 ttt loo", "gpu": True,
                "defaults": {"model": ["sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"]}},
    # TRIAGE del TTT en el 4B (post-veredicto nottt=9,58): TTT fiel r=256+embed/head+rsLoRA
    # pareado base/TTT sobre 20 tareas eval (incl. las 3 que la base resuelve = riesgo de
    # muerte). Produce: categorias mantiene/MATA/RESCATA, curvas de convergencia (pasos
    # 12/24/36/48), vectores por celda pre/post (picos), y POOLS VOLCADOS para el verificador.
    # LANZAR CON --accelerator NvidiaL4. ~60-90 min.
    "ttt-triage": {"notebook": "notebooks/ttt_triage.ipynb", "slug": "arc-agi2-ttt-triage",
                   "title": "arc agi2 ttt triage", "gpu": True,
                   "defaults": {"model": ["sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"]}},
    # DESCUENTO DE PICOS en la aceptacion del DFS (4B, eval): la anatomia (dfs-budget) mostro
    # que la NLL de la GT es llanura + 2-5 picos (mediana: 50% en 2 celdas, 82% en top-5) y el
    # presupuesto lineal fue nulo (3->3). Regla: (suma - top3 picos) < 1.609, paso < 2.0, cota
    # monotona (poda intacta). Guarda vectores por celda COMPLETOS -> reglas futuras offline.
    # LANZAR CON --accelerator NvidiaL4. ~30-60 min.
    "dfs-spikes": {"notebook": "notebooks/dfs_spikes.ipynb", "slug": "arc-agi2-dfs-spikes",
                   "title": "arc agi2 dfs spikes", "gpu": True,
                   "defaults": {"model": ["sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"]}},
    # UMBRAL POR POSICION en el DFS (4B, eval): tau(pos)=max(1.609, 0.0202*pos) + anatomia por
    # celda de la NLL de la GT + barrido estructural de b. Ataca GT-en-pool (0,42, la palanca
    # x2): con el umbral fijo 37/40 GT quedan fuera, 9 con confianza/celda >=98%.
    # LANZAR CON --accelerator NvidiaL4. ~25-45 min (presupuesto 60s/tarea).
    "dfs-budget": {"notebook": "notebooks/dfs_budget.ipynb", "slug": "arc-agi2-dfs-budget",
                   "title": "arc agi2 dfs budget", "gpu": True,
                   "defaults": {"model": ["sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"]}},
    # PRECISION del 4B (banco tier1) en bf16 nativo de L4: cierra la casilla vacia del argumento
    # del umbral del DFS (el log del campeon solo valida recall). Ids de eval embebidos (offline).
    # GATE de 30s al frente. LANZAR CON --accelerator NvidiaL4. ~20-30 min.
    "tier1-4b": {"notebook": "notebooks/tier1_4b.ipynb", "slug": "arc-agi2-tier1-4b",
                 "title": "arc agi2 tier1 4b", "gpu": True,
                 "defaults": {"model": ["sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"]}},
    # Descarga BARC-8B desde HF (repo abierto) a su output; su output se adjunta luego al probe
    # como kernel_source (evita la copia gated de ironbar). Internet ON -> T4, sin docker pin.
    "barc-download": {"notebook": "notebooks/barc_download.ipynb", "slug": "arc-agi2-barc-download",
                      "title": "arc agi2 barc download", "gpu": True},
    # Descarga un slice de la data de induccion BARC desde HF (para de-risk del SFT). T4+internet.
    "barc-data-dl": {"notebook": "notebooks/barc_data_dl.ipynb", "slug": "arc-agi2-barc-data-dl",
                     "title": "arc agi2 barc data dl", "gpu": False},
    # Fase B: data-gen compositiva (componer seeds -> tareas AGI-2 de 2 pasos verificadas).
    # v2-L4: seeds EMBEBIDOS (sin internet) -> corre en L4x4 y aprovecha sus vCPUs (el kernel
    # CPU solo tiene 4). Gasta ~0.5-1h de quota pero desbloquea el tren SFT->probe horas antes.
    "barc-datagen": {"notebook": "notebooks/barc_datagen.ipynb", "slug": "arc-agi2-barc-datagen",
                     "title": "arc agi2 barc datagen", "gpu": True},
    # Probe PAREADO base-vs-SFT (mismas tareas, metrica densa, MAX_NEW=2048). Imagen PINNEADA:
    # la "latest" volvio a romper peft (torchao 0.10) tambien en PeftModel.from_pretrained.
    "barc-probe-paired": {"notebook": "notebooks/barc_probe_paired.ipynb",
                          "slug": "arc-agi2-barc-probe-paired", "title": "arc agi2 barc probe paired",
                          "gpu": True,
                          "defaults": {"kernel_source": ["juliancamilovilla/arc-agi2-barc-download",
                                                         "juliancamilovilla/arc-agi2-barc-sft-c"]}},
    # Fase C.2: probe del modelo SFT'd (base + adapter de la Fase C) -> A/B del hit rate vs 6.25%.
    "barc-probe-lora": {"notebook": "notebooks/barc_probe_lora.ipynb",
                        "slug": "arc-agi2-barc-probe-lora", "title": "arc agi2 barc probe lora",
                        "gpu": True,
                        "defaults": {"kernel_source": ["juliancamilovilla/arc-agi2-barc-download",
                                                       "juliancamilovilla/arc-agi2-barc-sft-c"]}},
    # Fase C: SFT REAL de la induccion sobre la data compositiva (Fase B) + mezcla con la original.
    # docker: usa la imagen PINNEADA (DOCKER_IMAGE). La "latest" de Kaggle roto a py3.12 con
    # torchao 0.10 incompatible con peft (>0.16) -> ImportError al aplicar LoRA.
    "barc-sft-c": {"notebook": "notebooks/barc_sft_c.ipynb", "slug": "arc-agi2-barc-sft-c",
                   "title": "arc agi2 barc sft c", "gpu": True,
                   "defaults": {"kernel_source": ["juliancamilovilla/arc-agi2-barc-download",
                                                  "juliancamilovilla/arc-agi2-barc-data-dl",
                                                  "juliancamilovilla/arc-agi2-barc-datagen"]}},
    # Fase A: SFT LoRA del BARC-8B sobre el slice de data (de-risk de la infra). Imagen BYOD
    # (unsloth+trl); adjunta el modelo (barc-download) y la data (barc-data-dl) como kernel_sources.
    "barc-sft": {"notebook": "notebooks/barc_sft.ipynb", "slug": "arc-agi2-barc-sft",
                 "title": "arc agi2 barc sft", "gpu": True,
                 # imagen ESTANDAR (docker="") - transformers+peft, captura logs (la BYOD los vacia)
                 "docker": "",
                 "defaults": {"kernel_source": ["juliancamilovilla/arc-agi2-barc-download",
                                                "juliancamilovilla/arc-agi2-barc-data-dl"]}},
    # PROBE de sintesis de programas (FASE 0): BARC-8B (4-bit) sobre eval, mide verifiable-hit
    # rate + generalizacion. Save&Run (no submission). Imagen BYOD (transformers+bitsandbytes).
    "barc-probe": {"notebook": "notebooks/barc_probe.ipynb", "slug": "arc-agi2-barc-probe",
                   "title": "arc agi2 barc probe", "gpu": True,
                   # imagen ESTANDAR de Kaggle (docker="") — trae bitsandbytes+transformers para el
                   # 4-bit; la BYOD de unsloth NO tiene bitsandbytes. El probe no necesita unsloth.
                   "docker": "",
                   # copia propia del BARC (output del kernel de descarga) — evita la gated de ironbar
                   "defaults": {"kernel_source": ["juliancamilovilla/arc-agi2-barc-download"]}},
    # Ensemble de SELECCION (calidad, sin costo de tiempo): campeon + attempt_2 por confianza
    # (probmul) cuando discrepa del consenso (kgmon). Founded en trace-judging (arxiv 2606.31543).
    "lb33-selens": {"notebook": "notebooks/lb33_selens.ipynb", "slug": "arc-agi2-lb33-selens",
                    "title": "arc agi2 lb33 selens", "gpu": True,
                    "docker": ("gcr.io/kaggle-private-byod/python@sha256:"
                               "320043e14c68293f1c946585b9257123385205a58af4b94b17d31868cae4e868"),
                    "defaults": {"kernel_source": ["sorokin/pip-install-unsloth-flash-patch"],
                                 "model": ["sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"]}},
    # 2º modelo NVARC (Qwen3-2B) con la config campeona (cheap-first) — miembro de ensemble.
    "lb33-2b": {"notebook": "notebooks/lb33_2b.ipynb", "slug": "arc-agi2-lb33-2b",
                "title": "arc agi2 lb33 2b", "gpu": True,
                "docker": ("gcr.io/kaggle-private-byod/python@sha256:"
                           "320043e14c68293f1c946585b9257123385205a58af4b94b17d31868cae4e868"),
                "defaults": {"kernel_source": ["sorokin/pip-install-unsloth-flash-patch"],
                             "model": ["sorokin/qwen3_2b_grids15_sft141/Transformers/bfloat16/1"]}},
    # Ensemble 4B+2B neutro en cobertura (4B campeón, luego 2B en sobrante). Ambos modelos.
    "lb33-ensemble": {"notebook": "notebooks/lb33_ensemble.ipynb", "slug": "arc-agi2-lb33-ensemble",
                      "title": "arc agi2 lb33 ensemble", "gpu": True,
                      "docker": ("gcr.io/kaggle-private-byod/python@sha256:"
                                 "320043e14c68293f1c946585b9257123385205a58af4b94b17d31868cae4e868"),
                      "defaults": {"kernel_source": ["sorokin/pip-install-unsloth-flash-patch"],
                                   "model": ["sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1",
                                             "sorokin/qwen3_2b_grids15_sft141/Transformers/bfloat16/1"]}},
    # Etapa B de SPEED_PLAN: smoke instrumentado del campeon (12 tareas por el espectro de
    # coste, split TTT/decode/scoring por tarea). Mismo entorno del campeon.
    "lb33-timing": {"notebook": "notebooks/lb33_timing.ipynb", "slug": "arc-agi2-lb33-timing",
                    "title": "arc agi2 lb33 timing", "gpu": True,
                    "docker": ("gcr.io/kaggle-private-byod/python@sha256:"
                               "320043e14c68293f1c946585b9257123385205a58af4b94b17d31868cae4e868"),
                    "defaults": {"kernel_source": ["sorokin/pip-install-unsloth-flash-patch"],
                                 "model": ["sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"]}},
    # D4 (SCALE_PLAN): SFT LoRA del 4B campeon sobre ARC-Heavy (snapshot 19K, dataset privado).
    # Imagen PINNEADA (default, gate peft/torchao). Lanzar con --accelerator NvidiaL4 (offline).
    # Salida: adapter_4b + merged_4b (para el kernel de inferencia lb33-scale).
    "scale-sft-4b": {"notebook": "notebooks/scale_sft_4b.ipynb", "slug": "arc-agi2-scale-sft-4b",
                     "title": "arc agi2 scale sft 4b", "gpu": True,
                     "defaults": {"model": ["sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"],
                                  "dataset": ["juliancamilovilla/arcscale-archeavy-19k"]}},
    # Inferencia D4: campeon (lb33_graft) con el 4B FUSIONADO de scale-sft-4b (unico cambio =
    # ruta del modelo, resuelta por os.walk; sin fallback). SIN model source: el modelo viene
    # del output del kernel de SFT (kernel_source). Imagen BYOD del campeon.
    "lb33-scale": {"notebook": "notebooks/lb33_scale.ipynb", "slug": "arc-agi2-lb33-scale",
                   "title": "arc agi2 lb33 scale", "gpu": True,
                   "docker": ("gcr.io/kaggle-private-byod/python@sha256:"
                              "320043e14c68293f1c946585b9257123385205a58af4b94b17d31868cae4e868"),
                   "defaults": {"kernel_source": ["sorokin/pip-install-unsloth-flash-patch",
                                                  "juliancamilovilla/arc-agi2-scale-sft-4b"]}},
    # ENSAYO escala en el test oculto: lb33_2b (config campeona, baseline 18.33 del 15-jul)
    # con el 2B FUSIONADO con ARC-Heavy (dataset arcscale-merged-2b). Commit barato: solo 4
    # tareas smoke (rerun_mode hace el trabajo real gratis en el scoring). SIN model source.
    "lb33-2b-scale": {"notebook": "notebooks/lb33_2b_scale.ipynb", "slug": "arc-agi2-lb33-2b-scale",
                      "title": "arc agi2 lb33 2b scale", "gpu": True,
                      "docker": ("gcr.io/kaggle-private-byod/python@sha256:"
                                 "320043e14c68293f1c946585b9257123385205a58af4b94b17d31868cae4e868"),
                      "defaults": {"kernel_source": ["sorokin/pip-install-unsloth-flash-patch"],
                                   "dataset": ["juliancamilovilla/arcscale-merged-2b"]}},
    # Restauracion de cobertura del fusionado (mecanismo del 27.08): merged + ttt8 + scoring4,
    # ambos grafts pre-medidos. Mismo entorno que lb33-scale.
    "lb33-scale-fast": {"notebook": "notebooks/lb33_scale_fast.ipynb",
                        "slug": "arc-agi2-lb33-scale-fast", "title": "arc agi2 lb33 scale fast",
                        "gpu": True,
                        "docker": ("gcr.io/kaggle-private-byod/python@sha256:"
                                   "320043e14c68293f1c946585b9257123385205a58af4b94b17d31868cae4e868"),
                        "defaults": {"kernel_source": ["sorokin/pip-install-unsloth-flash-patch",
                                                       "juliancamilovilla/arc-agi2-scale-sft-4b"]}},
    # Ensemble ARQUITECTÓNICAMENTE diverso: Qwen 4B (cheap-first, cap ~10h) + TRM recursivo
    # (épocas reducidas ~1.5h) + merge pass@2. Un kernel, imagen BYOD del Qwen (TRM corre ahí).
    # Adjunta el Qwen 4B (model), unsloth (kernel_source) y los 4 datasets del TRM.
    "lb33-ensemble-trm": {"notebook": "notebooks/lb33_ensemble_trm.ipynb",
                          "slug": "arc-agi2-lb33-ensemble-trm", "title": "arc agi2 lb33 ensemble trm",
                          "gpu": True,
                          "docker": ("gcr.io/kaggle-private-byod/python@sha256:"
                                     "320043e14c68293f1c946585b9257123385205a58af4b94b17d31868cae4e868"),
                          "defaults": {"kernel_source": ["sorokin/pip-install-unsloth-flash-patch"],
                                       "model": ["sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"],
                                       "dataset": ["chanhainguyen/trm-code",
                                                   "cpmpml/arc-prize-trm-031",
                                                   "limyenwee/arc-agi2-trm-pydeps",
                                                   "juliancamilovilla/trm-extra-whls"]}},
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("kernel", choices=list(KERNELS))
    ap.add_argument("--code-dataset", help="slug user/dataset con src/ y scripts/")
    ap.add_argument("--dataset", action="append", default=[],
                    help="dataset adicional (p.ej. modelo entrenado); repetible")
    ap.add_argument("--model", action="append", default=[],
                    help="Kaggle Models source user/model/...; repetible")
    ap.add_argument("--kernel-source", action="append", default=[],
                    help="kernel adjunto (p.ej. wheels offline user/kernel); repetible")
    ap.add_argument("--internet", action="store_true",
                    help="habilita internet (SOLO T4x2/P100; L4x4 lo prohibe)")
    ap.add_argument("--accelerator", default="NvidiaTeslaT4",
                    choices=["NvidiaTeslaT4", "NvidiaTeslaP100", "NvidiaL4", "Tpu1VmV38"],
                    help="GPU/TPU. NvidiaL4 = L4x4 (96GB, 4 GPUs; quota al DOBLE, "
                         "internet OFF obligatorio). Tag verificado via kernels pull -m.")
    ap.add_argument("--docker-image", default=DOCKER_IMAGE,
                    help="imagen del kernel; '' = dejar que Kaggle asigne la última")
    ap.add_argument("--dry-run", action="store_true", help="solo genera metadata, no publica")
    args = ap.parse_args()

    load_env(ROOT / ".env")
    if "kaggle_username" in os.environ:
        os.environ["KAGGLE_USERNAME"] = os.environ["kaggle_username"]

    cfg = KERNELS[args.kernel]
    user = kaggle_username()
    defaults = cfg.get("defaults", {})
    datasets = list(args.dataset) or list(defaults.get("dataset", []))
    if args.code_dataset:
        datasets.insert(0, args.code_dataset)

    # defaults por-kernel (p.ej. lb33 trae su kernel de wheels y su modelo)
    models = args.model or defaults.get("model", [])
    kernel_sources = args.kernel_source or defaults.get("kernel_source", [])
    docker = args.docker_image if args.docker_image != DOCKER_IMAGE else cfg.get("docker", DOCKER_IMAGE)

    meta = {
        "id": f"{user}/{cfg['slug']}",
        "title": cfg["title"],
        "code_file": Path(cfg["notebook"]).name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": cfg["gpu"],
        "machine_shape": args.accelerator,
        "enable_internet": args.internet,
        "dataset_sources": datasets,
        "competition_sources": [COMP],
        "model_sources": models,
        "kernel_sources": kernel_sources,
    }
    if docker:
        meta["docker_image"] = docker

    # ── AVISOS DE CUOTA (reglas del usuario, ver CLAUDE.md) ─────────────────────────
    # La L4x4 quema cuota a TARIFA 2x del wall-clock y son 4 GPUs que se pagan enteras:
    # los kernels de banco DEBEN shardear en las 4 (patron: scripts/plantilla_kernel_4gpu.py).
    if args.accelerator == "NvidiaL4":
        nb_txt = (ROOT / cfg["notebook"]).read_text(encoding="utf-8")
        print("\n" + "=" * 76)
        print("⚠  L4x4: la cuota corre a 2x del wall-clock POR LA MAQUINA COMPLETA (4 GPUs).")
        if "CUDA_VISIBLE_DEVICES" not in nb_txt and args.kernel not in (
                "lb33", "lb33-graft", "lb33-cap", "lb33-roles", "lb33-nottt",
                "lb33-selens", "lb33-2b"):
            print("⚠  ESTE NOTEBOOK PARECE MONO-GPU (no menciona CUDA_VISIBLE_DEVICES):")
            print("   pagaras 4x de mas. Shardea con el patron de")
            print("   scripts/plantilla_kernel_4gpu.py salvo razon explicita.")
        print("   Tope semanal del proyecto: 10 h-quota; ledger SIEMPRE en horas-quota.")
        print("=" * 76 + "\n")

    tmp = Path(tempfile.mkdtemp(prefix="arc_kernel_"))
    shutil.copy(ROOT / cfg["notebook"], tmp / meta["code_file"])
    (tmp / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("metadata:\n" + json.dumps(meta, indent=2))

    if args.dry_run:
        print(f"\n[dry-run] carpeta lista en {tmp}")
        return 0

    print(f"\nPublicando kernel '{meta['id']}' ...")
    # PYTHONUTF8=1: el CLI de Kaggle en Windows lee el ipynb con cp1252 y CHOCA con bytes
    # UTF-8 (acentos/guiones de los comentarios) -> falla en silencio (exit!=0, sin mensaje).
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    r = subprocess.run(["kaggle", "kernels", "push", "-p", str(tmp)], env=env,
                       capture_output=True, text=True)
    out = (r.stdout or "") + (r.stderr or "")
    print(out.strip())
    # El CLI devuelve rc=0 incluso en "Kernel push error: Maximum weekly GPU quota..."
    # (medido 2026-08-13) — detectar el fallo por el TEXTO, no por el returncode.
    if r.returncode != 0 or "error" in out.lower():
        print(f"\n[ERROR] push FALLÓ (rc={r.returncode}) — la versión committeada NO cambió.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
