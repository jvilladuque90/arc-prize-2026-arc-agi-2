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
