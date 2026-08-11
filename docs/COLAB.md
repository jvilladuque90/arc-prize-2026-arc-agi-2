# Colab como banco de pruebas gratuito (T4, ~2h/sesión)

Complemento a la quota de Kaggle (~30h/semana de L4): sesiones T4 gratis para **experimentos
proxy** que no la gastan. Ventajas de Colab sobre los kernels offline de Kaggle: **internet +
pip libre** (bitsandbytes funciona → 4-bit posible) y iteración inmediata.

## Qué transfiere y qué no (medido)

- ❌ **Scores absolutos NO**: los modelos <4B dan ~0 exactas en ARC (medido: 0.5B y 3B → cero).
  Un proxy pequeño no predice el score del leaderboard.
- ✅ **Señales densas y relativas SÍ**: NLL teacher-forced antes/después del TTT, cell-accuracy,
  tiempos por fase, parse/exec rates, efectos de distribución de data (el desastre del SFT
  compositivo — estilo imitado, longitud 2.5×, exec rate a la mitad — se habría visto en 30 min
  de Colab con un modelo pequeño ANTES de gastar GPU de Kaggle).
- **Regla**: Colab = pre-filtro barato de hipótesis; Kaggle L4 = instrumento confirmatorio.

## ⚡ CLI OFICIAL de Google (junio 2026) — ejecución HEADLESS operativa

Google lanzó el **CLI oficial** (`pip install google-colab-cli`, Apache 2.0) que SÍ ejecuta
headless: `colab run --gpu T4 script.py` = aprovisionar VM fresca → ejecutar → liberar, todo
desde el terminal (diseñado para agentes; trae `COLAB_SKILL.md` embebido). **Validado
end-to-end en este proyecto (2026-08-11).**

Setup que FUNCIONA en Windows (el CLI dice "Linux/macOS only"; estos 2 fixes lo arreglan):
1. `pip install google-colab-cli gitpython` + **pin** `pip install jupyter-kernel-client==0.15.0`
   (la 1.0.1 renombró `KernelClient` → rompe el CLI).
2. **Parche termios** (consola interactiva Unix-only): en
   `site-packages/colab_cli/console.py`, envolver `import termios`/`import tty` en
   `try/except ImportError` con `termios = None` / `tty = None` (solo afecta a `colab console`,
   que no usamos).
3. Auth ADC (una vez, browser): `gcloud auth application-default login --scopes=openid,`
   `https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email,`
   `https://www.googleapis.com/auth/colaboratory` (los 4 scopes son obligatorios).

Uso: `colab --auth=adc run --gpu T4 scripts/mi_experimento.py [args]` (stdout del script →
stdout; el chatter `[colab]` → stderr). `colab sessions` lista; `colab stop -s N` libera;
**sesiones sin stop queman compute units** (`run` sin `--keep` se auto-limpia).

## Flujo de trabajo (alternativa manual, notebooks)

`colab-cli` (instalado: `pip install colab-cli gitpython`) **sincroniza y abre** notebooks en
Colab vía Google Drive — **no ejecuta headless** (Colab gratis exige "Run All" manual en el
navegador). Setup único de OAuth:

1. Google Cloud Console → crear proyecto → habilitar Drive API → credenciales OAuth
   ("Desktop app") → descargar `client_secrets.json`.
2. `colab-cli set-config client_secrets.json` y `colab-cli set-auth-user 0`.
3. Uso: `colab-cli open-nb notebooks/colab_ttt_ablation.ipynb` (sube y abre en el navegador) ·
   `colab-cli pull-nb …` (baja el notebook ejecutado con outputs).

**Ruta alternativa sin OAuth** (más simple): abrir directo desde GitHub —
`https://colab.research.google.com/github/<user>/<repo>/blob/main/notebooks/<nb>.ipynb`
(para repos privados, autorizar GitHub en Colab una vez). Los resultados se descargan como JSON
al final del notebook (`files.download`).

En Colab: Runtime → Change runtime type → **T4 GPU**, y añadir el secreto `KAGGLE_API_TOKEN`
(icono 🔑 Secrets) para que el notebook descargue data/modelos de Kaggle.

## Experimentos proxy disponibles

| Notebook | Pregunta que responde | Duración T4 |
|---|---|---|
| `scripts/colab_ttt_ablation.py` (headless via `colab run`) o `colab_ttt_ablation.ipynb` (manual) | ¿Cuánta calidad de adaptación pierde TTT con n_aug=8 vs 16, y cuánto tiempo ahorra? (guía del injerto `ttt8`) — NVARC 2B, 12 tareas holdout, diseño pareado, métricas densas (ΔNLL, cell-acc, tiempo) | ~75 min |

Los resultados (`*_result.json`) se pegan en la conversación o se commitean a `docs/` para
analizarlos y decidir el gasto de quota de Kaggle.
