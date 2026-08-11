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

## Flujo de trabajo

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
| `colab_ttt_ablation.ipynb` | ¿Cuánta calidad de adaptación pierde TTT con n_aug=8 vs 16, y cuánto tiempo ahorra? (guía del injerto `ttt8`) — NVARC 2B, 12 tareas holdout, diseño pareado, métricas densas (ΔNLL, cell-acc, tiempo) | ~75 min |

Los resultados (`*_result.json`) se pegan en la conversación o se commitean a `docs/` para
analizarlos y decidir el gasto de quota de Kaggle.
