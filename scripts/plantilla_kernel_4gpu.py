"""PLANTILLA OBLIGATORIA para kernels de banco en la L4x4 de Kaggle.

REGLA (usuario, 2026-08-22, ver CLAUDE.md): la L4x4 son 4 GPUs independientes (4x24 GB) y la
cuota corre a 2x del wall-clock POR LA MAQUINA COMPLETA, uses 1 GPU o 4. Un banco mono-GPU
paga 4x de mas. Este es el patron del kernel campeon aplicado al banco: el proceso principal
escribe el codigo del worker, spawnea 4 (uno por GPU), cada worker procesa el shard
KEYS[i::4] y escribe su JSON; el main espera, funde y publica el resumen.

Uso en un notebook de una celda:
    CODE_WORKER = r'''<todo el codigo del banco, parametrizado por SHARD>'''
    open("/kaggle/working/worker.py", "w").write(CODE_WORKER)
    <bloque LANZADOR de abajo>

Memoria por GPU (medida): 4B bf16 7,3 GB + adapter TTT r=256 fp32 ~2,1 GB + AdamW8bit
~0,6 GB + gradientes ~2,1 GB + activaciones -> ~13-14 GB de los 24 GB. Cabe una copia
completa por GPU; NO compartir modelo entre procesos.
"""

# ───────────────────────────── BLOQUE LANZADOR (main) ─────────────────────────────
LANZADOR = r'''
import json
import os
import subprocess
import sys
import time

N_GPUS = 4
T0 = time.time()
procs = []
for i in range(N_GPUS):
    env = dict(os.environ, SHARD=str(i), N_SHARDS=str(N_GPUS),
               CUDA_VISIBLE_DEVICES=str(i), PYTHONUNBUFFERED="1")
    p = subprocess.Popen([sys.executable, "/kaggle/working/worker.py"],
                         stdout=open(f"/kaggle/working/shard{i}.log", "w"),
                         stderr=subprocess.STDOUT, env=env)
    procs.append(p)
    print(f"[main] worker {i} lanzado (pid {p.pid})", flush=True)

for i, p in enumerate(procs):
    rc = p.wait()
    print(f"[main] worker {i} termino rc={rc} ({time.time()-T0:.0f}s)", flush=True)
    if rc != 0:                       # el log del shard es lo primero que hay que leer
        print(open(f"/kaggle/working/shard{i}.log", errors="replace").read()[-2000:],
              flush=True)

tareas = []
for i in range(N_GPUS):
    f = f"/kaggle/working/shard{i}.json"
    if os.path.exists(f):
        tareas += json.load(open(f))["tareas"]
json.dump({"tareas": tareas}, open("/kaggle/working/resultado.json", "w"))
print(f"[main] {len(tareas)} tareas fundidas en {time.time()-T0:.0f}s", flush=True)
print("RESULT_JSON:" + json.dumps({"n": len(tareas)}), flush=True)
'''

# ─────────────────────── ESQUELETO DEL WORKER (worker.py) ───────────────────────
# Dentro del codigo del banco, los unicos cambios respecto a la version mono-GPU son:
#
#   SHARD = int(os.environ.get("SHARD", "0"))
#   N_SHARDS = int(os.environ.get("N_SHARDS", "1"))
#   MIS_KEYS = KEYS[SHARD::N_SHARDS]          # <- el shard de este worker
#   ...
#   # device_map="cuda:0" NO cambia: CUDA_VISIBLE_DEVICES ya aisla la GPU i como cuda:0
#   ...
#   json.dump(RES, open(f"/kaggle/working/shard{SHARD}.json", "w"))   # persistir POR TAREA
#
# Reglas que NO se relajan en paralelo:
#   - gate de carga al frente de CADA worker (30 s; aborta el shard, no la maquina)
#   - persistencia incremental tras cada tarea (un shard caido conserva lo suyo)
#   - seeds deterministas por tarea (seed = base + indice GLOBAL de la tarea, no local)
