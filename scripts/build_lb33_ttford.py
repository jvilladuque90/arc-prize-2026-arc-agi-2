import io, json, os
os.chdir(r"C:\Users\Usuario\Documents\arc-prize-2026-arc-agi-2")

nb = json.load(open("notebooks/lb33_det.ipynb", encoding="utf-8"))
s3 = "".join(nb["cells"][3]["source"])
s5 = "".join(nb["cells"][5]["source"])

# ---------------------------------------------------------------- 1) shuffle_ex
old = """    def shuffle_ex(self, perm=None, keep_max=None):
        new_keys = []
        new_queries = {}
        new_replies = {}
        for key in self.keys:
            n = len(self.queries[key]['train'])
            p = np.random.permutation(n) if perm is None else perm
"""
new = """    def shuffle_ex(self, perm=None, keep_max=None, ultima_mayor=False):
        new_keys = []
        new_queries = {}
        new_replies = {}
        for key in self.keys:
            n = len(self.queries[key]['train'])
            if perm is not None:
                p = perm
            elif ultima_mayor and n > 1:
                # INJERTO ttford: orden del contexto ALEATORIO (se conserva la diversidad de
                # las 128 secuencias) pero la demostracion de SALIDA MAYOR queda SIEMPRE la
                # ultima. Asi cada secuencia del TTT predice la salida mas grande teniendo
                # como contexto solo las mas pequenas: un ensayo de extrapolacion.
                # Medido 9-sep: el campeon saca 75% de GT-en-pool cuando la salida cae DENTRO
                # de la envolvente de las salidas demo y 11% cuando cae fuera (82% del eval).
                # Coste de reloj: CERO (mismos tokens, mismas 128 secuencias).
                _ex = self.queries[key]['train']
                _ar = [len(_ex[i]['output']) * len(_ex[i]['output'][0]) for i in range(n)]
                _big = int(np.argmax(_ar))
                _rnd = np.random.permutation(n).tolist()
                p = np.array([i for i in _rnd if i != _big] + [_big])
            else:
                p = np.random.permutation(n)
"""
assert old in s3, "shuffle_ex no encontrado"
s3 = s3.replace(old, new, 1)

# ---------------------------------------------------------------- 2) augment
old = """    def augment(self, n=1, shfl_keys=False, seed=42):
        np.random.seed(seed)
        d = self
        d = d.mod(np.transpose, keep=True)
        d = d.mod(np.rot90, n=3, keep=True)
        d = d.mod(permute_mod, permute_rnd_all_, n=n, shuffle=shfl_keys, keep=False)
        d = d.shuffle_ex()
        return d"""
new = """    def augment(self, n=1, shfl_keys=False, seed=42, ultima_mayor=False):
        np.random.seed(seed)
        d = self
        d = d.mod(np.transpose, keep=True)
        d = d.mod(np.rot90, n=3, keep=True)
        d = d.mod(permute_mod, permute_rnd_all_, n=n, shuffle=shfl_keys, keep=False)
        d = d.shuffle_ex(ultima_mayor=ultima_mayor)
        return d"""
assert old in s3, "augment no encontrado"
s3 = s3.replace(old, new, 1)

# ---------------------------------------------------------------- 3) call site (SOLO el TTT)
old = "        train_ds = puzzle_ds.augment(n=16, shfl_keys=True, seed=1)"
new = ("        # INJERTO ttford: unico cambio respecto al campeon lb33-det. Solo el dataset de\n"
       "        # TTT; el de decode (eval_ds) queda EXACTAMENTE igual.\n"
       "        train_ds = puzzle_ds.augment(n=16, shfl_keys=True, seed=1, ultima_mayor=True)")
assert old in s5, "call site no encontrado"
assert s5.count(old) == 1
s5 = s5.replace(old, new, 1)

# el dataset de decode NO se toca
assert "eval_ds = puzzle_ds_multi.augment(n=2, seed=2)" in s5
assert "ultima_mayor" not in s5.split("eval_ds =")[1][:200]

nb["cells"][3]["source"] = s3.splitlines(keepends=True)
nb["cells"][5]["source"] = s5.splitlines(keepends=True)
md = ("# lb33-ttford — el TTT ensaya la extrapolacion de salida\n\n"
      "UN solo injerto sobre el campeon `lb33-det` (30,69 +/- 0,415).\n\n"
      "**Mecanismo medido el 9-sep** sobre 23 tareas de eval con resultado del campeon: la `d` de\n"
      "la ENTRADA de test no predice nada (Spearman +0,004), pero la de la SALIDA si (-0,234).\n"
      "Tasa condicionada: **75%** de GT-en-pool cuando la salida cae DENTRO de la envolvente de las\n"
      "salidas demo (18% de la poblacion) frente a **11%** cuando cae fuera (82%).\n\n"
      "Hoy **todos los gradientes del TTT apuntan a salidas mas pequenas** que la exigida: las\n"
      "salidas demo lo son en el 94,7% de las tareas. Este injerto pone la demostracion de salida\n"
      "MAYOR siempre la ultima, manteniendo el resto del orden aleatorio, de modo que cada una de\n"
      "las 128 secuencias del TTT predice la salida mas grande con solo las pequenas de contexto.\n\n"
      "**Coste de reloj: CERO.** Mismos tokens, mismas 128 secuencias, mismo decode (el `eval_ds`\n"
      "no se toca). Es reordenacion, no gasto -- lo que importa con el campeon al 95% del deadline.\n\n"
      "**PRERREG** (banda de +/-2 sigma alrededor de 30,69 con sigma=0,415 = [29,86 - 31,52]):\n"
      "`>31,52` adoptar | `29,86-31,52` ruido, no concluyente | `<29,86` revertir.\n")
nb["cells"][0]["source"] = md.splitlines(keepends=True)
json.dump(nb, open("notebooks/lb33_ttford.ipynb", "w", encoding="utf-8"), indent=1, ensure_ascii=False)

# ---------------------------------------------------------------- registro
p = "scripts/push_kernels.py"
s = io.open(p, encoding="utf-8").read()
anchor = '    # lb33-stack = TRATAMIENTO'
entry = '''    # lb33-ttford (10-sep) = campeon lb33-det + UN injerto: en el dataset de TTT la demostracion
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
'''
if '"lb33-ttford"' not in s:
    assert anchor in s
    io.open(p, "w", encoding="utf-8").write(s.replace(anchor, entry + anchor, 1))
    print("registrado lb33-ttford")
else:
    print("lb33-ttford ya estaba")

# ---------------------------------------------------------------- diff frente al campeon
base = json.load(open("notebooks/lb33_det.ipynb", encoding="utf-8"))
dif = [i for i in range(len(nb["cells"]))
       if "".join(nb["cells"][i]["source"]) != "".join(base["cells"][i]["source"])]
print("celdas que difieren del campeon:", dif, "(0=markdown, 3=arc_loader, 5=arc_solver)")
print("OK notebooks/lb33_ttford.ipynb")
