"""
Actualiza todas las rutas de guardado de graficas en el notebook
de 'grafica_X_*.png' a 'assets/imagenes/grafica_X_*.png'
"""
import json, re

NOTEBOOK_PATH = "visualizaciones_naivebayes.ipynb"

with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

changes = 0
for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    new_src = []
    for line in cell["source"]:
        new_line = re.sub(
            r"(plt\.savefig\(['\"])(?!assets/imagenes/)(grafica_\w+\.png)",
            r"\1assets/imagenes/\2",
            line
        )
        if new_line != line:
            changes += 1
        new_src.append(new_line)
    cell["source"] = new_src

with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"[OK] {changes} rutas de savefig actualizadas en el notebook.")
