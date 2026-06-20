"""
Agrega comentarios explicativos a la celda de Visualizacion 2 (heatmap)
sobre que son idx_muestras y labels_muestras.
"""
import json

NOTEBOOK_PATH = "visualizaciones_naivebayes.ipynb"

with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

OLD_BLOCK = [
    "# Seleccionamos 10 comentarios representativos (3 negativos, 3 neutrales, 4 positivos)\n",
    "idx_muestras = [\n",
    "    42, 46, 56,       # Negativos\n",
    "    73, 78, 81,       # Neutrales\n",
    "    0,  7, 13, 36,    # Positivos\n",
    "]\n",
]

NEW_BLOCK = [
    "# idx_muestras: lista de INDICES DE FILA del dataset original (0 a 99).\n",
    "# El CSV tiene 100 comentarios numerados desde 0; cada numero aqui\n",
    "# indica que fila queremos mostrar en el heatmap.\n",
    "# Se eligen 10 filas manualmente, asegurandose de incluir ejemplos\n",
    "# de las tres clases para que la comparacion visual sea representativa.\n",
    "# (No se muestran las 100 filas porque el heatmap seria ilegible.)\n",
    "idx_muestras = [\n",
    "    42, 46, 56,    # filas del CSV que contienen comentarios NEGATIVOS\n",
    "    73, 78, 81,    # filas del CSV que contienen comentarios NEUTRALES\n",
    "    0,  7, 13, 36, # filas del CSV que contienen comentarios POSITIVOS\n",
    "]\n",
]

changes = 0
for cell in nb["cells"]:
    if cell["cell_type"] != "code" or cell.get("id") != "12c6a7f7":
        continue

    src = cell["source"]

    # Buscar el bloque a reemplazar
    # Unimos para comparar de forma flexible
    joined = "".join(src)
    old_joined = "".join(OLD_BLOCK)

    if old_joined in joined:
        new_joined = joined.replace(old_joined, "".join(NEW_BLOCK), 1)
        cell["source"] = list(line + "" for line in new_joined.splitlines(keepends=True))
        changes += 1
        print("[OK] Comentarios agregados en celda 12c6a7f7 (Viz 2 - heatmap)")
    else:
        # Intentar reemplazo linea por linea
        new_src = []
        i = 0
        while i < len(src):
            if src[i] == OLD_BLOCK[0]:
                # Verificar que las siguientes lineas coincidan
                match = all(
                    i + j < len(src) and src[i + j] == OLD_BLOCK[j]
                    for j in range(len(OLD_BLOCK))
                )
                if match:
                    new_src.extend(NEW_BLOCK)
                    i += len(OLD_BLOCK)
                    changes += 1
                    print("[OK] Comentarios agregados (metodo linea por linea)")
                    continue
            new_src.append(src[i])
            i += 1
        cell["source"] = new_src

if changes == 0:
    print("[!] No se encontro el bloque exacto. Mostrando primeras lineas de la celda para debug:")
    for cell in nb["cells"]:
        if cell.get("id") == "12c6a7f7":
            for line in cell["source"][:8]:
                print(repr(line))
else:
    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print("[OK] Notebook guardado.")
    print("     Haz Kernel -> Restart & Run All para aplicar los cambios.")
