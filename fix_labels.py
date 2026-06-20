"""
Corrige el empalme de etiquetas numéricas en las celdas 3 y 5a del notebook.
- Viz 3 (barras horizontales): agrega xlim con espacio extra para el texto
- Viz 5a (barras verticales con rotacion): aumenta ylim para evitar el empalme
"""
import json

NOTEBOOK_PATH = "visualizaciones_naivebayes.ipynb"

with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]
changes = 0

for cell in cells:
    if cell["cell_type"] != "code":
        continue

    src = cell["source"]
    joined = "".join(src)

    # ── Visualización 3: barras horizontales — agregar xlim ──────────────
    if cell.get("id") == "13083ce6":
        new_src = []
        for line in src:
            new_src.append(line)
            # Justo después de set_xlabel de cada subplot, inyectar xlim
            if "ax.set_xlabel('Probabilidad P(palabra | clase)'" in line:
                indent = "    "
                new_src.append(
                    indent + "ax.set_xlim(0, max(top_probs) * 1.45)\n"
                )
        if new_src != src:
            cell["source"] = new_src
            changes += 1
            print("[OK] Viz 3 (barras horizontales): xlim ajustado")

    # ── Visualización 5a: barras verticales — aumentar ylim ──────────────
    if cell.get("id") == "104bf52f":
        new_src = []
        for line in src:
            # Aumentar el offset vertical del texto para separarlo de la barra
            if "barra.get_height() + 0.00005," in line:
                line = line.replace(
                    "barra.get_height() + 0.00005,",
                    "barra.get_height() + max(valores) * 0.08,"
                )
                changes += 1
            # Cambiar rotacion de 45 a 90 para que no se empalmen
            if "rotation=45" in line and "f'{val:.4f}'" in "".join(new_src[-4:]):
                line = line.replace("rotation=45", "rotation=90")
            new_src.append(line)

        # Agregar ylim generoso justo antes de plt.tight_layout() de la primera figura
        final_src = []
        inserted_ylim = False
        for line in new_src:
            # Insertar ylim antes del primer tight_layout de la seccion de barras agrupadas
            if (not inserted_ylim
                    and "plt.tight_layout()" in line
                    and "grafica_5a" in "".join(new_src)):
                final_src.append(
                    "ax.set_ylim(0, max(max(datos_votos[p][c] for p in palabras_en_vocab) "
                    "for c in CLASES_ORDEN) * 1.55)\n"
                )
                inserted_ylim = True
            final_src.append(line)

        cell["source"] = final_src
        print("[OK] Viz 5a (barras verticales): ylim y rotacion ajustados")

if changes == 0:
    print("[!] No se encontraron los patrones esperados. Verifica los IDs de celda.")
else:
    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"\n[OK] Notebook guardado con {changes} cambio(s).")
    print("     Haz Kernel -> Restart & Run All para ver los resultados.")
