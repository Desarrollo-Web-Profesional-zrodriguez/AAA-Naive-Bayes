# 🌍 Clasificador de Sentimientos — Comentarios Mundial 2026

**Algoritmo:** Naive Bayes Multinomial + Bag of Words  
**Actividad 2 — Análisis Supervisado**

---

## 📁 Estructura del proyecto

```
Naive-Bayes/
│
├── 📓 visualizaciones_naivebayes.ipynb   # Notebook con todas las visualizaciones
├── 🐍 naive_bayes_mundial2026.py         # Script principal del modelo
├── 🌐 app.py                             # Aplicativo web (Flask) para la exposición
├── 📄 dataset_mundial2026.csv            # Dataset: 100 comentarios etiquetados
│
├── 📂 assets/
│   └── imagenes/                        # Gráficas generadas por el notebook
│       ├── grafica_1_distribucion.png
│       ├── grafica_2_vectorizacion.png
│       ├── grafica_3_palabras_por_clase.png
│       ├── grafica_4_matriz_confusion.png
│       ├── grafica_5a_votos.png
│       ├── grafica_5b_resultado_final.png
│       └── grafica_6_demo_interactiva.png
│
├── 📂 docs/                             # GitHub Pages (demo web estática)
│   └── index.html
│
└── 📂 scripts/                          # Utilidades de desarrollo (no son parte del modelo)
    ├── add_setup_cell.py               # Inyecta celda de setup en el notebook
    ├── comment_idx.py                  # Agrega comentarios a idx_muestras
    ├── fix_labels.py                   # Corrige empalme de etiquetas en gráficas
    ├── inspect_nb.py                   # Inspecciona celdas del notebook
    └── update_paths.py                 # Actualiza rutas de savefig en el notebook
```

---

## 🚀 Cómo ejecutar

### Notebook (visualizaciones)
```bash
jupyter notebook visualizaciones_naivebayes.ipynb
# Ejecutar: Kernel → Restart & Run All
```

### Script principal
```bash
python naive_bayes_mundial2026.py
```

### Aplicativo web (exposición)
```bash
python app.py
# Abrir: http://127.0.0.1:5000
```

---

## 📊 Resultados del modelo

| Métrica | Valor |
|---|---|
| Exactitud (Accuracy) | 65% |
| Algoritmo | Multinomial Naive Bayes |
| Vocabulario | 457 palabras |
| Dataset | 100 comentarios (40 Pos · 32 Neg · 28 Neu) |

---

## 🌐 Demo en línea

[Ver demo en GitHub Pages](https://desarrollo-web-profesional-zrodriguez.github.io/AAA-Naive-Bayes/)
