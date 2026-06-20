"""
Inserta una celda de setup al inicio del notebook visualizaciones_naivebayes.ipynb
(justo después del primer markdown de título).
"""
import json

NOTEBOOK_PATH = "visualizaciones_naivebayes.ipynb"

SETUP_CELL = {
    "cell_type": "code",
    "execution_count": None,
    "id": "setup-cell-00",
    "metadata": {},
    "outputs": [],
    "source": [
        "# ══════════════════════════════════════════════════════════════════════\n",
        "# CELDA DE CONFIGURACIÓN — ejecutar PRIMERO antes de cualquier otra\n",
        "# Carga librerías, dataset, entrena los modelos y define colores\n",
        "# ══════════════════════════════════════════════════════════════════════\n",
        "\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "import matplotlib.patches as mpatches\n",
        "import seaborn as sns\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.feature_extraction.text import CountVectorizer\n",
        "from sklearn.naive_bayes import MultinomialNB\n",
        "from sklearn.metrics import accuracy_score, confusion_matrix, classification_report\n",
        "\n",
        "# ── Paleta de colores ─────────────────────────────────────────────────\n",
        "COLORES = {\n",
        "    'Negativo': '#E74C3C',\n",
        "    'Neutral':  '#95A5A6',\n",
        "    'Positivo': '#27AE60',\n",
        "}\n",
        "CLASES_ORDEN  = ['Negativo', 'Neutral', 'Positivo']\n",
        "LISTA_COLORES = [COLORES[c] for c in CLASES_ORDEN]\n",
        "\n",
        "# ── Cargar dataset ────────────────────────────────────────────────────\n",
        "df = pd.read_csv('dataset_mundial2026.csv')\n",
        "X_texto = df['comentario']\n",
        "y       = df['sentimiento']\n",
        "print(f'Dataset cargado: {len(df)} comentarios')\n",
        "print(df['sentimiento'].value_counts())\n",
        "\n",
        "# ── Vectorización ─────────────────────────────────────────────────────\n",
        "STOP_WORDS_ES = [\n",
        "    'el','la','los','las','un','una','unos','unas','de','del','al',\n",
        "    'en','y','o','que','con','por','para','es','fue','fueron','ser',\n",
        "    'se','su','sus','lo','le','les','a','ante','entre','como','muy',\n",
        "    'más','mas','pero','sin','sobre','tras','esta','este','esa','ese',\n",
        "    'esto','eso','también','ya','no','sí','si','tiene','tuvo','han',\n",
        "    'ha','será','están','está','todo','toda','todos','todas',\n",
        "]\n",
        "vectorizador = CountVectorizer(stop_words=STOP_WORDS_ES, lowercase=True, max_features=500)\n",
        "X = vectorizador.fit_transform(X_texto)\n",
        "print(f'Vocabulario: {len(vectorizador.vocabulary_)} palabras | Matriz: {X.shape}')\n",
        "\n",
        "# ── División 80/20 ────────────────────────────────────────────────────\n",
        "X_train, X_test, y_train, y_test = train_test_split(\n",
        "    X, y, test_size=0.2, random_state=42, stratify=y\n",
        ")\n",
        "\n",
        "# ── Modelo 80% (evaluación honesta) ───────────────────────────────────\n",
        "modelo = MultinomialNB()\n",
        "modelo.fit(X_train, y_train)\n",
        "y_pred = modelo.predict(X_test)\n",
        "print(f'Accuracy (80/20): {accuracy_score(y_test, y_pred):.2%}')\n",
        "\n",
        "# ── Modelo final 100% (predicciones y visualizaciones de palabras) ────\n",
        "modelo_final = MultinomialNB()\n",
        "modelo_final.fit(X, y)\n",
        "\n",
        "print('\\n✅ Setup completo — ya puedes ejecutar las celdas de visualización')\n",
    ]
}

with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]

# Verificar si ya existe la celda de setup para no duplicarla
already_exists = any(
    c.get("id") == "setup-cell-00" for c in cells if c["cell_type"] == "code"
)

if already_exists:
    print("[!] La celda de setup ya existe en el notebook. No se realizaron cambios.")
else:
    # Insertar después del primer markdown (celda de título, índice 0)
    cells.insert(1, SETUP_CELL)
    nb["cells"] = cells

    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

    print("[OK] Celda de setup insertada correctamente en visualizaciones_naivebayes.ipynb")
    print("     Ahora ejecuta esa celda PRIMERO en Jupyter y el resto funcionara.")
