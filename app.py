"""
app.py — Backend Flask para el clasificador de sentimientos Mundial 2026
Expone el modelo Naive Bayes entrenado y una API REST para predicciones.
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

app = Flask(__name__, template_folder='docs')

# ── Constantes ────────────────────────────────────────────────────────────────
STOP_WORDS_ES = [
    "el","la","los","las","un","una","unos","unas","de","del","al",
    "en","y","o","que","con","por","para","es","fue","fueron","ser",
    "se","su","sus","lo","le","les","a","ante","entre","como","muy",
    "más","mas","pero","sin","sobre","tras","esta","este","esa","ese",
    "esto","eso","también","ya","no","sí","si","tiene","tuvo","han",
    "ha","será","están","está","todo","toda","todos","todas",
]

COLORES = {
    "Negativo": "#E74C3C",
    "Neutral":  "#95A5A6",
    "Positivo": "#27AE60",
}

# ── Cargar y entrenar el modelo al iniciar la app ─────────────────────────────
CSV_PATH = os.path.join(os.path.dirname(__file__), "dataset_mundial2026.csv")
df = pd.read_csv(CSV_PATH)

vectorizador = CountVectorizer(stop_words=STOP_WORDS_ES, lowercase=True, max_features=500)
X = vectorizador.fit_transform(df["comentario"])
y = df["sentimiento"]

# Modelo 80/20 para accuracy
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
modelo_eval = MultinomialNB()
modelo_eval.fit(X_train, y_train)
accuracy = accuracy_score(y_test, modelo_eval.predict(X_test))

# Modelo final (100% datos) para predicciones en producción
modelo = MultinomialNB()
modelo.fit(X, y)

CLASES = list(modelo.classes_)  # ['Negativo', 'Neutral', 'Positivo']
vocab = vectorizador.vocabulary_
nombres_features = vectorizador.get_feature_names_out()
priors = np.exp(modelo.class_log_prior_)

print(f"[OK] Modelo entrenado — Accuracy: {accuracy:.2%} | Vocabulario: {len(vocab)} palabras")


# ── Rutas ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    stats = {
        "total": len(df),
        "accuracy": f"{accuracy:.1%}",
        "vocabulario": len(vocab),
        "distribucion": df["sentimiento"].value_counts().to_dict(),
    }
    return render_template("index.html", stats=stats)


@app.route("/predecir", methods=["POST"])
def predecir():
    data = request.get_json()
    comentario = data.get("comentario", "").strip()

    if not comentario:
        return jsonify({"error": "Comentario vacío"}), 400

    # ── 1. Vectorizar ──────────────────────────────────────────────────────────
    vec = vectorizador.transform([comentario])

    # ── 2. Predecir ───────────────────────────────────────────────────────────
    prediccion = modelo.predict(vec)[0]
    probabilidades = modelo.predict_proba(vec)[0]
    proba_dict = {cls: float(p) for cls, p in zip(CLASES, probabilidades)}

    # ── 3. Calcular votos por palabra ──────────────────────────────────────────
    palabras_raw = comentario.lower().split()
    palabras_en_vocab = [p for p in palabras_raw if p in vocab and p not in STOP_WORDS_ES]

    votos = {}
    for palabra in palabras_en_vocab:
        col_idx = vocab[palabra]
        votos[palabra] = {}
        for i, clase in enumerate(CLASES):
            log_prob = modelo.feature_log_prob_[i][col_idx]
            votos[palabra][clase] = float(np.exp(log_prob))

    palabras_ignoradas = [p for p in palabras_raw if p not in vocab or p in STOP_WORDS_ES]

    # ── 4. Calcular score bayesiano paso a paso ───────────────────────────────
    pasos = []
    scores_raw = {}
    for i, clase in enumerate(CLASES):
        prior = float(priors[i])
        producto = prior
        detalle_palabras = []
        for palabra in palabras_en_vocab:
            w = votos[palabra][clase]
            detalle_palabras.append({
                "palabra": palabra,
                "peso": w,
                "peso_fmt": f"{w:.5f}",
            })
            producto *= w
        scores_raw[clase] = producto
        pasos.append({
            "clase": clase,
            "color": COLORES[clase],
            "prior": prior,
            "prior_fmt": f"{prior:.3f}",
            "score_bruto": producto,
            "score_fmt": f"{producto:.2e}",
            "palabras": detalle_palabras,
        })

    total_scores = sum(scores_raw.values())
    for p in pasos:
        p["probabilidad"] = float(scores_raw[p["clase"]] / total_scores) if total_scores > 0 else 0
        p["porcentaje"] = f"{p['probabilidad']*100:.1f}%"

    return jsonify({
        "prediccion": prediccion,
        "color_prediccion": COLORES[prediccion],
        "probabilidades": proba_dict,
        "palabras_reconocidas": palabras_en_vocab,
        "palabras_ignoradas": palabras_ignoradas,
        "votos": votos,
        "pasos": pasos,
        "clases": CLASES,
        "colores": COLORES,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
