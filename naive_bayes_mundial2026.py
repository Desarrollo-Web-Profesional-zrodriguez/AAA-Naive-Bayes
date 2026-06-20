"""
======================================================================
 CLASIFICADOR DE SENTIMIENTOS - COMENTARIOS MUNDIAL 2026
 Algoritmo: Naive Bayes (Multinomial)
 Actividad 2 - Análisis Supervisado
======================================================================
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------------------------------------
# PASO 1: CARGAR EL DATASET
# ----------------------------------------------------------------------
print("=" * 60)
print("PASO 1: Cargando el dataset")
print("=" * 60)

df = pd.read_csv("dataset_mundial2026.csv")
print(f"Total de comentarios: {len(df)}")
print(f"\nDistribución de clases:")
print(df["sentimiento"].value_counts())
print()

X_texto = df["comentario"]
y = df["sentimiento"]

# ----------------------------------------------------------------------
# PASO 2: VECTORIZACIÓN (convertir texto en números)
# ----------------------------------------------------------------------
print("=" * 60)
print("PASO 2: Vectorizando el texto (Bag of Words)")
print("=" * 60)

# Stop words en español: palabras muy frecuentes que no aportan
# significado de sentimiento (el, la, de, que, en...). Quitarlas deja
# que las palabras importantes (goleó, pésimo, decepcionante...) pesen
# más en el cálculo de probabilidades de Naive Bayes.
STOP_WORDS_ES = [
    "el","la","los","las","un","una","unos","unas","de","del","al",
    "en","y","o","que","con","por","para","es","fue","fueron","ser",
    "se","su","sus","lo","le","les","a","ante","entre","como","muy",
    "más","mas","pero","sin","sobre","tras","esta","este","esa","ese",
    "esto","eso","también","ya","no","sí","si","tiene","tuvo","han",
    "ha","será","están","está","todo","toda","todos","todas",
]

# CountVectorizer convierte cada comentario en un vector de frecuencias
# de palabras, ignorando las stop words definidas arriba.
vectorizador = CountVectorizer(stop_words=STOP_WORDS_ES, lowercase=True, max_features=500)
X = vectorizador.fit_transform(X_texto)

print(f"Vocabulario total: {len(vectorizador.vocabulary_)} palabras únicas")
print(f"Dimensiones de la matriz: {X.shape}")
print()

# ----------------------------------------------------------------------
# PASO 3: DIVIDIR EN ENTRENAMIENTO Y PRUEBA
# ----------------------------------------------------------------------
print("=" * 60)
print("PASO 3: Dividiendo datos (80% entrenamiento / 20% prueba)")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Comentarios de entrenamiento: {X_train.shape[0]}")
print(f"Comentarios de prueba: {X_test.shape[0]}")
print()

# ----------------------------------------------------------------------
# PASO 4: ENTRENAR EL MODELO NAIVE BAYES
# ----------------------------------------------------------------------
print("=" * 60)
print("PASO 4: Entrenando el modelo Naive Bayes")
print("=" * 60)

modelo = MultinomialNB()
modelo.fit(X_train, y_train)

print("Modelo entrenado exitosamente")
print(f"Clases aprendidas: {modelo.classes_}")
print()

# ----------------------------------------------------------------------
# PASO 5: EVALUAR EL MODELO
# ----------------------------------------------------------------------
print("=" * 60)
print("PASO 5: Evaluando el modelo con datos de prueba")
print("=" * 60)

y_pred = modelo.predict(X_test)
exactitud = accuracy_score(y_test, y_pred)

print(f"Exactitud (Accuracy): {exactitud:.2%}")
print()
print("Reporte de clasificación:")
print(classification_report(y_test, y_pred))

# Matriz de confusión
matriz = confusion_matrix(y_test, y_pred, labels=modelo.classes_)
print("Matriz de Confusión:")
print(matriz)
print()

# Visualización de la matriz de confusión
plt.figure(figsize=(6, 5))
sns.heatmap(
    matriz,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=modelo.classes_,
    yticklabels=modelo.classes_,
)
plt.title("Matriz de Confusión - Naive Bayes\nComentarios Mundial 2026")
plt.xlabel("Predicción del Modelo")
plt.ylabel("Valor Real")
plt.tight_layout()
plt.savefig("matriz_confusion.png", dpi=150)
print("Gráfica guardada como 'matriz_confusion.png'")
print()

# ----------------------------------------------------------------------
# PASO 6: PROBAR CON COMENTARIOS NUEVOS (esto es lo que se muestra en vivo)
# ----------------------------------------------------------------------
print("=" * 60)
print("PASO 6: Clasificando comentarios NUEVOS (demo en vivo)")
print("=" * 60)

# Para la demo en vivo reentrenamos con el 100% de los datos disponibles
# (ya validamos el desempeño del modelo en el Paso 5 con datos de prueba
# que el modelo nunca había visto). Usar todo el dataset aquí le da al
# modelo el máximo conocimiento posible para clasificar mejor.
modelo_final = MultinomialNB()
modelo_final.fit(X, y)

comentarios_nuevos = [
    "Messi vuelve a hacer historia con otro golazo espectacular",
    "Terrible y flojo nivel mostrado por el equipo durante todo el partido",
    "México y Corea del Sur se enfrentan esta noche en Guadalajara",
    "Qué decepción la defensa estuvo terrible todo el partido",
    "Excelente actuación del equipo merecieron la victoria por completo",
]

nuevos_vectorizados = vectorizador.transform(comentarios_nuevos)
predicciones = modelo_final.predict(nuevos_vectorizados)
probabilidades = modelo_final.predict_proba(nuevos_vectorizados)

for comentario, pred, prob in zip(comentarios_nuevos, predicciones, probabilidades):
    print(f"\nComentario: \"{comentario}\"")
    print(f"  → Predicción: {pred}")
    for clase, p in zip(modelo_final.classes_, prob):
        print(f"     {clase}: {p:.1%}")

print()
print("=" * 60)
print("FIN DEL PROGRAMA")
print("=" * 60)
