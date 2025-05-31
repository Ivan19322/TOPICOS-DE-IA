import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import json

# Cargar modelo
modelo = load_model("modelo_plantas.h5")
print("✅ Modelo cargado.")

# Cargar clases desde archivo JSON
with open("clases.json", "r") as f:
    clases_dict = json.load(f)

# Crear lista ordenada de clases
clases_lista = [None] * len(clases_dict)
for nombre, idx in clases_dict.items():
    clases_lista[idx] = nombre

# Iniciar cámara
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ No se pudo abrir la cámara.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ No se pudo leer el frame.")
        break

    # Preprocesar imagen
    img = cv2.resize(frame, (224, 224))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    # Predecir
    pred = modelo.predict(img, verbose=0)
    indice = int(np.argmax(pred))
    etiqueta = clases_lista[indice] if indice < len(clases_lista) else "Desconocida"

    # Mostrar etiqueta en pantalla
    cv2.putText(frame, f"Planta: {etiqueta}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Clasificador de Plantas", frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
print("✅ Cámara cerrada.")
