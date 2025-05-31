import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import os
import json

# Ruta del dataset
DATASET_PATH = "plant_species"

# Aumento de datos
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

# Cargar imágenes
train_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    subset="training"
)
val_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    subset="validation",
    shuffle=False  # Importante para métricas
)

# Guardar las clases
with open("clases.json", "w") as f:
    json.dump(train_generator.class_indices, f)

# Modelo base
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation="relu")(x)
x = Dense(len(train_generator.class_indices), activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=x)

# Congelar capas base
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Checkpoint
checkpoint = ModelCheckpoint("mejor_modelo.h5", monitor="val_accuracy", save_best_only=True, mode="max", verbose=1)

# Entrenamiento inicial
model.fit(train_generator, validation_data=val_generator, epochs=10, callbacks=[checkpoint])

# Descongelar últimas 20 capas
for layer in base_model.layers[-20:]:
    layer.trainable = True

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss="categorical_crossentropy", metrics=["accuracy"])

# Fine-tuning
model.fit(train_generator, validation_data=val_generator, epochs=5, callbacks=[checkpoint])

# Guardar modelo
model.save("modelo_plantas.h5")
model.save_weights("modelo_plantas.weights.h5")

# === Evaluación Final ===

# Obtener predicciones
val_generator.reset()
Y_pred = model.predict(val_generator)
y_pred = np.argmax(Y_pred, axis=1)
y_true = val_generator.classes

# 1. Exactitud total
acc = accuracy_score(y_true, y_pred)
print(f"\nExactitud en validación: {acc:.4f}")

# 2. Matriz de confusión
print("\nMatriz de confusión:")
print(confusion_matrix(y_true, y_pred))

# 3. Reporte de clasificación
print("\nReporte de clasificación:")
print(classification_report(y_true, y_pred, target_names=list(val_generator.class_indices.keys())))

print("\n✅ Entrenamiento y evaluación completados correctamente.")
