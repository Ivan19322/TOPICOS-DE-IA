import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
import os
import json

# Ruta del dataset
DATASET_PATH = "plant_species"

# Aumento de datos (Data Augmentation)
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # 80% entrenamiento, 20% validación
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
    subset="validation"
)

# Guardar las clases (etiquetas) para uso futuro
with open("clases.json", "w") as f:
    json.dump(train_generator.class_indices, f)

# Modelo base MobileNetV2
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# Capas finales personalizadas
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation="relu")(x)
x = Dense(len(train_generator.class_indices), activation="softmax")(x)

# 🔧 Modelo completo
model = Model(inputs=base_model.input, outputs=x)

# Congelar capas base (no entrenables)
for layer in base_model.layers:
    layer.trainable = False

# Compilar el modelo
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Guardar solo el mejor modelo
checkpoint = ModelCheckpoint("mejor_modelo.h5", monitor="val_accuracy", save_best_only=True, mode="max", verbose=1)

# Entrenar (1ra etapa)
EPOCHS = 10
model.fit(train_generator, validation_data=val_generator, epochs=EPOCHS, callbacks=[checkpoint])

# Descongelar las últimas 20 capas del modelo base
for layer in base_model.layers[-20:]:
    layer.trainable = True

# Recompilar con tasa de aprendizaje más baja
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss="categorical_crossentropy", metrics=["accuracy"])

# Entrenar (2da etapa - fine-tuning)
model.fit(train_generator, validation_data=val_generator, epochs=5, callbacks=[checkpoint])

# Guardar el modelo final completo y pesos
model.save("modelo_plantas.h5")
model.save_weights("modelo_plantas.weights.h5")


print("Entrenamiento completo. Modelo y pesos guardados correctamente.")


