import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# 1. Generamos los datos de entrenamiento con normalización
def generar_datos(cantidad):
    x = np.random.randint(1, 100, size=(cantidad, 2))
    y = np.zeros((cantidad, 4))
    y[:, 0] = x[:, 0] + x[:, 1]  # Suma
    y[:, 1] = x[:, 0] - x[:, 1]  # Resta
    y[:, 2] = x[:, 0] * x[:, 1]  # Multiplicación
    y[:, 3] = np.divide(x[:, 0], x[:, 1] + 1e-7)  # División evitando división por 0
    return x, y

X, y = generar_datos(10000)

# 2. Normalización de los datos
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
X = scaler_x.fit_transform(X)
y = scaler_y.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 3. Definir el modelo
modelo = Sequential([
    Dense(64, input_dim=2, activation='relu'),
    Dense(64, activation='relu'),
    Dense(4, activation='linear')  # Activación lineal en la salida
])

# 4. Compilar el modelo
modelo.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# 5. Entrenar el modelo
modelo.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)

# 6. Evaluar el modelo
loss, mae = modelo.evaluate(X_test, y_test)
print(f"Error promedio absoluto: {mae:.4f}")

# 7. Probar con nuevos datos
def predecir(x1, x2):
    entrada = scaler_x.transform(np.array([[x1, x2]]))
    prediccion = scaler_y.inverse_transform(modelo.predict(entrada))[0]
    print(f"{x1} + {x2} ≈ {prediccion[0]:.2f}")
    print(f"{x1} - {x2} ≈ {prediccion[1]:.2f}")
    print(f"{x1} * {x2} ≈ {prediccion[2]:.2f}")
    print(f"{x1} / {x2} ≈ {prediccion[3]:.2f}")

# Ejemplo de uso
predecir(12, 4)
