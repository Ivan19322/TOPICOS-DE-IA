# Importar las librerías necesarias
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Definir las variables de entrada difusas
# Temperatura puede variar de 0 a 40 grados Celsius
temperatura = ctrl.Antecedent(np.arange(0, 41, 1), 'temperatura')

# Humedad puede variar de 0% a 100%
humedad = ctrl.Antecedent(np.arange(0, 101, 1), 'humedad')

# Definir la variable de salida difusa
# Velocidad del ventilador puede variar de 0% a 100%
velocidad_ventilador = ctrl.Consequent(np.arange(0, 101, 1), 'velocidad_ventilador')

# Definir las funciones de membresía para Temperatura
temperatura['baja'] = fuzz.trimf(temperatura.universe, [0, 0, 20])
temperatura['media'] = fuzz.trimf(temperatura.universe, [10, 20, 30])
temperatura['alta'] = fuzz.trimf(temperatura.universe, [20, 40, 40])

# Definir las funciones de membresía para Humedad
humedad['baja'] = fuzz.trimf(humedad.universe, [0, 0, 50])
humedad['media'] = fuzz.trimf(humedad.universe, [25, 50, 75])
humedad['alta'] = fuzz.trimf(humedad.universe, [50, 100, 100])

# Definir las funciones de membresía para Velocidad del Ventilador
velocidad_ventilador['baja'] = fuzz.trimf(velocidad_ventilador.universe, [0, 0, 50])
velocidad_ventilador['media'] = fuzz.trimf(velocidad_ventilador.universe, [25, 50, 75])
velocidad_ventilador['alta'] = fuzz.trimf(velocidad_ventilador.universe, [50, 100, 100])

# DEFINIR LAS REGLAS DIFUSAS
# Si la temperatura es alta y la humedad es alta, la velocidad del ventilador es alta
regla1 = ctrl.Rule(temperatura['alta'] & humedad['alta'], velocidad_ventilador['alta'])

# Si la temperatura es media y la humedad es media, la velocidad del ventilador es media
regla2 = ctrl.Rule(temperatura['media'] & humedad['media'], velocidad_ventilador['media'])

# Si la temperatura es baja y la humedad es baja, la velocidad del ventilador es baja
regla3 = ctrl.Rule(temperatura['baja'] & humedad['baja'], velocidad_ventilador['baja'])

# Crear el sistema de control difuso
control_ventilador = ctrl.ControlSystem([regla1, regla2, regla3])
simulacion = ctrl.ControlSystemSimulation(control_ventilador)

# Asignar valores de entrada para la simulación
simulacion.input['temperatura'] = 30  # Cambia este valor para probar diferentes escenarios
simulacion.input['humedad'] = 60  # Cambia este valor para probar diferentes escenarios

# Calcular el resultado
simulacion.compute()

# Imprimir el resultado de la velocidad del ventilador
print(f"Velocidad del Ventilador: {simulacion.output['velocidad_ventilador']:.2f}%")
