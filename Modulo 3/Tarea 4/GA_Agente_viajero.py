import random
import numpy as np

# Paso 1: Representar el grafo de conexiones y parámetros del algoritmo genético
grafo_conexiones = {
    "Bilbao": ["Zaragoza", "Celta"],
    "Zaragoza": ["Bilbao", "Gerona", "Barcelona", "Madrid"],
    "Celta": ["Bilbao", "Vigo"],
    "Vigo": ["Celta", "Valladolid"],
    "Valladolid": ["Vigo", "Madrid"],
    "Madrid": ["Zaragoza", "Valladolid", "Jaen", "Albacete"],
    "Jaen": ["Madrid", "Sevilla", "Granada"],
    "Sevilla": ["Jaen"],
    "Granada": ["Jaen", "Albacete", "Murcia"],
    "Albacete": ["Madrid", "Granada", "Murcia", "Valencia"],
    "Murcia": ["Albacete", "Granada", "Valencia"],
    "Valencia": ["Albacete", "Murcia", "Barcelona"],
    "Barcelona": ["Zaragoza", "Gerona", "Valencia"],
    "Gerona": ["Zaragoza", "Barcelona"],
}

distancias = {
    ("Bilbao", "Zaragoza"): 324,
    ("Bilbao", "Celta"): 378,
    ("Zaragoza", "Gerona"): 289,
    ("Zaragoza", "Barcelona"): 296,
    ("Zaragoza", "Madrid"): 390,
    ("Celta", "Vigo"): 171,
    ("Vigo", "Valladolid"): 235,
    ("Valladolid", "Madrid"): 193,
    ("Madrid", "Jaen"): 411,
    ("Madrid", "Albacete"): 251,
    ("Jaen", "Sevilla"): 125,
    ("Jaen", "Granada"): 207,
    ("Granada", "Albacete"): 244,
    ("Granada", "Murcia"): 257,
    ("Albacete", "Murcia"): 150,
    ("Albacete", "Valencia"): 191,
    ("Murcia", "Valencia"): 241,
    ("Valencia", "Barcelona"): 349,
    ("Barcelona", "Gerona"): 100,
}

# Asegurar que las distancias sean simétricas
for (ciudad1, ciudad2), distancia in list(distancias.items()):
    distancias[(ciudad2, ciudad1)] = distancia

tamaño_poblacion = 50
probabilidad_cruce = 0.8
probabilidad_mutacion = 0.2
generaciones = 100

# Paso 2: Definir función de aptitud
def calcular_distancia(ruta):
    distancia_total = 0
    for i in range(len(ruta) - 1):
        if (ruta[i], ruta[i + 1]) not in distancias:
            raise KeyError(f"No hay conexión entre {ruta[i]} y {ruta[i + 1]}")
        distancia_total += distancias[(ruta[i], ruta[i + 1])]
    if (ruta[-1], ruta[0]) not in distancias:
        raise KeyError(f"No hay conexión entre {ruta[-1]} y {ruta[0]}")
    distancia_total += distancias[(ruta[-1], ruta[0])]  # Regresar al inicio
    return distancia_total

# Paso 3: Generar rutas válidas basadas en el grafo de conexiones
def generar_ruta_valida(ciudades, grafo):
    ruta = [random.choice(ciudades)]
    while len(ruta) < len(ciudades):
        ultima_ciudad = ruta[-1]
        opciones = [ciudad for ciudad in grafo[ultima_ciudad] if ciudad not in ruta]
        if not opciones:
            break  # No hay más opciones válidas
        ruta.append(random.choice(opciones))
    return ruta if len(ruta) == len(ciudades) else None

def crear_poblacion(ciudades, tamaño_poblacion, grafo):
    poblacion = []
    while len(poblacion) < tamaño_poblacion:
        ruta = generar_ruta_valida(ciudades, grafo)
        if ruta:
            poblacion.append(ruta)
    return poblacion

# Paso 4: Calcular aptitudes
def calcular_aptitudes(poblacion):
    aptitudes = []
    for ruta in poblacion:
        try:
            aptitudes.append(1 / calcular_distancia(ruta))
        except KeyError as e:
            print(f"Ruta inválida {ruta}: {e}")
            aptitudes.append(0)  # Penalizar rutas inválidas
    return aptitudes

# Paso 5: Selección (Ruleta)
def seleccion_ruleta(poblacion, aptitudes):
    total_aptitud = sum(aptitudes)
    if total_aptitud == 0:
        raise ValueError("Todas las rutas tienen aptitud cero. Verifica las conexiones.")
    probabilidades = [aptitud / total_aptitud for aptitud in aptitudes]
    indices_seleccionados = np.random.choice(len(poblacion), size=2, p=probabilidades)
    return [poblacion[indices_seleccionados[0]], poblacion[indices_seleccionados[1]]]

# Paso 6: Cruce
def cruce(padre1, padre2, grafo):
    punto_corte = random.randint(0, len(padre1) - 1)
    hijo = padre1[:punto_corte]
    for ciudad in padre2:
        if ciudad not in hijo and ciudad in grafo[hijo[-1]]:
            hijo.append(ciudad)
    return hijo

# Paso 6: Mutación
def mutacion(ruta, probabilidad_mutacion, grafo):
    if random.random() < probabilidad_mutacion:
        i, j = random.sample(range(len(ruta)), 2)
        if ruta[j] in grafo[ruta[i - 1]] and ruta[(i + 1) % len(ruta)] in grafo[ruta[j]]:
            ruta[i], ruta[j] = ruta[j], ruta[i]

# Paso 7: Reemplazo y iteración
def algoritmo_genetico(ciudades, grafo, generaciones, tamaño_poblacion, probabilidad_cruce, probabilidad_mutacion):
    poblacion = crear_poblacion(ciudades, tamaño_poblacion, grafo)
    mejor_ruta = None
    mejor_distancia = float('inf')
    
    for _ in range(generaciones):
        aptitudes = calcular_aptitudes(poblacion)
        nueva_poblacion = []
        
        for _ in range(tamaño_poblacion // 2):
            padre1, padre2 = seleccion_ruleta(poblacion, aptitudes)
            if random.random() < probabilidad_cruce:
                hijo1 = cruce(padre1, padre2, grafo)
                hijo2 = cruce(padre2, padre1, grafo)
            else:
                hijo1, hijo2 = padre1[:], padre2[:]
            
            mutacion(hijo1, probabilidad_mutacion, grafo)
            mutacion(hijo2, probabilidad_mutacion, grafo)
            
            nueva_poblacion.extend([hijo1, hijo2])
        
        poblacion = nueva_poblacion
        
        # Evaluar el mejor cromosoma
        for ruta in poblacion:
            try:
                distancia = calcular_distancia(ruta)
                if distancia < mejor_distancia:
                    mejor_ruta, mejor_distancia = ruta, distancia
            except KeyError:
                continue
    
    return mejor_ruta, mejor_distancia

# Paso 8: Ejecutar el algoritmo
try:
    mejor_ruta, mejor_distancia = algoritmo_genetico(list(grafo_conexiones.keys()), grafo_conexiones, generaciones, tamaño_poblacion, probabilidad_cruce, probabilidad_mutacion)
    print("Mejor ruta:", mejor_ruta)
    print("Distancia total:", mejor_distancia)
except ValueError as e:
    print("Error en el algoritmo:", e)