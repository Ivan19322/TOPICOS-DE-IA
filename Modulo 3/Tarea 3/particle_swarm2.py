import random

class Particle:
    def __init__(self, position_limits, velocity_limits, cognitive_weight, social_weight):
        """
        Inicializa una partícula con posición, velocidad, memoria, valor,
        peso cognitivo y peso social.
        """
        # Generar posición inicial aleatoria dentro de los límites
        self.position = [random.uniform(*limits) for limits in position_limits]
        
        # Generar velocidad inicial aleatoria dentro de los límites
        self.velocity = [random.uniform(*limits) for limits in velocity_limits]
        
        # Inicializar memoria (mejor posición encontrada por la partícula)
        self.memory = self.position[:]
        
        # Valor asociado a la partícula (puede ser evaluado posteriormente)
        self.value = None
        
        # Pesos para el movimiento (cognitivo y social)
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight

    def evaluar_particula(self, funcion_objetivo):
        """
        Evalúa la partícula en función de una función objetivo.
        """
        self.value = funcion_objetivo(self.position)
        # Actualiza la mejor posición (memoria) si el valor actual es mejor
        if self.value < funcion_objetivo(self.memory):  # Asume que estamos minimizando
            self.memory = self.position[:]

    def mover_particula(self, global_best, inertia, velocity_limits):
        """
        Actualiza la velocidad y posición de la partícula en base a la inercia,
        su mejor posición (máximo local) y el mejor global.
        """
        for i in range(len(self.position)):
            r1 = random.uniform(0.8, 1.2)  # Factor aleatorio para el componente cognitivo
            r2 = random.uniform(0.8, 1.2)  # Factor aleatorio para el componente social

            # Actualizar velocidad
            cognitive_component = self.cognitive_weight * r1 * (self.memory[i] - self.position[i])
            social_component = self.social_weight * r2 * (global_best[i] - self.position[i])
            self.velocity[i] = inertia * self.velocity[i] + cognitive_component + social_component

            # Limitar la velocidad dentro de los límites
            self.velocity[i] = max(min(self.velocity[i], velocity_limits[i][1]), velocity_limits[i][0])

            # Actualizar posición
            self.position[i] += self.velocity[i]

    def __repr__(self):
        """
        Representación en texto de la partícula.
        """
        return (f"Particle(position={self.position}, velocity={self.velocity}, "
                f"memory={self.memory}, value={self.value}, "
                f"cognitive_weight={self.cognitive_weight}, social_weight={self.social_weight})")


class Swarm:
    def __init__(self, num_particles, position_limits, velocity_limits, cognitive_weight, social_weight):
        """
        Inicializa el enjambre con una cantidad específica de partículas.
        """
        self.particles = [
            Particle(position_limits, velocity_limits, cognitive_weight, social_weight)
            for _ in range(num_particles)
        ]
        self.global_best = None  # Mejor posición global encontrada
        self.global_best_value = float('inf')  # Valor asociado al mejor global

    def evaluar_enjambre(self, funcion_objetivo):
        """
        Evalúa todas las partículas en el enjambre y actualiza el mejor global.
        """
        for particle in self.particles:
            particle.evaluar_particula(funcion_objetivo)
            if particle.value < self.global_best_value:  # Actualizar el mejor global si es necesario
                self.global_best = particle.position[:]
                self.global_best_value = particle.value

    def mover_enjambre(self, inertia, velocity_limits):
        """
        Mueve todas las partículas en el enjambre.
        """
        for particle in self.particles:
            particle.mover_particula(self.global_best, inertia, velocity_limits)

    def __repr__(self):
        """
        Representación en texto del enjambre.
        """
        return "\n".join(str(particle) for particle in self.particles)


# Ejemplo de uso
if __name__ == "__main__":
    # Definir límites para posición y velocidad
    position_limits = [(-10, 10), (-5, 5)]  # Por ejemplo, 2 dimensiones con diferentes límites
    velocity_limits = [(-1, 1), (-0.5, 0.5)]

    # Definir los pesos cognitivo y social
    cognitive_weight = 2.0  # Valor máximo local
    social_weight = 2.0     # Valor máximo global

    # Crear un enjambre con 10 partículas
    swarm = Swarm(
        num_particles=10,
        position_limits=position_limits,
        velocity_limits=velocity_limits,
        cognitive_weight=cognitive_weight,
        social_weight=social_weight
    )

    # Definir una función objetivo (ejemplo: minimizar la suma de cuadrados)
    def funcion_objetivo(pos):
        return sum(x**2 for x in pos)

    # Parámetros para el movimiento
    inertia_max = 0.9  # Inercia inicial
    inertia_min = 0.4  # Inercia final
    total_iteraciones = 20  # Número total de iteraciones

    # Ciclo de optimización
    for iteracion in range(total_iteraciones):
        # Calcular el factor de inercia dinámico
        inertia = inertia_max - ((inertia_max - inertia_min) * iteracion / total_iteraciones)

        print(f"Iteración {iteracion + 1}, Inercia: {inertia:.2f}")
        swarm.evaluar_enjambre(funcion_objetivo)
        swarm.mover_enjambre(inertia, velocity_limits)
        print(f"Mejor valor global: {swarm.global_best_value}, Mejor posición global: {swarm.global_best}")

    # Mostrar estado final del enjambre
    print("\nEstado final del enjambre:")
    print(swarm)