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

    def __repr__(self):
        """
        Representación en texto del enjambre.
        """
        return "\n".join(str(particle) for particle in self.particles)

# Ejemplo de uso:
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

    # Mostrar el enjambre y sus partículas
    print(swarm)