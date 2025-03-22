import random
import math
import time

class SimulatedAnnealing:
    def __init__(self, initial_state, initial_temp, cooling_rate, max_iterations):
        """
        Inicializa el algoritmo de recocido simulado.
        
        :param initial_state: Lista representando el estado inicial del tablero.
        :param initial_temp: Temperatura inicial del algoritmo.
        :param cooling_rate: Factor de enfriamiento (valor entre 0 y 1).
        :param max_iterations: Número máximo de iteraciones.
        """
        self.n = len(initial_state)
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.max_iterations = max_iterations
        self.board = initial_state[:]
        self.best_solution = None
        self.best_cost = float('inf')

    def cost(self, board):
        """
        Calcula el número de ataques entre reinas en el tablero.
        
        :param board: Lista representando la posición de las reinas.
        :return: Número de ataques.
        """
        row_conflicts = [0] * self.n
        diag1_conflicts = [0] * (2 * self.n)
        diag2_conflicts = [0] * (2 * self.n)
        
        for i in range(self.n):
            row_conflicts[board[i]] += 1
            diag1_conflicts[i + board[i]] += 1
            diag2_conflicts[i - board[i] + self.n] += 1

        attacks = 0
        for conflicts in (row_conflicts, diag1_conflicts, diag2_conflicts):
            attacks += sum(c * (c - 1) // 2 for c in conflicts if c > 1)

        return attacks

    def generate_neighbor(self, board):
        """
        Genera un vecino al intercambiar dos posiciones en el tablero.
        
        :param board: Estado actual del tablero.
        :return: Nuevo tablero vecino.
        """
        neighbor = board[:]
        i, j = random.sample(range(self.n), 2)
        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        return neighbor

    def search(self):
        """
        Realiza la búsqueda de solución usando recocido simulado.
        
        :return: Mejor solución encontrada y costo asociado.
        """
        current_solution = self.board[:]
        current_cost = self.cost(current_solution)
        best_solution = current_solution[:]
        best_cost = current_cost
        temperature = self.initial_temp

        for iteration in range(self.max_iterations):
            if temperature <= 0:
                break

            neighbor = self.generate_neighbor(current_solution)
            neighbor_cost = self.cost(neighbor)
            cost_diff = neighbor_cost - current_cost

            if cost_diff < 0 or random.uniform(0, 1) < math.exp(-cost_diff / temperature):
                current_solution = neighbor
                current_cost = neighbor_cost

            if current_cost < best_cost:
                best_solution = current_solution[:]
                best_cost = current_cost

            temperature *= self.cooling_rate

            if best_cost == 0:
                break

        return best_solution, best_cost

def validar_tablero_inicial(initial_state):
    """
    Valida que el estado inicial sea válido.
    
    :param initial_state: Lista representando el estado inicial del tablero.
    :raises ValueError: Si el estado inicial no es válido.
    """
    n = len(initial_state)
    if not all(0 <= pos < n for pos in initial_state):
        raise ValueError("El estado inicial contiene valores fuera del rango permitido.")

def imprimir_tablero(tablero):
    """
    Imprime visualmente el tablero con las reinas.
    
    :param tablero: Lista representando las posiciones de las reinas.
    """
    n = len(tablero)
    for i in range(n):
        row = ['.'] * n
        row[tablero[i]] = 'Q'
        print(' '.join(row))
    print()

def problema_n_reinas_recocido_simulado(initial_state, initial_temp=1000, cooling_rate=0.995, max_iterations=1000, num_executions=10):
    """
    Resuelve el problema de las N reinas usando recocido simulado.
    
    :param initial_state: Estado inicial del tablero.
    :param initial_temp: Temperatura inicial.
    :param cooling_rate: Tasa de enfriamiento.
    :param max_iterations: Número máximo de iteraciones por ejecución.
    :param num_executions: Número de ejecuciones del algoritmo.
    """
    validar_tablero_inicial(initial_state)

    best_overall_solution = None
    best_overall_cost = float('inf')
    start_time = time.time()

    for _ in range(num_executions):
        sa = SimulatedAnnealing(initial_state, initial_temp, cooling_rate, max_iterations)
        solution, cost = sa.search()

        if cost < best_overall_cost:
            best_overall_solution = solution
            best_overall_cost = cost

        if best_overall_cost == 0:
            break

    end_time = time.time()
    total_time = end_time - start_time

    if best_overall_cost == 0:
        print("Solución óptima encontrada:")
        imprimir_tablero(best_overall_solution)
    else:
        print("No se encontró una solución óptima.")
        if best_overall_solution:
            imprimir_tablero(best_overall_solution)

   
    print(f"Tiempo total de ejecución: {total_time:.2f} segundos")

# Ejemplo de uso
initial_state = [0, 3, 2, 1, 6, 5, 4, 7]
problema_n_reinas_recocido_simulado(initial_state)
