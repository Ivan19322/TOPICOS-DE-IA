import random
import time

class TabuSearch:
    def __init__(self, initial_state, tabu_tenure, max_iterations):
        self.n = len(initial_state)  # Tamaño del tablero (n x n)
        self.tabu_tenure = tabu_tenure  # Duración del tabu
        self.max_iterations = max_iterations  # Número máximo de iteraciones
        self.board = initial_state[:]  # Estado inicial del tablero
        self.tabu_list = [[0 for _ in range(self.n)] for _ in range(self.n)]  # Lista tabu para evitar movimientos recientes
        self.best_solution = None  # Mejor solución encontrada
        self.best_cost = float('inf')  # Costo de la mejor solución (inicialmente infinito)
        self.total_moves = 0  # Contador de movimientos totales

    # Calcula el número de ataques entre reinas en el tablero.
    def cost(self, board):
        attacks = 0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if abs(board[i] - board[j]) == abs(i - j):
                    attacks += 1
        return attacks

    # Genera un vecino del tablero actual intercambiando dos reinas.
    def generate_neighbor(self, board):
        neighbor = board[:]
        i, j = random.sample(range(self.n), 2)
        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        return neighbor, (i, j)

    # Realiza la búsqueda tabú para encontrar una solución al problema.
    def search(self):
        current_solution = self.board[:]
        current_cost = self.cost(current_solution)
        total_moves = 0

        for iteration in range(self.max_iterations):
            best_neighbor = None
            best_neighbor_cost = float('inf')
            best_move = None

            # Genera vecinos y selecciona el mejor vecino no tabú o que mejore la mejor solución global.
            for _ in range(self.n * self.n):
                neighbor, move = self.generate_neighbor(current_solution)
                neighbor_cost = self.cost(neighbor)

                if self.tabu_list[move[0]][move[1]] == 0 or neighbor_cost < self.best_cost:
                    if neighbor_cost < best_neighbor_cost:
                        best_neighbor = neighbor
                        best_neighbor_cost = neighbor_cost
                        best_move = move

            if best_neighbor is None:
                break

            current_solution = best_neighbor
            current_cost = best_neighbor_cost
            self.tabu_list[best_move[0]][best_move[1]] = self.tabu_tenure
            total_moves += 1  # Incrementa el contador de movimientos

            # Reduce el tiempo tabú de todos los movimientos en la lista tabú.
            for i in range(self.n):
                for j in range(self.n):
                    if self.tabu_list[i][j] > 0:
                        self.tabu_list[i][j] -= 1

            # Actualiza la mejor solución si se encuentra una mejor.
            if current_cost < self.best_cost:
                self.best_solution = current_solution
                self.best_cost = current_cost

            # Si se encuentra una solución óptima (costo 0), se termina la búsqueda.
            if self.best_cost == 0:
                break

        return self.best_solution, self.best_cost, total_moves

# Función para imprimir el tablero con las reinas colocadas.
def imprimir_tablero(tablero):
    n = len(tablero)
    for i in range(n):
        row = ['.'] * n
        row[tablero[i]] = 'Q'
        print(' '.join(row))
    print()

# Función principal para resolver el problema de las N reinas utilizando búsqueda tabú.
def problema_n_reinas_tabu(initial_state, tabu_tenure=5, max_iterations=1000, num_executions=10):
    best_overall_solution = None
    best_overall_cost = float('inf')
    best_overall_moves = float('inf')
    start_time = time.time()  # Tiempo de inicio

    for _ in range(num_executions):
        tabu_search = TabuSearch(initial_state, tabu_tenure, max_iterations)
        solution, cost, moves = tabu_search.search()

        if cost < best_overall_cost or (cost == best_overall_cost and moves < best_overall_moves):
            best_overall_solution = solution
            best_overall_cost = cost
            best_overall_moves = moves

        if best_overall_cost == 0:
            break

    end_time = time.time()  # Tiempo de finalización
    total_time = end_time - start_time  # Tiempo total

    if best_overall_cost == 0:
        print("Solución óptima encontrada:")
        imprimir_tablero(best_overall_solution)
    else:
        print("No se encontró una solución óptima.")
        if best_overall_solution:
            imprimir_tablero(best_overall_solution)

    print(f"Cantidad total de movimientos: {best_overall_moves}")
    print(f"Tiempo total de ejecución: {total_time:.2f} segundos")

# Ejemplo de uso
initial_state = [0, 3, 2, 1, 6, 5, 4, 7]
problema_n_reinas_tabu(initial_state)