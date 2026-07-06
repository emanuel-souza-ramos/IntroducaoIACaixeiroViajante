import matplotlib.pyplot as plt
import random
import math
import numpy as np
import time
import tracemalloc

# Iniciar medição de tempo e memória
tracemalloc.start()
start_time = time.perf_counter()

from random_initial_solution_tsp import random_initial_solution_tsp
from tsp_reading import tsp_reading
from distance_matrix import distance_matrix
from objective_function_tsp import objective_function_tsp
from swap_permutation import swap_permutation
from plotting_route import plotting_route
from two_opt_best_improvement import two_opt_best_improvement

# Input
file_name= "kroA100.tsp"
P=tsp_reading(file_name)
n=len(P)
D=distance_matrix(n,P)

# Parameters
T_0 = 1000
T_f = 1.5
alpha = 0.95
patience = 500 # Num. de iterações sem melhora antes de parar o algoritmo

# Initial setting
num_iter = 0
max_iter = 2000
best_solution_found = []
T = T_0
s_0=random_initial_solution_tsp(n)
f_0 = objective_function_tsp(n,s_0,D)

s_best = s_0.copy()
f_best = f_0
last_improvement_iter = 0

print("Initial solution: ",s_best,f_best)


while num_iter < max_iter:
    s_1 = swap_permutation(s_0,n)
    f_1 = objective_function_tsp(n,s_1,D)
    if (f_1 - f_0 < 0) or (random.random() < math.exp((f_0 - f_1)/T)):
        s_0 = s_1.copy()
        f_0 = f_1
    if f_0 < f_best:
        f_best = f_0
        s_best = s_0.copy()
        last_improvement_iter = num_iter

    if T <= T_f:
        T = T_0
    else:
        T = alpha*T

    best_solution_found.append(f_best)
    
    # Critério de parada antecipada se não houver melhora
    if num_iter - last_improvement_iter > patience:
        print(f"Parada antecipada na iteração {num_iter}: sem melhoras por {patience} iterações.")
        break
        
    num_iter += 1


print("Melhor rota encontrada: ", s_best, f_best)

# Plota o gráfico até a iteração em que a melhor rota foi alcançada
plt.xlabel('Iterations')
plt.ylabel('Objective function values')
plt.plot(best_solution_found[:last_improvement_iter + 1])
plt.savefig("best_solutions.jpg") #save as jpg

print("Simulated annealing found a solution: ",s_best)

# Finalizar medição de tempo e memória
end_time = time.perf_counter()
current_mem, peak_mem = tracemalloc.get_traced_memory()
tracemalloc.stop()

execution_time = end_time - start_time
peak_mem_mb = peak_mem / (1024 * 1024)

print(f"\n--- Estatísticas de Desempenho ---")
print(f"Tempo de execução: {execution_time:.4f} segundos")
print(f"Pico de uso de memória: {peak_mem_mb:.4f} MB")
