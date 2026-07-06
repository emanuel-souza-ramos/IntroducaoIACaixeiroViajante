import sys
import os
import time
import tracemalloc
import numpy as np
import tsplib95
import matplotlib.pyplot as plt
from ga import Ga

# Heurística para resolver o problema do Caixeiro Viajante (TSP) usando Algoritmo Genético.
# O Algoritmo Genético busca a melhor rota aproximada simulando a evolução biológica de uma
# população de caminhos (rotas válidas) que competem e se reproduzem de geração em geração.

def main():
    # Inicia o rastreamento de memória e tempo para medir tudo usado na execução
    tracemalloc.start()
    start_time = time.perf_counter()

    # Lista arquivos .tsp no diretório atual
    tsp_files = [f for f in os.listdir('.') if f.endswith('.tsp')]
    
    # Permite passar o mapa por linha de comando ou escolher interativamente
    if len(sys.argv) > 1:
        selected_file = sys.argv[1]
    else:
        print("Arquivos .tsp encontrados no diretorio:")
        for idx, f in enumerate(tsp_files):
            print(f"{idx + 1}: {f}")
        
        try:
            choice = input("Escolha o numero do arquivo (padrao 1): ").strip()
            if choice == "":
                selected_file = tsp_files[0]
            else:
                selected_file = tsp_files[int(choice) - 1]
        except (ValueError, IndexError, KeyboardInterrupt, EOFError):
            selected_file = tsp_files[0]

    print(f"\n[INFO] Carregando o arquivo de mapa: {selected_file}...")
    problem = tsplib95.load(selected_file)
    nodes = list(problem.get_nodes())
    n = len(nodes)
    
    print(f"[INFO] Mapa carregado com sucesso. Total de cidades (nos): {n}")
    
    # Extrai as coordenadas X e Y de cada cidade do arquivo de mapa
    city_pos_list = np.array([problem.node_coords[node] for node in nodes])
    
    # Constrói a matriz de distância com os pesos das arestas oficiais do formato TSPLIB (EUC_2D)
    city_dist_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            city_dist_mat[i, j] = problem.get_weight(nodes[i], nodes[j])
            
    print("[INFO] Matriz de distancias gerada. Iniciando o Algoritmo Genetico...")
    
    # Inicializa o Algoritmo Genético com os hiperparâmetros:
    # - individual_num: Tamanho da população de caminhos simultâneos
    # - gen_num: Número de gerações para buscar a convergência (máximo de 2000)
    # - mutate_prob: Probabilidade de aplicar mutação em cada nova rota gerada
    # - patience: Número de gerações consecutivas sem melhora antes de parar
    # - tournament_size: Tamanho do torneio para a seleção dos indivíduos
    ga = Ga(city_dist_mat, individual_num=100, gen_num=2000, mutate_prob=0.2, patience=300, tournament_size=5)
    result_list, fitness_list = ga.train()
    result = result_list[-1]
    
    # Mapeia os índices das cidades na melhor rota final para as suas coordenadas geométricas correspondentes
    result_pos_list = city_pos_list[result, :]
    
    print(f"\n[SUCESSO] Execucao concluida!")
    print(f"Melhor distancia encontrada (Fitness final): {fitness_list[-1]:.2f}")
    
    # Gráfico 1: Plot da Rota Física Encontrada
    plt.figure(figsize=(8, 6))
    plt.plot(result_pos_list[:, 0], result_pos_list[:, 1], 'o-r', label='Trecho da Rota')
    # Destaca o ponto inicial/final do caixeiro
    plt.plot(result_pos_list[0, 0], result_pos_list[0, 1], 'g^', markersize=12, label='Ponto Inicial/Final')
    plt.title(f"Melhor Rota Encontrada - Mapa: {selected_file}")
    plt.xlabel("Coordenada X (Leste-Oeste)")
    plt.ylabel("Coordenada Y (Norte-Sul)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='best')
    
    # Gráfico 2: Evolução da Aptidão (Fitness)
    plt.figure(figsize=(8, 5))
    plt.plot(fitness_list, color='blue', linewidth=2, label='Distancia da Melhor Rota')
    plt.title("Curva de Aprendizado / Evolucao da Aptidao (Fitness)")
    plt.xlabel("Geracao (Iteracao)")
    plt.ylabel("Distancia Total do Caminho (Menor e melhor)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='best')
    
    # Captura o tempo total e pico de memória antes do plt.show() (que bloqueia a execução)
    end_time = time.perf_counter()
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    elapsed_time = end_time - start_time
    peak_memory_mb = peak_memory / (1024 * 1024)
    
    print("\n" + "=" * 50)
    print("ESTATÍSTICAS DE PERFORMANCE DA EXECUÇÃO")
    print("=" * 50)
    print(f"Tempo total de processamento: {elapsed_time:.3f} segundos")
    print(f"Pico de consumo de memória: {peak_memory_mb:.3f} MB")
    print("=" * 50 + "\n")
    
    plt.show()

if __name__ == "__main__":
    main()
