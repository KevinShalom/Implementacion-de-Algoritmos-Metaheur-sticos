import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Función objetivo: Himmelblau
def himmelblau(x):
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2

# Función objetivo: Booth
def booth(x):
    return (x[0] + 2 * x[1] - 7)**2 + (2 * x[0] + x[1] - 5)**2

# Algoritmo Genético (GA)
def ga(func, dim, iter_max, pop_size=30, bounds=(-5, 5), mutation_rate=0.1, crossover_rate=0.8):
    # Inicialización
    pop = np.random.uniform(bounds[0], bounds[1], (pop_size, dim))
    fitness = np.array([func(ind) for ind in pop])
    best_idx = np.argmin(fitness)
    best = pop[best_idx]
    best_value = fitness[best_idx]
    evolution = []

    for _ in range(iter_max):
        # Selección por torneo
        parents = []
        for _ in range(pop_size):
            i, j = np.random.choice(pop_size, 2, replace=False)
            if fitness[i] < fitness[j]:
                parents.append(pop[i])
            else:
                parents.append(pop[j])
        parents = np.array(parents)

        # Crossover de un punto
        offspring = []
        for i in range(0, pop_size, 2):
            if np.random.rand() < crossover_rate:
                crossover_point = np.random.randint(1, dim)
                child1 = np.concatenate([parents[i][:crossover_point], parents[i+1][crossover_point:]])
                child2 = np.concatenate([parents[i+1][:crossover_point], parents[i][crossover_point:]])
                offspring.extend([child1, child2])
            else:
                offspring.extend([parents[i], parents[i+1]])
        offspring = np.array(offspring)

        # Mutación
        for i in range(pop_size):
            if np.random.rand() < mutation_rate:
                mutation_idx = np.random.randint(dim)
                offspring[i][mutation_idx] += np.random.uniform(-1, 1)
                offspring[i] = np.clip(offspring[i], bounds[0], bounds[1])

        # Evaluar nueva generación
        pop = offspring
        fitness = np.array([func(ind) for ind in pop])

        # Actualizar el mejor global
        best_idx = np.argmin(fitness)
        best = pop[best_idx]
        best_value = fitness[best_idx]
        evolution.append(best_value)

    return best, best_value, evolution

# Parámetros generales
dim = 2
iter_max = 500
pop_size = 30
runs = 10

# Función para ejecutar múltiples corridas y obtener estadísticas
def multiple_runs_ga(func):
    best_values = []
    for _ in range(runs):
        _, best_value, _ = ga(func, dim, iter_max, pop_size)
        best_values.append(best_value)
    return best_values

# Comparativa entre Himmelblau y Booth
himmelblau_best_values = multiple_runs_ga(himmelblau)
booth_best_values = multiple_runs_ga(booth)

# Crear un DataFrame para visualizar los valores
df_best_values = pd.DataFrame({
    "Run": range(1, runs + 1),
    "Himmelblau Best Value": himmelblau_best_values,
    "Booth Best Value": booth_best_values
})

# Estadísticas
himmelblau_mean = np.mean(himmelblau_best_values)
himmelblau_std = np.std(himmelblau_best_values)
booth_mean = np.mean(booth_best_values)
booth_std = np.std(booth_best_values)

results = {
    "Himmelblau": {"mean": himmelblau_mean, "std": himmelblau_std},
    "Booth": {"mean": booth_mean, "std": booth_std},
}
ranking = sorted(results.items(), key=lambda x: x[1]["mean"])

# Graficar la evolución de una corrida para cada función
_, _, himmelblau_evolution = ga(himmelblau, dim, iter_max, pop_size)
_, _, booth_evolution = ga(booth, dim, iter_max, pop_size)

plt.figure(figsize=(12, 6))
plt.plot(himmelblau_evolution, label="Himmelblau", linewidth=2)
plt.plot(booth_evolution, label="Booth", linewidth=2)
plt.title("Evolución del valor de la función objetivo (una corrida)")
plt.xlabel("Iteraciones")
plt.ylabel("Valor de la función objetivo")
plt.legend()
plt.grid()
plt.show()

print("\n--- Mejor valor en cada corrida (10 ejecuciones) ---")
print(df_best_values)

print("\n--- Estadísticas finales (promedio y desviación estándar) ---")
for func, stats in results.items():
    print(f"{func}:")
    print(f"  - Mean: {stats['mean']}")
    print(f"  - Standard Deviation: {stats['std']}")

print("\n--- Ranking basado en promedio ---")
for i, (func, stats) in enumerate(ranking, 1):
    print(f"{i}. {func} (Mean: {stats['mean']:.2e})")
