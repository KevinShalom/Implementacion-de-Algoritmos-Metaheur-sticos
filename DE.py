import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Función objetivo: Ackley
def ackley(x, a=20, b=0.2, c=2 * np.pi):
    n = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(c * x))
    return -a * np.exp(-b * np.sqrt(sum1 / n)) - np.exp(sum2 / n) + a + np.exp(1)

# Función objetivo: Rastrigin
def rastrigin(x, A=10):
    n = len(x)
    return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))

# Algoritmo DE
def de(func, dim, iter_max, pop_size=30, bounds=(-5, 5), F=0.5, CR=0.9):
    # Inicialización
    pop = np.random.uniform(bounds[0], bounds[1], (pop_size, dim))
    fitness = np.array([func(ind) for ind in pop])
    best_idx = np.argmin(fitness)
    best = pop[best_idx]
    best_value = fitness[best_idx]
    evolution = []

    for _ in range(iter_max):
        for i in range(pop_size):
            # Selección de individuos distintos
            indices = np.arange(pop_size)
            indices = indices[indices != i]
            a, b, c = pop[np.random.choice(indices, 3, replace=False)]

            # Mutación y recombinación
            mutant = np.clip(a + F * (b - c), bounds[0], bounds[1])
            trial = np.array([
                mutant[j] if np.random.rand() < CR else pop[i, j]
                for j in range(dim)
            ])

            # Evaluación del trial vector
            trial_value = func(trial)
            if trial_value < fitness[i]:
                pop[i] = trial
                fitness[i] = trial_value

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
def multiple_runs_de(func):
    best_values = []
    for _ in range(runs):
        _, best_value, _ = de(func, dim, iter_max, pop_size)
        best_values.append(best_value)
    return best_values

# Comparativa entre Ackley y Rastrigin
ackley_best_values = multiple_runs_de(ackley)
rastrigin_best_values = multiple_runs_de(rastrigin)

# Crear un DataFrame para visualizar los valores
df_best_values = pd.DataFrame({
    "Run": range(1, runs + 1),
    "Ackley Best Value": ackley_best_values,
    "Rastrigin Best Value": rastrigin_best_values
})

# Estadísticas
ackley_mean = np.mean(ackley_best_values)
ackley_std = np.std(ackley_best_values)
rastrigin_mean = np.mean(rastrigin_best_values)
rastrigin_std = np.std(rastrigin_best_values)

results = {
    "Ackley": {"mean": ackley_mean, "std": ackley_std},
    "Rastrigin": {"mean": rastrigin_mean, "std": rastrigin_std},
}
ranking = sorted(results.items(), key=lambda x: x[1]["mean"])

# Graficar la evolución de una corrida para cada función
_, _, ackley_evolution = de(ackley, dim, iter_max, pop_size)
_, _, rastrigin_evolution = de(rastrigin, dim, iter_max, pop_size)

plt.figure(figsize=(12, 6))
plt.plot(ackley_evolution, label="Ackley", linewidth=2)
plt.plot(rastrigin_evolution, label="Rastrigin", linewidth=2)
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
