import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# *** Funciones matemáticas ***
# PSO: Sphere y Rosenbrock
def sphere(x):
    return np.sum(x**2)

def rosenbrock(x):
    return sum(100 * (x[i+1] - x[i]**2)**2 + (x[i] - 1)**2 for i in range(len(x)-1))

# DE: Ackley y Rastrigin
def ackley(x, a=20, b=0.2, c=2 * np.pi):
    n = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(c * x))
    return -a * np.exp(-b * np.sqrt(sum1 / n)) - np.exp(sum2 / n) + a + np.exp(1)

def rastrigin(x, A=10):
    n = len(x)
    return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))

# GA: Himmelblau y Booth
def himmelblau(x):
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2

def booth(x):
    return (x[0] + 2 * x[1] - 7)**2 + (2 * x[0] + x[1] - 5)**2

# *** Algoritmos ***
# PSO
def pso(func, dim, iter_max, swarm_size=30, bounds=(-5, 5)):
    swarm = np.random.uniform(bounds[0], bounds[1], (swarm_size, dim))
    velocities = np.random.uniform(-1, 1, (swarm_size, dim))
    personal_best = swarm.copy()
    personal_best_value = np.array([func(p) for p in swarm])
    global_best = personal_best[np.argmin(personal_best_value)]
    global_best_value = np.min(personal_best_value)
    evolution = []

    for _ in range(iter_max):
        for i in range(swarm_size):
            r1, r2 = np.random.random(dim), np.random.random(dim)
            velocities[i] = (
                0.5 * velocities[i]
                + 1.5 * r1 * (personal_best[i] - swarm[i])
                + 1.5 * r2 * (global_best - swarm[i])
            )
            swarm[i] += velocities[i]
            swarm[i] = np.clip(swarm[i], bounds[0], bounds[1])
        values = np.array([func(p) for p in swarm])
        for i in range(swarm_size):
            if values[i] < personal_best_value[i]:
                personal_best[i] = swarm[i]
                personal_best_value[i] = values[i]
        if np.min(values) < global_best_value:
            global_best = swarm[np.argmin(values)]
            global_best_value = np.min(values)
        evolution.append(global_best_value)

    return global_best, global_best_value, evolution

# DE
def de(func, dim, iter_max, pop_size=30, bounds=(-5, 5), F=0.5, CR=0.9):
    pop = np.random.uniform(bounds[0], bounds[1], (pop_size, dim))
    fitness = np.array([func(ind) for ind in pop])
    best_idx = np.argmin(fitness)
    best = pop[best_idx]
    best_value = fitness[best_idx]
    evolution = []

    for _ in range(iter_max):
        for i in range(pop_size):
            indices = np.arange(pop_size)
            indices = indices[indices != i]
            a, b, c = pop[np.random.choice(indices, 3, replace=False)]
            mutant = np.clip(a + F * (b - c), bounds[0], bounds[1])
            trial = np.array([
                mutant[j] if np.random.rand() < CR else pop[i, j]
                for j in range(dim)
            ])
            trial_value = func(trial)
            if trial_value < fitness[i]:
                pop[i] = trial
                fitness[i] = trial_value
        best_idx = np.argmin(fitness)
        best = pop[best_idx]
        best_value = fitness[best_idx]
        evolution.append(best_value)

    return best, best_value, evolution

# GA
def ga(func, dim, iter_max, pop_size=30, bounds=(-5, 5), mutation_rate=0.1, crossover_rate=0.8):
    pop = np.random.uniform(bounds[0], bounds[1], (pop_size, dim))
    fitness = np.array([func(ind) for ind in pop])
    best_idx = np.argmin(fitness)
    best = pop[best_idx]
    best_value = fitness[best_idx]
    evolution = []

    for _ in range(iter_max):
        parents = []
        for _ in range(pop_size):
            i, j = np.random.choice(pop_size, 2, replace=False)
            if fitness[i] < fitness[j]:
                parents.append(pop[i])
            else:
                parents.append(pop[j])
        parents = np.array(parents)
        offspring = []
        for i in range(0, pop_size, 2):
            if np.random.rand() < crossover_rate:
                crossover_point = np.random.randint(1, dim)
                child1 = np.concatenate([parents[i][:crossover_point], parents[i+1][crossover_point:]])
                child2 = np.concatenate([parents[i+1][:crossover_point], parents[i][:crossover_point]])
                offspring.extend([child1, child2])
            else:
                offspring.extend([parents[i], parents[i+1]])
        offspring = np.array(offspring)
        for i in range(pop_size):
            if np.random.rand() < mutation_rate:
                mutation_idx = np.random.randint(dim)
                offspring[i][mutation_idx] += np.random.uniform(-1, 1)
                offspring[i] = np.clip(offspring[i], bounds[0], bounds[1])
        pop = offspring
        fitness = np.array([func(ind) for ind in pop])
        best_idx = np.argmin(fitness)
        best = pop[best_idx]
        best_value = fitness[best_idx]
        evolution.append(best_value)

    return best, best_value, evolution

# *** Ejecución y análisis ***
dim = 2
iter_max = 500
pop_size = 30
runs = 10

# Algoritmos y funciones
algorithms = {
    "PSO": [(sphere, "Sphere"), (rosenbrock, "Rosenbrock")],
    "DE": [(ackley, "Ackley"), (rastrigin, "Rastrigin")],
    "GA": [(himmelblau, "Himmelblau"), (booth, "Booth")]
}

results = []

for algo_name, functions in algorithms.items():
    for func, func_name in functions:
        best_values = []
        for _ in range(runs):
            if algo_name == "PSO":
                _, best_value, _ = pso(func, dim, iter_max, pop_size)
            elif algo_name == "DE":
                _, best_value, _ = de(func, dim, iter_max, pop_size)
            elif algo_name == "GA":
                _, best_value, _ = ga(func, dim, iter_max, pop_size)
            best_values.append(best_value)
        mean = np.mean(best_values)
        std = np.std(best_values)
        results.append({"Algorithm": algo_name, "Function": func_name, "Mean": mean, "Std": std})

# Crear DataFrame de resultados
df_results = pd.DataFrame(results)

# Ranking general
ranking = df_results.groupby("Algorithm")["Mean"].mean().sort_values()

# Mostrar resultados
print("\n--- Resultados por Algoritmo y Función ---")
print(df_results)

print("\n--- Ranking General de Algoritmos (Promedio de Medias) ---")
print(ranking)

# *** Graficar resultados por algoritmo y función ***
# Gráfica de rendimiento por función
plt.figure(figsize=(14, 8))
for algo_name in df_results["Algorithm"].unique():
    algo_data = df_results[df_results["Algorithm"] == algo_name]
    plt.bar(
        algo_data["Function"],
        algo_data["Mean"],
        label=algo_name,
        alpha=0.7
    )

plt.title("Rendimiento Promedio de Cada Algoritmo en Sus Funciones", fontsize=14)
plt.xlabel("Función", fontsize=12)
plt.ylabel("Valor Promedio del Mejor Resultado", fontsize=12)
plt.yscale("log")  # Escala logarítmica para mejor visualización
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.show()

# Gráfica de comparación general entre algoritmos
plt.figure(figsize=(10, 6))
ranking.plot(kind="bar", color="skyblue", alpha=0.8)
plt.title("Ranking General de Algoritmos (Promedio de Medias)", fontsize=14)
plt.xlabel("Algoritmo", fontsize=12)
plt.ylabel("Media Promedio del Mejor Resultado", fontsize=12)
