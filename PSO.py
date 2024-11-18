import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Función objetivo: Sphere
def sphere(x):
    return np.sum(x**2)

# Función objetivo: Rosenbrock
def rosenbrock(x):
    return sum(100 * (x[i+1] - x[i]**2)**2 + (x[i] - 1)**2 for i in range(len(x)-1))

# Algoritmo PSO
def pso(func, dim, iter_max, swarm_size=30, bounds=(-5, 5)):
    swarm = np.random.uniform(bounds[0], bounds[1], (swarm_size, dim))
    velocities = np.random.uniform(-1, 1, (swarm_size, dim))
    personal_best = swarm.copy()
    personal_best_value = np.array([func(p) for p in swarm])
    global_best = personal_best[np.argmin(personal_best_value)]
    global_best_value = np.min(personal_best_value)

    w = 0.5  # Inercia
    c1 = c2 = 1.5  # Coeficientes cognitivo y social
    evolution = []

    for _ in range(iter_max):
        for i in range(swarm_size):
            r1, r2 = np.random.random(dim), np.random.random(dim)
            velocities[i] = (
                w * velocities[i]
                + c1 * r1 * (personal_best[i] - swarm[i])
                + c2 * r2 * (global_best - swarm[i])
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

dim = 2
iter_max = 500
swarm_size = 30
runs = 10

def multiple_runs(func):
    best_values = []
    for _ in range(runs):
        _, best_value, _ = pso(func, dim, iter_max, swarm_size)
        best_values.append(best_value)
    return best_values

sphere_best_values = multiple_runs(sphere)
rosenbrock_best_values = multiple_runs(rosenbrock)

df_best_values = pd.DataFrame({
    "Run": range(1, runs + 1),
    "Sphere Best Value": sphere_best_values,
    "Rosenbrock Best Value": rosenbrock_best_values
})

sphere_mean = np.mean(sphere_best_values)
sphere_std = np.std(sphere_best_values)
rosenbrock_mean = np.mean(rosenbrock_best_values)
rosenbrock_std = np.std(rosenbrock_best_values)

results = {
    "Sphere": {"mean": sphere_mean, "std": sphere_std},
    "Rosenbrock": {"mean": rosenbrock_mean, "std": rosenbrock_std},
}
ranking = sorted(results.items(), key=lambda x: x[1]["mean"])

_, _, sphere_evolution = pso(sphere, dim, iter_max, swarm_size)
_, _, rosenbrock_evolution = pso(rosenbrock, dim, iter_max, swarm_size)

plt.figure(figsize=(12, 6))
plt.plot(sphere_evolution, label="Sphere", linewidth=2)
plt.plot(rosenbrock_evolution, label="Rosenbrock", linewidth=2)
plt.title("Evolución del valor de la función objetivo")
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
