import numpy as np
import matplotlib.pyplot as plt
import time
from pso import PSO
from qpso import QPSO

class MultimodalPSO(PSO):
    def objective_function(self, position):
        """Rastrigin function - a complex multimodal problem
        Global minimum at x = 0 with f(x) = 0
        Has many local minima arranged in a regular lattice"""
        A = 10
        return A * len(position) + np.sum(position**2 - A * np.cos(2 * np.pi * position))

class MultimodalQPSO(QPSO):
    def objective_function(self, position):
        """Rastrigin function - same as above"""
        A = 10
        return A * len(position) + np.sum(position**2 - A * np.cos(2 * np.pi * position))

# Problem parameters
n_particles = 50  # Increased particle count for better exploration
n_dimensions = 10  # Increased dimensions to make problem more challenging
max_iterations = 200  # More iterations for complex landscape
bounds = [-5.12, 5.12]  # Standard bounds for Rastrigin function

# Run multiple trials to account for randomness
n_trials = 5
pso_histories = []
qpso_histories = []
pso_times = []
qpso_times = []
pso_final_values = []
qpso_final_values = []

print(f"\nRunning comparison over {n_trials} trials...")
print(f"Problem dimensions: {n_dimensions}")
print(f"Number of particles: {n_particles}")
print(f"Maximum iterations: {max_iterations}\n")

for trial in range(n_trials):
    print(f"\nTrial {trial + 1}/{n_trials}")
    
    # Run PSO
    print("Running traditional PSO...")
    pso = MultimodalPSO(n_particles, n_dimensions, max_iterations, bounds)
    start_time = time.time()
    _, pso_best_value, pso_history = pso.optimize()
    pso_time = time.time() - start_time
    pso_histories.append(pso_history)
    pso_times.append(pso_time)
    pso_final_values.append(pso_best_value)
    
    # Run QPSO
    print("\nRunning Quantum-inspired PSO...")
    qpso = MultimodalQPSO(n_particles, n_dimensions, max_iterations, bounds)
    start_time = time.time()
    _, qpso_best_value, qpso_history = qpso.optimize()
    qpso_time = time.time() - start_time
    qpso_histories.append(qpso_history)
    qpso_times.append(qpso_time)
    qpso_final_values.append(qpso_best_value)

# Calculate average histories
pso_avg_history = np.mean(pso_histories, axis=0)
qpso_avg_history = np.mean(qpso_histories, axis=0)

# Print statistical results
print("\nResults Comparison:")
print(f"{'Algorithm':<10} {'Best Value':<15} {'Avg Value':<15} {'Std Dev':<15} {'Avg Time (s)':<10}")
print("-" * 65)
print(f"{'PSO':<10} {min(pso_final_values):<15.6f} {np.mean(pso_final_values):<15.6f} {np.std(pso_final_values):<15.6f} {np.mean(pso_times):<10.3f}")
print(f"{'QPSO':<10} {min(qpso_final_values):<15.6f} {np.mean(qpso_final_values):<15.6f} {np.std(qpso_final_values):<15.6f} {np.mean(qpso_times):<10.3f}")

# Plotting
plt.figure(figsize=(12, 8))

# Main convergence plot
plt.subplot(2, 1, 1)
plt.plot(pso_avg_history, label='Traditional PSO', alpha=0.7)
plt.plot(qpso_avg_history, label='Quantum-inspired PSO', alpha=0.7)
plt.yscale('log')
plt.xlabel('Iteration')
plt.ylabel('Best Value (log scale)')
plt.title(f'Convergence Comparison on Rastrigin Function (n={n_dimensions})')
plt.legend()
plt.grid(True)

# Box plot of final values
plt.subplot(2, 1, 2)
plt.boxplot([pso_final_values, qpso_final_values], labels=['PSO', 'QPSO'])
plt.ylabel('Final Value')
plt.title('Distribution of Final Values Across Trials')
plt.grid(True)

plt.tight_layout()
plt.savefig('multimodal_comparison.png')
plt.close()

print("\nConvergence plot has been saved as 'multimodal_comparison.png'")
