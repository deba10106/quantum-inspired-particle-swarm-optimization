import numpy as np
import matplotlib.pyplot as plt
import time
from pso import PSO
from qpso import QPSO

class IrregularPSO(PSO):
    def objective_function(self, position):
        """Modified Ackley function with discontinuities and noise
        Creates a very challenging landscape with:
        1. Many local minima (from Ackley function)
        2. Discontinuities (from step function)
        3. Noise (random perturbations)
        4. Deceptive global structure
        """
        # Basic Ackley function
        a = 20
        b = 0.2
        c = 2 * np.pi
        d = len(position)
        
        sum_sq = np.sum(position**2)
        sum_cos = np.sum(np.cos(c * position))
        
        term1 = -a * np.exp(-b * np.sqrt(sum_sq/d))
        term2 = -np.exp(sum_cos/d)
        ackley = term1 + term2 + a + np.exp(1)
        
        # Add discontinuities
        steps = np.floor(position)
        discontinuities = np.sum(np.abs(steps)) * 2
        
        # Add noise based on position
        noise = np.sin(np.sum(position) * 5) * 0.5
        
        # Combine all components
        result = ackley + discontinuities + noise
        
        # Add deceptive valley
        distance_from_deceptive = np.sum((position - 2.5)**2)
        deceptive_valley = 10 / (1 + distance_from_deceptive)
        
        return result - deceptive_valley

class IrregularQPSO(QPSO):
    def objective_function(self, position):
        """Same as above"""
        a = 20
        b = 0.2
        c = 2 * np.pi
        d = len(position)
        
        sum_sq = np.sum(position**2)
        sum_cos = np.sum(np.cos(c * position))
        
        term1 = -a * np.exp(-b * np.sqrt(sum_sq/d))
        term2 = -np.exp(sum_cos/d)
        ackley = term1 + term2 + a + np.exp(1)
        
        steps = np.floor(position)
        discontinuities = np.sum(np.abs(steps)) * 2
        
        noise = np.sin(np.sum(position) * 5) * 0.5
        
        result = ackley + discontinuities + noise
        
        distance_from_deceptive = np.sum((position - 2.5)**2)
        deceptive_valley = 10 / (1 + distance_from_deceptive)
        
        return result - deceptive_valley

# Problem parameters
n_particles = 50
n_dimensions = 5  # Moderate dimensionality
max_iterations = 300  # More iterations for complex landscape
bounds = [-5, 5]

# Run multiple trials
n_trials = 10
pso_histories = []
qpso_histories = []
pso_times = []
qpso_times = []
pso_final_values = []
qpso_final_values = []
pso_best_positions = []
qpso_best_positions = []

print(f"\nRunning comparison over {n_trials} trials...")
print(f"Problem dimensions: {n_dimensions}")
print(f"Number of particles: {n_particles}")
print(f"Maximum iterations: {max_iterations}\n")

for trial in range(n_trials):
    print(f"\nTrial {trial + 1}/{n_trials}")
    
    # Run PSO
    print("Running traditional PSO...")
    pso = IrregularPSO(n_particles, n_dimensions, max_iterations, bounds)
    start_time = time.time()
    pos, val, history = pso.optimize()
    pso_time = time.time() - start_time
    pso_histories.append(history)
    pso_times.append(pso_time)
    pso_final_values.append(val)
    pso_best_positions.append(pos)
    
    # Run QPSO
    print("\nRunning Quantum-inspired PSO...")
    qpso = IrregularQPSO(n_particles, n_dimensions, max_iterations, bounds)
    start_time = time.time()
    pos, val, history = qpso.optimize()
    qpso_time = time.time() - start_time
    qpso_histories.append(history)
    qpso_times.append(qpso_time)
    qpso_final_values.append(val)
    qpso_best_positions.append(pos)

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
plt.figure(figsize=(15, 10))

# Main convergence plot
plt.subplot(2, 2, 1)
plt.plot(pso_avg_history, label='Traditional PSO', alpha=0.7)
plt.plot(qpso_avg_history, label='Quantum-inspired PSO', alpha=0.7)
plt.yscale('log')
plt.xlabel('Iteration')
plt.ylabel('Best Value (log scale)')
plt.title('Average Convergence Over All Trials')
plt.legend()
plt.grid(True)

# Box plot of final values
plt.subplot(2, 2, 2)
plt.boxplot([pso_final_values, qpso_final_values], labels=['PSO', 'QPSO'])
plt.ylabel('Final Value')
plt.title('Distribution of Final Values')
plt.grid(True)

# Scatter plot of best positions (first 2 dimensions)
plt.subplot(2, 2, 3)
pso_positions = np.array(pso_best_positions)
qpso_positions = np.array(qpso_best_positions)
plt.scatter(pso_positions[:, 0], pso_positions[:, 1], label='PSO', alpha=0.6)
plt.scatter(qpso_positions[:, 0], qpso_positions[:, 1], label='QPSO', alpha=0.6)
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('Best Positions Found (2D Projection)')
plt.legend()
plt.grid(True)

# Convergence stability
plt.subplot(2, 2, 4)
pso_std = np.std(pso_histories, axis=0)
qpso_std = np.std(qpso_histories, axis=0)
plt.plot(pso_std, label='PSO Variance', alpha=0.7)
plt.plot(qpso_std, label='QPSO Variance', alpha=0.7)
plt.xlabel('Iteration')
plt.ylabel('Standard Deviation')
plt.title('Search Stability (Lower is More Stable)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('irregular_comparison.png')
plt.close()

print("\nDetailed comparison plot has been saved as 'irregular_comparison.png'")
