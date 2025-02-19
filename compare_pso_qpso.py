import numpy as np
import matplotlib.pyplot as plt
import time
from pso import PSO
from qpso import QPSO

# Add history tracking to QPSO
def add_history_to_qpso():
    with open('qpso.py', 'r') as file:
        content = file.read()
    if 'self.history = []' not in content:
        content = content.replace(
            'def __init__(self, n_particles, n_dimensions, max_iterations, bounds):',
            'def __init__(self, n_particles, n_dimensions, max_iterations, bounds):\n        self.history = []'
        )
        content = content.replace(
            '# Print progress every 10 iterations',
            '# Store history\n            self.history.append(self.gbest_value)\n\n            # Print progress every 10 iterations'
        )
        content = content.replace(
            'return self.gbest_position, self.gbest_value',
            'return self.gbest_position, self.gbest_value, self.history'
        )
    with open('qpso.py', 'w') as file:
        file.write(content)

# Add history tracking to QPSO
add_history_to_qpso()

# Problem parameters
n_particles = 30
n_dimensions = 2
max_iterations = 100
bounds = [-10, 10]

# Run PSO
print("\nRunning traditional PSO...")
pso = PSO(n_particles, n_dimensions, max_iterations, bounds)
start_time = time.time()
pso_best_position, pso_best_value, pso_history = pso.optimize()
pso_time = time.time() - start_time

# Run QPSO
print("\nRunning Quantum-inspired PSO...")
qpso = QPSO(n_particles, n_dimensions, max_iterations, bounds)
start_time = time.time()
qpso_best_position, qpso_best_value, qpso_history = qpso.optimize()
qpso_time = time.time() - start_time

# Print results
print("\nResults Comparison:")
print(f"{'Algorithm':<10} {'Best Value':<15} {'Time (s)':<10}")
print("-" * 35)
print(f"{'PSO':<10} {pso_best_value:<15.6f} {pso_time:<10.3f}")
print(f"{'QPSO':<10} {qpso_best_value:<15.6f} {qpso_time:<10.3f}")

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(pso_history, label='Traditional PSO', alpha=0.7)
plt.plot(qpso_history, label='Quantum-inspired PSO', alpha=0.7)
plt.yscale('log')
plt.xlabel('Iteration')
plt.ylabel('Best Value (log scale)')
plt.title('Convergence Comparison: PSO vs QPSO')
plt.legend()
plt.grid(True)
plt.savefig('convergence_comparison.png')
plt.close()

print("\nConvergence plot has been saved as 'convergence_comparison.png'")
