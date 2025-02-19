import numpy as np
import matplotlib.pyplot as plt
import time
import psutil
import os
from pso import PSO
from qpso import QPSO

class PortfolioPSO(PSO):
    def __init__(self, n_particles, n_dimensions, max_iterations, bounds):
        # Initialize memory tracking first
        self.memory_usage = []
        
        # Generate random returns and covariance matrix for our assets
        np.random.seed(42)  # For reproducibility
        self.returns = np.random.normal(0.1, 0.2, n_dimensions)  # Expected returns
        # Create a realistic covariance matrix
        random_matrix = np.random.randn(n_dimensions, n_dimensions)
        self.covariance = np.dot(random_matrix, random_matrix.T)
        self.covariance = self.covariance / np.max(self.covariance) * 0.5  # Scale for realism
        
        # Constraints
        self.min_weight = 0.01  # Minimum weight per asset
        self.risk_tolerance = 0.3  # Maximum acceptable portfolio risk
        
        # Initialize parent class
        super().__init__(n_particles, n_dimensions, max_iterations, bounds)

    def objective_function(self, position):
        """Portfolio optimization objective
        Maximizes Sharpe ratio while respecting constraints
        """
        # Track memory usage
        process = psutil.Process(os.getpid())
        self.memory_usage.append(process.memory_info().rss / 1024 / 1024)  # MB
        
        # Normalize weights to sum to 1
        weights = position / np.sum(np.abs(position))
        
        # Calculate portfolio metrics
        portfolio_return = np.sum(weights * self.returns)
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(self.covariance, weights)))
        
        # Sharpe ratio (assuming risk-free rate = 0 for simplicity)
        sharpe_ratio = portfolio_return / portfolio_risk if portfolio_risk > 0 else -1000
        
        # Penalty for violating constraints
        penalty = 0
        
        # Minimum weight constraint
        min_weight_violation = np.sum(np.maximum(0, self.min_weight - np.abs(weights)))
        penalty += 1000 * min_weight_violation
        
        # Risk constraint
        risk_violation = np.maximum(0, portfolio_risk - self.risk_tolerance)
        penalty += 1000 * risk_violation
        
        # We want to maximize Sharpe ratio, but PSO minimizes, so return negative
        return -sharpe_ratio + penalty

class PortfolioQPSO(QPSO):
    def __init__(self, n_particles, n_dimensions, max_iterations, bounds):
        # Initialize memory tracking first
        self.memory_usage = []
        
        np.random.seed(42)
        self.returns = np.random.normal(0.1, 0.2, n_dimensions)
        random_matrix = np.random.randn(n_dimensions, n_dimensions)
        self.covariance = np.dot(random_matrix, random_matrix.T)
        self.covariance = self.covariance / np.max(self.covariance) * 0.5
        
        self.min_weight = 0.01
        self.risk_tolerance = 0.3
        
        # Initialize parent class
        super().__init__(n_particles, n_dimensions, max_iterations, bounds)

    def objective_function(self, position):
        """Same as PSO version"""
        process = psutil.Process(os.getpid())
        self.memory_usage.append(process.memory_info().rss / 1024 / 1024)
        
        weights = position / np.sum(np.abs(position))
        portfolio_return = np.sum(weights * self.returns)
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(self.covariance, weights)))
        
        sharpe_ratio = portfolio_return / portfolio_risk if portfolio_risk > 0 else -1000
        
        penalty = 0
        min_weight_violation = np.sum(np.maximum(0, self.min_weight - np.abs(weights)))
        penalty += 1000 * min_weight_violation
        
        risk_violation = np.maximum(0, portfolio_risk - self.risk_tolerance)
        penalty += 1000 * risk_violation
        
        return -sharpe_ratio + penalty

# Problem parameters
n_particles = 100
n_dimensions = 500  # Large number of assets
max_iterations = 200
bounds = [-1, 1]  # Allow short selling within limits

# Run comparison
print(f"\nRunning portfolio optimization comparison...")
print(f"Number of assets: {n_dimensions}")
print(f"Number of particles: {n_particles}")
print(f"Maximum iterations: {max_iterations}\n")

# Run PSO
print("Running traditional PSO...")
pso = PortfolioPSO(n_particles, n_dimensions, max_iterations, bounds)
start_time = time.time()
pso_best_position, pso_best_value, pso_history = pso.optimize()
pso_time = time.time() - start_time
pso_memory = pso.memory_usage

# Run QPSO
print("\nRunning Quantum-inspired PSO...")
qpso = PortfolioQPSO(n_particles, n_dimensions, max_iterations, bounds)
start_time = time.time()
qpso_best_position, qpso_best_value, qpso_history = qpso.optimize()
qpso_time = time.time() - start_time
qpso_memory = qpso.memory_usage

# Calculate portfolio metrics for best solutions
def calculate_portfolio_metrics(weights, returns, covariance):
    weights = weights / np.sum(np.abs(weights))
    portfolio_return = np.sum(weights * returns)
    portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(covariance, weights)))
    sharpe_ratio = portfolio_return / portfolio_risk if portfolio_risk > 0 else -1000
    return portfolio_return, portfolio_risk, sharpe_ratio

pso_return, pso_risk, pso_sharpe = calculate_portfolio_metrics(
    pso_best_position, pso.returns, pso.covariance
)
qpso_return, qpso_risk, qpso_sharpe = calculate_portfolio_metrics(
    qpso_best_position, qpso.returns, qpso.covariance
)

# Print results
print("\nResults Comparison:")
print(f"{'Metric':<20} {'PSO':<15} {'QPSO':<15}")
print("-" * 50)
print(f"{'Best Sharpe Ratio':<20} {-pso_best_value:<15.4f} {-qpso_best_value:<15.4f}")
print(f"{'Portfolio Return':<20} {pso_return:<15.4f} {qpso_return:<15.4f}")
print(f"{'Portfolio Risk':<20} {pso_risk:<15.4f} {qpso_risk:<15.4f}")
print(f"{'Time (s)':<20} {pso_time:<15.4f} {qpso_time:<15.4f}")
print(f"{'Peak Memory (MB)':<20} {max(pso_memory):<15.4f} {max(qpso_memory):<15.4f}")

# Plotting
plt.figure(figsize=(15, 10))

# Convergence plot
plt.subplot(2, 2, 1)
plt.plot(pso_history, label='Traditional PSO', alpha=0.7)
plt.plot(qpso_history, label='Quantum-inspired PSO', alpha=0.7)
plt.xlabel('Iteration')
plt.ylabel('Objective Value')
plt.title('Convergence Comparison')
plt.legend()
plt.grid(True)

# Memory usage plot
plt.subplot(2, 2, 2)
plt.plot(pso_memory, label='Traditional PSO', alpha=0.7)
plt.plot(qpso_memory, label='Quantum-inspired PSO', alpha=0.7)
plt.xlabel('Function Evaluations')
plt.ylabel('Memory Usage (MB)')
plt.title('Memory Usage Comparison')
plt.legend()
plt.grid(True)

# Weight distribution
plt.subplot(2, 2, 3)
plt.hist(pso_best_position / np.sum(np.abs(pso_best_position)), 
         bins=50, alpha=0.5, label='PSO')
plt.hist(qpso_best_position / np.sum(np.abs(qpso_best_position)), 
         bins=50, alpha=0.5, label='QPSO')
plt.xlabel('Weight')
plt.ylabel('Frequency')
plt.title('Portfolio Weight Distribution')
plt.legend()
plt.grid(True)

# Risk-Return plot
plt.subplot(2, 2, 4)
plt.scatter(pso_risk, pso_return, label='PSO', s=100)
plt.scatter(qpso_risk, qpso_return, label='QPSO', s=100)
plt.xlabel('Portfolio Risk')
plt.ylabel('Portfolio Return')
plt.title('Risk-Return Profile')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('portfolio_comparison.png')
plt.close()

print("\nDetailed comparison plot has been saved as 'portfolio_comparison.png'")
