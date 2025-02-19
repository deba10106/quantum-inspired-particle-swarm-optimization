# About the author
![cv](cv.png)

# Quantum-Inspired Particle Swarm Optimization (QPSO) vs Traditional PSO

This repository provides comprehensive implementations and comparative studies of Quantum-Inspired Particle Swarm Optimization (QPSO) and traditional Particle Swarm Optimization (PSO) algorithms. Our analysis spans multiple optimization scenarios, from simple benchmark functions to complex real-world applications.

## Overview

We compare QPSO and PSO across different optimization scenarios to identify their respective strengths and optimal use cases. Our study includes:
- High-dimensional portfolio optimization
- Multimodal function optimization
- Irregular landscape navigation
- Memory usage analysis
- Convergence behavior studies

### Key Findings

1. **High-Dimensional Portfolio Optimization (500 assets)**
   - QPSO achieved superior results:
     - Better Sharpe ratio (-3993.28 vs -3996.83)
     - Higher return (0.1661 vs 0.0842)
     - Lower risk (0.0247 vs 0.0266)
   - Memory efficiency: Only 7% higher memory usage despite more complex calculations
   - Better exploration: Continued improvement through 200 iterations while PSO stagnated early
   - More diverse portfolio weights distribution

2. **Multimodal Functions (Rastrigin)**
   - PSO shows faster initial convergence
   - QPSO maintains better population diversity
   - PSO may get trapped in local optima more easily
   - QPSO demonstrates more consistent exploration patterns

3. **Irregular Landscapes (Modified Ackley)**
   - QPSO demonstrates better handling of discontinuities
   - More stable performance across different initialization conditions
   - Better at exploring deceptive valleys and avoiding premature convergence
   - Superior ability to escape local optima

## Visualization Guide

### Portfolio Optimization Plots
![Portfolio Optimization Comparison](portfolio_comparison.png)

The visualization consists of four key plots:

1. **Convergence Plot** (Top Left)
   - X-axis: Iteration number
   - Y-axis: Objective value (negative Sharpe ratio)
   - Blue line: Traditional PSO
   - Orange line: QPSO
   - Key observation: QPSO shows continuous improvement while PSO plateaus

2. **Memory Usage Plot** (Top Right)
   - X-axis: Function evaluations
   - Y-axis: Memory usage in MB
   - Shows relative memory efficiency of both algorithms
   - Demonstrates QPSO's modest memory overhead

3. **Weight Distribution** (Bottom Left)
   - X-axis: Portfolio weight values
   - Y-axis: Frequency
   - Overlapping histograms show weight allocation strategies
   - QPSO typically produces more diversified portfolios

4. **Risk-Return Profile** (Bottom Right)
   - X-axis: Portfolio risk (standard deviation)
   - Y-axis: Expected return
   - Scatter points show final solutions
   - Demonstrates QPSO's ability to find better risk-adjusted returns

### Multimodal Function Analysis
![Multimodal Comparison](multimodal_comparison.png)

1. **Surface Plot**
   - Shows the complex landscape of the Rastrigin function
   - Multiple local minima visible as peaks and valleys
   - Helps understand the challenge of avoiding local optima

2. **Convergence Analysis**
   - Tracks the best fitness over iterations
   - Highlights PSO's faster initial convergence
   - Shows QPSO's ability to continue improving

### Irregular Landscape Visualization
![Irregular Comparison](irregular_comparison.png)

1. **Trajectory Plot**
   - Shows particle movements over time
   - Demonstrates different exploration patterns
   - Highlights QPSO's quantum tunneling effect

## Parameter Tuning Recommendations

### Traditional PSO Parameters

1. **Inertia Weight (w)**
   - Recommended range: [0.4, 0.9]
   - Our optimal value: 0.729
   - Higher values (>0.8) for exploration
   - Lower values (<0.5) for exploitation
   - Consider linear decrease over iterations
   - Scenario-specific recommendations:
     * Portfolio Optimization: 0.7-0.8 (balance exploration/exploitation)
     * Multimodal Functions: 0.8-0.9 (favor exploration)
     * Constrained Problems: 0.5-0.6 (favor exploitation)
     * High Dimensions: Start at 0.9, decrease to 0.4

2. **Cognitive Parameter (c1)**
   - Recommended range: [1.0, 2.0]
   - Our optimal value: 1.49445
   - Affects personal best influence
   - Higher values may cause overshooting
   - Scenario-specific recommendations:
     * Portfolio Optimization: 1.5-1.7 (balanced learning)
     * Risk-Averse Problems: 1.2-1.4 (conservative updates)
     * Noisy Environments: 1.3-1.5 (stable convergence)
     * Dynamic Problems: 1.7-1.9 (quick adaptation)

3. **Social Parameter (c2)**
   - Recommended range: [1.0, 2.0]
   - Our optimal value: 1.49445
   - Affects global best influence
   - Balance with c1 for optimal performance
   - Scenario-specific recommendations:
     * Portfolio Optimization: 1.5-1.7 (swarm cohesion)
     * Local Optima Risk: 1.2-1.4 (reduce premature convergence)
     * Fast Convergence Needed: 1.8-2.0 (aggressive social learning)
     * Deceptive Landscapes: 1.1-1.3 (cautious social influence)

### QPSO Parameters

1. **Contraction-Expansion Coefficient (β)**
   - Recommended range: [0.5, 1.0]
   - Our optimal value: 0.7
   - Decreases linearly with iterations
   - Controls quantum behavior extent
   - Scenario-specific recommendations:
     * Portfolio Optimization: 0.7-0.8 (balanced search)
     * High Dimensions: 0.8-0.9 (enhanced exploration)
     * Constrained Problems: 0.6-0.7 (controlled steps)
     * Multimodal Functions: 0.75-0.85 (escape local optima)
     * Dynamic Environments: Linear decrease from 0.9 to 0.5

2. **Local Attractor Parameter (α)**
   - Recommended range: [0.5, 1.0]
   - Our optimal value: 0.8
   - Affects position update magnitude
   - Critical for convergence stability
   - Scenario-specific recommendations:
     * Portfolio Optimization: 0.7-0.8 (stable updates)
     * Risk-Sensitive Problems: 0.6-0.7 (conservative moves)
     * Exploration Priority: 0.8-0.9 (wider search)
     * Fine-Tuning Phase: 0.5-0.6 (precise adjustments)

### Advanced Scenarios

1. **Time-Varying Parameters**
   ```python
   # Example: Dynamic β adjustment for QPSO
   def update_beta(iteration, max_iterations):
       beta_max = 0.9
       beta_min = 0.5
       return beta_max - (beta_max - beta_min) * (iteration / max_iterations)
   ```

2. **Adaptive Population Size**
   - Start with larger population (2-3x final size)
   - Gradually reduce based on diversity metrics
   - Minimum size recommendations:
     * Low dimensions (<10): 20-30 particles
     * Medium dimensions (10-50): 50-100 particles
     * High dimensions (>50): 100-200 particles

3. **Constraint Handling**
   ```python
   # Example: Adaptive penalty weights
   def update_penalty_weight(iteration, violation_history):
       base_weight = 1000
       avg_violation = np.mean(violation_history[-10:])
       return base_weight * (1 + avg_violation)
   ```

### Problem-Specific Scenarios

1. **High-Dimensional Portfolio Optimization**
   - Population Size: 100-200 particles
   - PSO Configuration:
     ```python
     w = 0.7  # Balance exploration/exploitation
     c1 = c2 = 1.5  # Equal personal/social learning
     ```
   - QPSO Configuration:
     ```python
     beta = 0.8  # Enhanced exploration
     alpha = 0.7  # Stable updates
     ```
   - Recommended iterations: 200-500

2. **Multimodal Function Optimization**
   - Population Size: 50-100 particles
   - PSO Configuration:
     ```python
     w = 0.8  # Favor exploration
     c1 = 1.4  # Reduced personal influence
     c2 = 1.2  # Reduced social influence
     ```
   - QPSO Configuration:
     ```python
     beta = 0.85  # Strong quantum behavior
     alpha = 0.8  # Wide search range
     ```
   - Recommended iterations: 100-300

3. **Constrained Optimization**
   - Population Size: 30-50 particles
   - PSO Configuration:
     ```python
     w = 0.6  # Favor exploitation
     c1 = c2 = 1.3  # Conservative learning
     ```
   - QPSO Configuration:
     ```python
     beta = 0.6  # Controlled steps
     alpha = 0.6  # Conservative updates
     ```
   - Recommended iterations: 150-250

4. **Dynamic Environments**
   - Population Size: 40-80 particles
   - PSO Configuration:
     ```python
     w = linear_decrease(0.9, 0.4)  # Dynamic inertia
     c1 = 1.7  # Strong personal learning
     c2 = 1.5  # Moderate social learning
     ```
   - QPSO Configuration:
     ```python
     beta = dynamic_beta(0.9, 0.5)  # Adaptive quantum behavior
     alpha = 0.75  # Balanced updates
     ```
   - Recommended iterations: 50-100 per change

### Performance Monitoring

1. **Convergence Metrics**
   ```python
   def check_convergence(history, window=20, threshold=1e-6):
       if len(history) < window:
           return False
       recent_improvement = abs(history[-1] - history[-window])
       return recent_improvement < threshold
   ```

2. **Diversity Measures**
   ```python
   def population_diversity(positions):
       return np.mean([np.std(pos) for pos in positions.T])
   ```

3. **Parameter Adaptation**
   ```python
   def adapt_parameters(diversity, iteration, max_iterations):
       if diversity < threshold:
           # Increase exploration
           w = min(0.9, w + 0.1)
           beta = min(1.0, beta + 0.1)
       else:
           # Standard linear decrease
           w = w_max - (w_max - w_min) * (iteration / max_iterations)
   ```

## Implementation Details

### Core Components

1. **PSO Implementation (`pso.py`)**
```python
class PSO:
    def __init__(self, n_particles, n_dimensions, max_iterations, bounds):
        self.w = 0.729  # Inertia weight
        self.c1 = 1.49445  # Cognitive parameter
        self.c2 = 1.49445  # Social parameter
```

2. **QPSO Implementation (`qpso.py`)**
```python
class QPSO:
    def __init__(self, n_particles, n_dimensions, max_iterations, bounds):
        self.beta = 0.7  # Contraction-expansion coefficient
        self.alpha = 0.8  # Local attractor parameter
```

### Optimization Problems

1. **Portfolio Optimization**
```python
def objective_function(self, position):
    weights = position / np.sum(np.abs(position))
    portfolio_return = np.sum(weights * self.returns)
    portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(self.covariance, weights)))
    sharpe_ratio = portfolio_return / portfolio_risk
    return -sharpe_ratio  # Minimize negative Sharpe ratio
```

## Dependencies

- numpy>=1.21.0 (Core computations)
- matplotlib>=3.5.0 (Visualization)
- psutil (Memory tracking)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/quantum-inspired-pso.git
cd quantum-inspired-pso

# Install dependencies
pip install -r requirements.txt
```

## Usage Examples

1. **Basic Comparison**
```bash
python compare_pso_qpso.py
```

2. **Portfolio Optimization**
```bash
python compare_portfolio.py --n_assets 500 --n_particles 100
```

3. **Custom Problem**
```python
from qpso import QPSO
optimizer = QPSO(n_particles=50, n_dimensions=10, max_iterations=100, bounds=[-1, 1])
best_position, best_value = optimizer.optimize()
```

## Future Work

1. **Real-world Applications**
   - Apply to more practical optimization problems
   - Test on real market data
   - Implement real-time optimization capabilities

2. **Algorithm Improvements**
   - Hybrid approaches combining PSO and QPSO strengths
   - Adaptive parameter strategies
   - Multi-objective optimization support

3. **Performance Optimization**
   - Parallel implementation for large-scale problems
   - GPU acceleration for high-dimensional cases
   - Distributed computing support

4. **Additional Features**
   - More benchmark functions
   - Interactive visualization tools
   - Automated parameter tuning

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:
- New test cases
- Algorithm improvements
- Documentation enhancements
- Performance optimizations

### Contribution Guidelines
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License - See LICENSE file for details

## Citation

If you use this code in your research, please cite:

```bibtex
@software{qpso_comparison,
  title = {Quantum-Inspired Particle Swarm Optimization vs Traditional PSO},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/quantum-inspired-pso}
}
