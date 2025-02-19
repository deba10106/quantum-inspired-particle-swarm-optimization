import numpy as np

class QPSO:
    def __init__(self, n_particles, n_dimensions, max_iterations, bounds):
        self.history = []
        self.n_particles = n_particles
        self.n_dimensions = n_dimensions
        self.max_iterations = max_iterations
        self.bounds = bounds
        
        # Initialize particles with random positions
        self.positions = np.random.uniform(
            bounds[0], bounds[1], 
            (n_particles, n_dimensions)
        )
        
        # Initialize personal best positions and values
        self.pbest_positions = self.positions.copy()
        self.pbest_values = np.array([self.objective_function(p) for p in self.positions])
        
        # Initialize global best
        self.gbest_index = np.argmin(self.pbest_values)
        self.gbest_position = self.pbest_positions[self.gbest_index].copy()
        self.gbest_value = self.pbest_values[self.gbest_index]

    def objective_function(self, position):
        """Simple sphere function as an example objective"""
        return np.sum(position**2)

    def update_particle(self, position, iteration):
        """Update particle position using quantum behavior"""
        beta = np.random.uniform(0, 1)
        mbest = np.mean(self.pbest_positions, axis=0)
        phi = np.random.uniform(0, 2 * np.pi)
        
        # Quantum state - using wave function
        L = 2 * np.abs(mbest - position)
        u = np.random.uniform(0, 1, self.n_dimensions)
        
        if np.random.random() > 0.5:
            position_new = mbest + beta * L * np.log(1/u)
        else:
            position_new = mbest - beta * L * np.log(1/u)
            
        # Ensure the particle stays within bounds
        position_new = np.clip(position_new, self.bounds[0], self.bounds[1])
        return position_new

    def optimize(self):
        """Main optimization loop"""
        for iteration in range(self.max_iterations):
            # Update each particle
            for i in range(self.n_particles):
                # Update particle position
                self.positions[i] = self.update_particle(self.positions[i], iteration)
                
                # Evaluate new position
                current_value = self.objective_function(self.positions[i])
                
                # Update personal best
                if current_value < self.pbest_values[i]:
                    self.pbest_positions[i] = self.positions[i].copy()
                    self.pbest_values[i] = current_value
                    
                    # Update global best
                    if current_value < self.gbest_value:
                        self.gbest_position = self.positions[i].copy()
                        self.gbest_value = current_value
            
            # Store history
            self.history.append(self.gbest_value)

            # Print progress every 10 iterations
            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{self.max_iterations}, Best value: {self.gbest_value:.6f}")
        
        return self.gbest_position, self.gbest_value, self.history

# Example usage
if __name__ == "__main__":
    # Problem parameters
    n_particles = 30
    n_dimensions = 2
    max_iterations = 100
    bounds = [-10, 10]  # Search space bounds

    # Initialize and run QPSO
    qpso = QPSO(n_particles, n_dimensions, max_iterations, bounds)
    best_position, best_value = qpso.optimize()

    print("\nOptimization finished!")
    print(f"Best position found: {best_position}")
    print(f"Best value found: {best_value:.6f}")
