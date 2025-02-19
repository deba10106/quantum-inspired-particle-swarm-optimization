import numpy as np

class PSO:
    def __init__(self, n_particles, n_dimensions, max_iterations, bounds):
        self.n_particles = n_particles
        self.n_dimensions = n_dimensions
        self.max_iterations = max_iterations
        self.bounds = bounds
        
        # Initialize particles with random positions and velocities
        self.positions = np.random.uniform(
            bounds[0], bounds[1], 
            (n_particles, n_dimensions)
        )
        self.velocities = np.random.uniform(
            -abs(bounds[1] - bounds[0]), 
            abs(bounds[1] - bounds[0]), 
            (n_particles, n_dimensions)
        )
        
        # PSO parameters
        self.w = 0.729  # Inertia weight
        self.c1 = 1.49445  # Cognitive parameter
        self.c2 = 1.49445  # Social parameter
        
        # Initialize personal best positions and values
        self.pbest_positions = self.positions.copy()
        self.pbest_values = np.array([self.objective_function(p) for p in self.positions])
        
        # Initialize global best
        self.gbest_index = np.argmin(self.pbest_values)
        self.gbest_position = self.pbest_positions[self.gbest_index].copy()
        self.gbest_value = self.pbest_values[self.gbest_index]
        
        # History for plotting
        self.history = []

    def objective_function(self, position):
        """Simple sphere function as an example objective"""
        return np.sum(position**2)

    def update_particle(self, i, position, velocity):
        """Update particle position using standard PSO equations"""
        r1, r2 = np.random.rand(2)
        
        # Update velocity
        velocity_new = (self.w * velocity + 
                       self.c1 * r1 * (self.pbest_positions[i] - position) +
                       self.c2 * r2 * (self.gbest_position - position))
        
        # Update position
        position_new = position + velocity_new
        
        # Ensure the particle stays within bounds
        position_new = np.clip(position_new, self.bounds[0], self.bounds[1])
        velocity_new = np.clip(velocity_new, -abs(self.bounds[1] - self.bounds[0]), 
                             abs(self.bounds[1] - self.bounds[0]))
        
        return position_new, velocity_new

    def optimize(self):
        """Main optimization loop"""
        for iteration in range(self.max_iterations):
            # Update each particle
            for i in range(self.n_particles):
                # Update particle position and velocity
                self.positions[i], self.velocities[i] = self.update_particle(
                    i, self.positions[i], self.velocities[i]
                )
                
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
