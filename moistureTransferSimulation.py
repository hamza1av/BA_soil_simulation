import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

class MoistureTransferSimulation:
    def __init__(self, a=1, b=1, M1=40, M2=40, N=500, T1=1, soil_type='sandy'):
        """
        Initialize the moisture transfer simulation.
        
        Parameters:
        a, b - width and depth of the region
        M1, M2 - number of grid points in x and z directions
        N - number of time steps
        T1 - final moment of time
        soil_type - type of soil ('sandy', 'light_loamy', 'medium_loamy', 'heavy_loamy')
        """
        # Grid parameters
        self.a = a
        self.b = b
        self.M1 = M1
        self.M2 = M2
        self.N = N
        self.T1 = T1
        
        self.hx = a / M1
        self.hz = b / M2
        self.tau = T1 / N
        
        # Soil parameters
        self.soil_type = soil_type
        
        # Set parameters based on soil type
        if soil_type == 'sandy':
            self.Kf = 0.5  # filtration coefficient
            self.W_star = 0.17  # maximal molecular moisture capacity
            self.m = 0.31  # complete moisture capacity of soil
            self.mu = 0.042  # empiric parameter for PS function
            self.n = 1.5  # empiric parameter for PS function
        elif soil_type == 'light_loamy':
            self.Kf = 0.3
            self.W_star = 0.23
            self.m = 0.35
            self.mu = 0.035
            self.n = 2.0
        elif soil_type == 'medium_loamy':
            self.Kf = 0.2
            self.W_star = 0.28
            self.m = 0.38
            self.mu = 0.028
            self.n = 2.5
        elif soil_type == 'heavy_loamy':
            self.Kf = 0.1
            self.W_star = 0.32
            self.m = 0.40
            self.mu = 0.022
            self.n = 3.0
        
        self.hk = 1.25  # height of capillary rising
        
        # Initialize matrices for volume humidity and piezometric head
        self.W = np.zeros((M1+1, M2+1, N+1))
        self.U = np.zeros((M1+1, M2+1, N+1))
        
        # Newton's method parameters
        self.epsilon = 1e-7
        self.max_iter = 100
    
    def set_initial_condition(self, initial_moisture=0.2):
        """Set initial moisture distribution."""
        self.W[:,:,0] = initial_moisture
        
        # Update piezometric head based on initial moisture
        for i in range(self.M1+1):
            for j in range(self.M2+1):
                self.U[i,j,0] = self.calculate_piezometric_head(self.W[i,j,0]) - j*self.hz
    
    def set_point_sources(self, sources):
        """
        Set irrigation sources.
        
        Parameters:
        sources - list of tuples [(x1, z1, intensity1), (x2, z2, intensity2), ...]
        """
        self.sources = sources
    
    def moisture_transfer_coefficient(self, W):
        """Calculate moisture transfer coefficient using Averianov's formula."""
        if W <= self.W_star:
            return 0
        return self.Kf * ((W - self.W_star) / (self.m - self.W_star)) ** 3.5
    
    def calculate_piezometric_head(self, W):
        """Calculate piezometric head using the hydrophysical characteristic."""
        if W <= self.W_star:
            return -1000  # Very low value for dry soil
        
        # PS = self.mu * ((self.m - self.W_star) / (W - self.W_star)) ** self.n
        PS = self.mu * ((self.m - self.W_star) / (W - self.W_star)) ** self.n
        return PS
    
    def source_function(self, x, z, t):
        """Function representing irrigation sources."""
        f = 0
        for src_x, src_z, intensity in self.sources:
            # Simple Gaussian representation of point source
            sigma = 0.05
            f += intensity * np.exp(-((x - src_x)**2 + (z - src_z)**2) / (2 * sigma**2))
        return f
    
    def solve(self):
        """Solve the moisture transfer equation using two-step symmetrized algorithm."""
        # Set up subsets for two-step algorithm
        subset1 = []  # i + j + n is even
        subset2 = []  # i + j + n is odd
        
        for i in range(1, self.M1):
            for j in range(1, self.M2):
                if (i + j) % 2 == 0:
                    subset1.append((i, j))
                else:
                    subset2.append((i, j))
        
        # Time stepping
        for n in range(self.N):
            t = n * self.tau
            
            # First solve for subset 1 (explicit scheme)
            for i, j in subset1:
                x = i * self.hx
                z = -j * self.hz
                
                # Current values
                W_c = self.W[i, j, n]
                K_c = self.moisture_transfer_coefficient(W_c)
                
                # Adjacent values
                W_e = self.W[i+1, j, n]
                W_w = self.W[i-1, j, n]
                W_n = self.W[i, j-1, n]
                W_s = self.W[i, j+1, n]
                
                K_e = self.moisture_transfer_coefficient(W_e)
                K_w = self.moisture_transfer_coefficient(W_w)
                K_n = self.moisture_transfer_coefficient(W_n)
                K_s = self.moisture_transfer_coefficient(W_s)
                
                # Average K values at cell interfaces
                K_xe = 0.5 * (K_c + K_e)
                K_xw = 0.5 * (K_c + K_w)
                K_zn = 0.5 * (K_c + K_n)
                K_zs = 0.5 * (K_c + K_s)
                
                # Compute derivatives of piezometric head
                U_e = self.calculate_piezometric_head(W_e) - j*self.hz
                U_w = self.calculate_piezometric_head(W_w) - j*self.hz
                U_n = self.calculate_piezometric_head(W_n) - (j-1)*self.hz
                U_s = self.calculate_piezometric_head(W_s) - (j+1)*self.hz
                U_c = self.calculate_piezometric_head(W_c) - j*self.hz
                
                # Discretized Laplacian
                L_h = (K_xe * (U_e - U_c) - K_xw * (U_c - U_w)) / (self.hx**2) + \
                      (K_zn * (U_n - U_c) - K_zs * (U_c - U_s)) / (self.hz**2) + \
                      K_zs / self.hz + self.source_function(x, z, t)
                
                # Update W using explicit scheme
                self.W[i, j, n+1] = W_c + self.tau * L_h
                
                # Ensure moisture stays within physical limits
                self.W[i, j, n+1] = max(self.W_star * 0.5, min(self.m, self.W[i, j, n+1]))
                
                # Update piezometric head
                self.U[i, j, n+1] = self.calculate_piezometric_head(self.W[i, j, n+1]) - j*self.hz
            
            # Apply boundary conditions
            self.apply_boundary_conditions(n+1)
            
            # Then solve for subset 2 (implicit scheme using Newton's method)
            for i, j in subset2:
                x = i * self.hx
                z = -j * self.hz
                
                # Initial guess for Newton's method
                w_curr = self.W[i, j, n]
                
                for iter_count in range(self.max_iter):
                    # Current moisture and coefficient
                    K_c = self.moisture_transfer_coefficient(w_curr)
                    
                    # Adjacent values (already computed for subset 1 or from previous time step)
                    W_e = self.W[i+1, j, n+1]
                    W_w = self.W[i-1, j, n+1]
                    W_n = self.W[i, j-1, n+1]
                    W_s = self.W[i, j+1, n+1]
                    
                    K_e = self.moisture_transfer_coefficient(W_e)
                    K_w = self.moisture_transfer_coefficient(W_w)
                    K_n = self.moisture_transfer_coefficient(W_n)
                    K_s = self.moisture_transfer_coefficient(W_s)
                    
                    # Average K values at cell interfaces
                    K_xe = 0.5 * (K_c + K_e)
                    K_xw = 0.5 * (K_c + K_w)
                    K_zn = 0.5 * (K_c + K_n)
                    K_zs = 0.5 * (K_c + K_s)
                    
                    # Compute derivatives of piezometric head
                    U_e = self.calculate_piezometric_head(W_e) - j*self.hz
                    U_w = self.calculate_piezometric_head(W_w) - j*self.hz
                    U_n = self.calculate_piezometric_head(W_n) - (j-1)*self.hz
                    U_s = self.calculate_piezometric_head(W_s) - (j+1)*self.hz
                    U_c = self.calculate_piezometric_head(w_curr) - j*self.hz
                    
                    # Discretized Laplacian
                    L_h = (K_xe * (U_e - U_c) - K_xw * (U_c - U_w)) / (self.hx**2) + \
                          (K_zn * (U_n - U_c) - K_zs * (U_c - U_s)) / (self.hz**2) + \
                          K_zs / self.hz + self.source_function(x, z, t)
                    
                    # Residual function F
                    F = w_curr - self.W[i, j, n] - self.tau * L_h
                    
                    # Derivative of residual function F'
                    # Calculate numerical derivative of L_h with respect to w_curr
                    delta = 1e-6
                    w_plus = w_curr + delta
                    
                    K_c_plus = self.moisture_transfer_coefficient(w_plus)
                    U_c_plus = self.calculate_piezometric_head(w_plus) - j*self.hz
                    
                    K_xe_plus = 0.5 * (K_c_plus + K_e)
                    K_xw_plus = 0.5 * (K_c_plus + K_w)
                    K_zn_plus = 0.5 * (K_c_plus + K_n)
                    K_zs_plus = 0.5 * (K_c_plus + K_s)
                    
                    L_h_plus = (K_xe_plus * (U_e - U_c_plus) - K_xw_plus * (U_c_plus - U_w)) / (self.hx**2) + \
                               (K_zn_plus * (U_n - U_c_plus) - K_zs_plus * (U_c_plus - U_s)) / (self.hz**2) + \
                               K_zs_plus / self.hz + self.source_function(x, z, t)
                    
                    dL_h = (L_h_plus - L_h) / delta
                    F_prime = 1 - self.tau * dL_h
                    
                    # Newton's method update
                    w_next = w_curr - F / F_prime
                    
                    # Check convergence
                    if abs(w_next - w_curr) < self.epsilon:
                        break
                    
                    w_curr = w_next
                
                # Ensure moisture stays within physical limits
                self.W[i, j, n+1] = max(self.W_star * 0.5, min(self.m, w_curr))
                
                # Update piezometric head
                self.U[i, j, n+1] = self.calculate_piezometric_head(self.W[i, j, n+1]) - j*self.hz
            
            # Apply boundary conditions again
            self.apply_boundary_conditions(n+1)
            
            if n % 50 == 0:
                print(f"Completed time step {n}/{self.N} ({n/self.N*100:.1f}%)")
    
    def apply_boundary_conditions(self, n):
        """Apply boundary conditions for the moisture equation."""
        # On the surface z = 0 (j = 0)
        for i in range(self.M1+1):
            # U = 0 (no deeper sources from the surface)
            self.U[i, 0, n] = 0
            # Back-calculate W from U
            # This is approximate since we need to solve inverse function
            # Simple approach: try different W values
            best_w = self.W_star
            min_diff = float('inf')
            for w_try in np.linspace(self.W_star, self.m, 100):
                u_try = self.calculate_piezometric_head(w_try) - 0
                diff = abs(u_try - 0)
                if diff < min_diff:
                    min_diff = diff
                    best_w = w_try
            self.W[i, 0, n] = best_w
        
        # On the lower boundary z = -b (j = M2)
        for i in range(self.M1+1):
            # U = -z (hydrostatic pressure distribution)
            self.U[i, self.M2, n] = self.b
            # Back-calculate W from U
            best_w = self.W_star
            min_diff = float('inf')
            for w_try in np.linspace(self.W_star, self.m, 100):
                u_try = self.calculate_piezometric_head(w_try) - self.b
                diff = abs(u_try - self.b)
                if diff < min_diff:
                    min_diff = diff
                    best_w = w_try
            self.W[i, self.M2, n] = best_w
        
        # Left and right vertical boundaries (i = 0 and i = M1)
        for j in range(self.M2+1):
            # Zero flux boundary condition (Neumann)
            self.W[0, j, n] = self.W[1, j, n]
            self.W[self.M1, j, n] = self.W[self.M1-1, j, n]
            
            self.U[0, j, n] = self.calculate_piezometric_head(self.W[0, j, n]) - j*self.hz
            self.U[self.M1, j, n] = self.calculate_piezometric_head(self.W[self.M1, j, n]) - j*self.hz
    
    def plot_results(self, timesteps=None):
        """
        Plot the moisture distribution at specified timesteps.
        
        Parameters:
        timesteps - list of time steps to plot, if None plots initial, middle and final
        """
        if timesteps is None:
            timesteps = [0, self.N//2, self.N]
        
        for n in timesteps:
            if n > self.N:
                continue
                
            # Create meshgrid for plotting
            x = np.linspace(0, self.a, self.M1+1)
            z = np.linspace(0, -self.b, self.M2+1)
            X, Z = np.meshgrid(x, z)
            
            # Extract moisture data for this timestep
            W_data = self.W[:,:,n].T
            
            # Create figure
            fig = plt.figure(figsize=(12, 8))
            
            # 3D surface plot
            ax1 = fig.add_subplot(121, projection='3d')
            surf = ax1.plot_surface(X, Z, W_data, cmap=cm.viridis, linewidth=0, antialiased=False)
            ax1.set_xlabel('x (m)')
            ax1.set_ylabel('z (m)')
            ax1.set_zlabel('Volume Humidity')
            ax1.set_title(f'Moisture Distribution at t = {n*self.tau:.3f}')
            fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)
            
            # 2D contour plot
            ax2 = fig.add_subplot(122)
            contour = ax2.contourf(X, Z, W_data, 20, cmap=cm.viridis)
            ax2.set_xlabel('x (m)')
            ax2.set_ylabel('z (m)')
            ax2.set_title(f'Moisture Contour at t = {n*self.tau:.3f}')
            
            # Plot source positions
            for src_x, src_z, _ in self.sources:
                ax2.plot(src_x, src_z, 'ro', markersize=8)
                
            fig.colorbar(contour, ax=ax2)
            
            plt.tight_layout()
            plt.show()
    
    def animate_results(self, interval=20):
        """
        Create an animation of the moisture distribution.
        
        Parameters:
        interval - every nth time step to include in animation
        """
        import matplotlib.animation as animation
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create meshgrid for plotting
        x = np.linspace(0, self.a, self.M1+1)
        z = np.linspace(0, -self.b, self.M2+1)
        X, Z = np.meshgrid(x, z)
        
        # Initial contour plot
        contour = ax.contourf(X, Z, self.W[:,:,0].T, 20, cmap=cm.viridis)
        title = ax.set_title(f'Moisture Distribution at t = 0')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('z (m)')
        
        # Plot source positions
        for src_x, src_z, _ in self.sources:
            ax.plot(src_x, src_z, 'ro', markersize=8)
        
        fig.colorbar(contour, ax=ax)
        
        # Animation function
        def update_plot(frame):
            ax.clear()
            n = frame * interval
            if n > self.N:
                n = self.N
            
            contour = ax.contourf(X, Z, self.W[:,:,n].T, 20, cmap=cm.viridis)
            title = ax.set_title(f'Moisture Distribution at t = {n*self.tau:.3f}')
            ax.set_xlabel('x (m)')
            ax.set_ylabel('z (m)')
            
            # Re-plot source positions
            for src_x, src_z, _ in self.sources:
                ax.plot(src_x, src_z, 'ro', markersize=8)
            
            return contour, title
        
        frames = self.N // interval + 1
        ani = animation.FuncAnimation(fig, update_plot, frames=frames, interval=200, blit=False)
        
        plt.tight_layout()
        plt.show()
        
        return ani


# Example usage
if __name__ == "__main__":
    # Create simulation instance
    sim = MoistureTransferSimulation(a=1, b=1, M1=40, M2=40, N=500, T1=1, soil_type='sandy')
    
    # Set initial condition (dry soil)
    sim.set_initial_condition(initial_moisture=0.18)
    
    # Set three irrigation sources
    sources = [
        (0.25, -0.1, 0.5),  # (x, z, intensity)
        (0.5, -0.15, 0.6),
        (0.75, -0.1, 0.5)
    ]
    sim.set_point_sources(sources)
    
    # Run simulation
    print("Starting simulation...")
    sim.solve()
    print("Simulation complete!")
    
    # Plot results at different timesteps
    sim.plot_results(timesteps=[0, 100, 200, 300, 400, 500])
    
    # Create animation
    ani = sim.animate_results(interval=25)
