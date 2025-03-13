import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import matplotlib.animation as animation
from matplotlib import cm

def richards_equation_solver(
    depth=1.0,          # Total soil depth [m]
    n_cells=100,        # Number of spatial cells
    total_time=24.0,    # Total simulation time [hours]
    dt=0.1,             # Time step [hours]
    Ks=0.1,             # Saturated hydraulic conductivity [m/hour]
    theta_r=0.1,        # Residual water content
    theta_s=0.4,        # Saturated water content
    alpha=3.0,          # van Genuchten parameter [1/m]
    n=1.5,              # van Genuchten parameter
    initial_condition="dry",  # Initial condition ('dry', 'wet', 'custom')
    top_boundary="constant",  # Top boundary condition ('constant', 'rainfall', 'evaporation')
    bottom_boundary="free_drainage",  # Bottom boundary condition ('free_drainage', 'constant', 'zero_flux')
):
    """
    Solve the Richards equation for 1D unsaturated water flow in soil
    using the mixed form of the equation and a fully implicit finite volume method.
    
    Richards equation (mixed form):
    C(h) * ∂h/∂t = ∂/∂z [K(h) * (∂h/∂z + 1)]
    
    where:
    - h is the pressure head [m]
    - C(h) is the specific moisture capacity [1/m]
    - K(h) is the hydraulic conductivity [m/hour]
    - z is the depth coordinate [m] (positive downward)
    """
    
    # Convert parameters to appropriate units
    m = 1 - 1/n  # van Genuchten parameter
    
    # Set up space discretization
    dz = depth / n_cells
    z = np.linspace(dz/2, depth - dz/2, n_cells)  # Cell centers
    
    # Time setup
    n_time_steps = int(total_time / dt)
    time = np.linspace(0, total_time, n_time_steps+1)
    
    # Initial pressure head conditions
    if initial_condition == "dry":
        h = -1.0 * np.ones(n_cells)  # Dry condition (negative pressure head)
    elif initial_condition == "wet":
        h = -0.1 * np.ones(n_cells)  # Wet condition
    elif initial_condition == "custom":
        # Linear gradient from wet at top to dry at bottom
        h = np.linspace(-0.1, -1.0, n_cells)
    else:
        h = -0.5 * np.ones(n_cells)  # Default moderate condition
    
    # Storage for results
    h_profile = np.zeros((n_time_steps+1, n_cells))
    h_profile[0, :] = h
    
    # Top boundary condition value
    if top_boundary == "constant":
        h_top = -0.1  # Constant pressure head [m]
    else:
        h_top = -0.5  # Default value
    
    # Bottom boundary condition value
    if bottom_boundary == "constant":
        h_bottom = -1.0  # Constant pressure head [m]
    else:
        h_bottom = None  # Will be determined by selected boundary condition
    
    # Time loop
    for t in range(n_time_steps):
        # Copy current solution
        h_old = h.copy()
        
        # Newton-Raphson iteration
        max_iter = 20
        tolerance = 1e-6
        
        for iteration in range(max_iter):
            # Calculate van Genuchten relationships for current h
            # Saturation function Se
            Se = np.zeros_like(h)
            idx_negative = h < 0
            Se[idx_negative] = (1 + (alpha * np.abs(h[idx_negative])) ** n) ** (-m)
            Se[~idx_negative] = 1.0
            
            # Water content
            theta = theta_r + (theta_s - theta_r) * Se
            
            # Hydraulic conductivity
            K = Ks * Se ** 0.5 * (1 - (1 - Se ** (1/m)) ** m) ** 2
            
            # Specific water capacity
            C = np.zeros_like(h)
            C[idx_negative] = (alpha * m * n * (theta_s - theta_r) * 
                               (alpha * np.abs(h[idx_negative])) ** (n-1) * 
                               Se[idx_negative] ** (1/m) / (1 - Se[idx_negative] ** (1/m)))
            
            # Calculate interface hydraulic conductivities
            K_interface = np.zeros(n_cells + 1)
            K_interface[1:n_cells] = 0.5 * (K[:-1] + K[1:])  # arithmetic mean for interior
            
            # Setup system matrix and right-hand side
            main_diag = np.zeros(n_cells)
            upper_diag = np.zeros(n_cells-1)
            lower_diag = np.zeros(n_cells-1)
            rhs = np.zeros(n_cells)
            
            # Interior cells
            for i in range(1, n_cells-1):
                # Flux terms
                lower_diag[i-1] = -dt * K_interface[i] / dz**2  # coefficient for h[i-1]
                upper_diag[i] = -dt * K_interface[i+1] / dz**2  # coefficient for h[i+1]
                main_diag[i] = (C[i] + dt * (K_interface[i] + K_interface[i+1]) / dz**2)  # coefficient for h[i]
                
                # Right-hand side
                rhs[i] = C[i] * h_old[i] - dt * (K_interface[i+1] - K_interface[i]) / dz  # gravity term
            
            # Top boundary (cell 0)
            if top_boundary == "constant":
                # Dirichlet boundary condition
                main_diag[0] = 1.0
                upper_diag[0] = 0.0
                rhs[0] = h_top
            elif top_boundary == "rainfall":
                # Flux boundary condition (rainfall)
                rain_rate = 0.01  # m/hour
                K_interface[0] = K[0]  # Use the cell's hydraulic conductivity
                main_diag[0] = C[0] + dt * K_interface[1] / dz**2
                upper_diag[0] = -dt * K_interface[1] / dz**2
                rhs[0] = C[0] * h_old[0] - dt * K_interface[1] / dz + dt * rain_rate / dz
            elif top_boundary == "evaporation":
                # Flux boundary condition (evaporation)
                evap_rate = 0.005  # m/hour
                K_interface[0] = K[0]
                main_diag[0] = C[0] + dt * K_interface[1] / dz**2
                upper_diag[0] = -dt * K_interface[1] / dz**2
                rhs[0] = C[0] * h_old[0] - dt * K_interface[1] / dz - dt * evap_rate / dz
            
            # Bottom boundary (cell n_cells-1)
            if bottom_boundary == "free_drainage":
                # Free drainage boundary condition
                K_interface[n_cells] = K[n_cells-1]
                main_diag[n_cells-1] = C[n_cells-1] + dt * K_interface[n_cells-1] / dz**2
                lower_diag[n_cells-2] = -dt * K_interface[n_cells-1] / dz**2
                rhs[n_cells-1] = C[n_cells-1] * h_old[n_cells-1] + dt * K_interface[n_cells] / dz 
            elif bottom_boundary == "constant":
                # Dirichlet boundary condition
                main_diag[n_cells-1] = 1.0
                lower_diag[n_cells-2] = 0.0
                rhs[n_cells-1] = h_bottom
            elif bottom_boundary == "zero_flux":
                # No-flux boundary condition
                K_interface[n_cells] = 0
                main_diag[n_cells-1] = C[n_cells-1] + dt * K_interface[n_cells-1] / dz**2
                lower_diag[n_cells-2] = -dt * K_interface[n_cells-1] / dz**2
                rhs[n_cells-1] = C[n_cells-1] * h_old[n_cells-1]
            
            # Assemble the sparse matrix
            A = diags(
                [main_diag, upper_diag, lower_diag],
                [0, 1, -1],
                shape=(n_cells, n_cells)
            )
            
            # Solve the system
            h_new = spsolve(A, rhs)
            
            # Check convergence
            err = np.max(np.abs(h_new - h))
            h = h_new
            
            if err < tolerance:
                break
                
        # Store the result
        h_profile[t+1, :] = h
    
    # Convert pressure head to water content for output
    theta_profile = np.zeros_like(h_profile)
    for i in range(n_time_steps+1):
        Se = np.zeros_like(h_profile[i, :])
        idx_negative = h_profile[i, :] < 0
        Se[idx_negative] = (1 + (alpha * np.abs(h_profile[i, idx_negative])) ** n) ** (-m)
        Se[~idx_negative] = 1.0
        theta_profile[i, :] = theta_r + (theta_s - theta_r) * Se
    
    return z, time, h_profile, theta_profile

def plot_results(z, time, h_profile, theta_profile, plot_times=None):
    """
    Plot the results of the Richards equation simulation.
    
    Parameters:
    - z: depth coordinates
    - time: simulation time points
    - h_profile: pressure head profiles over time
    - theta_profile: water content profiles over time
    - plot_times: specific time points to plot (in hours)
    """
    if plot_times is None:
        # Default: plot initial, 25%, 50%, 75% and final time
        idx = [0, 
               int(len(time)/4), 
               int(len(time)/2), 
               int(3*len(time)/4), 
               len(time)-1]
        plot_times = time[idx]
    else:
        # Find closest time indices
        idx = [np.abs(time - t).argmin() for t in plot_times]
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot pressure head profiles
    for i, t_idx in enumerate(idx):
        ax1.plot(h_profile[t_idx, :], z, label=f't = {time[t_idx]:.1f} hr')
    
    ax1.set_xlabel('Pressure Head [m]')
    ax1.set_ylabel('Depth [m]')
    ax1.set_title('Pressure Head Profiles')
    ax1.invert_yaxis()  # Invert y-axis to have surface at top
    ax1.grid(True)
    ax1.legend()
    
    # Plot water content profiles
    for i, t_idx in enumerate(idx):
        ax2.plot(theta_profile[t_idx, :], z, label=f't = {time[t_idx]:.1f} hr')
    
    ax2.set_xlabel('Water Content [m³/m³]')
    ax2.set_ylabel('Depth [m]')
    ax2.set_title('Water Content Profiles')
    ax2.invert_yaxis()  # Invert y-axis to have surface at top
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    return fig

def create_heatmap_animation(z, time, theta_profile, interval=100, save_path=None):
    """
    Create an animated heatmap showing the evolution of water content over time.
    
    Parameters:
    - z: depth coordinates [m]
    - time: time points [hours]
    - theta_profile: water content profiles over time
    - interval: time between animation frames [ms]
    - save_path: if provided, save animation to this file
    
    Returns:
    - Animation object
    """
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Prepare time steps for display
    # Use fewer frames for smoother animation
    ani_frames = 100
    frame_idx = np.linspace(0, len(time)-1, ani_frames, dtype=int)
    
    # Calculate extent for imshow
    extent = [0, time[-1], z[-1], z[0]]  # [xmin, xmax, ymin, ymax]
    
    # Calculate vmin and vmax for consistent colorbar
    vmin = np.min(theta_profile)
    vmax = np.max(theta_profile)
    
    # Create initial heatmap
    heatmap = ax.imshow(
        theta_profile[:, :].T,  # Transpose to have depth on y-axis and time on x-axis
        aspect='auto',
        extent=extent,
        origin='upper',
        cmap='viridis',
        vmin=vmin,
        vmax=vmax,
        interpolation='nearest'
    )
    
    # Add colorbar
    cbar = fig.colorbar(heatmap, ax=ax, label='Water Content [m³/m³]')
    
    # Configure axes
    ax.set_xlabel('Time [hours]')
    ax.set_ylabel('Depth [m]')
    ax.set_title('Soil Moisture Content Evolution')
    
    # Add a vertical line to show current time
    time_line = ax.axvline(x=0, color='red', linestyle='-', linewidth=2)
    
    time_text = ax.text(0.02, 0.02, 'Time: 0.0 hr', transform=ax.transAxes, 
                        color='white', fontweight='bold', backgroundcolor='black')
    
    def init():
        time_line.set_xdata(0)
        time_text.set_text('Time: 0.0 hr')
        return time_line, time_text
    
    def update(frame):
        current_time = time[frame_idx[frame]]
        time_line.set_xdata(current_time)
        time_text.set_text(f'Time: {current_time:.1f} hr')
        return time_line, time_text
    
    ani = animation.FuncAnimation(
        fig, update, frames=range(len(frame_idx)), 
        init_func=init, blit=True, interval=interval
    )
    
    if save_path:
        ani.save(save_path, writer='pillow', fps=10)
    
    plt.tight_layout()
    return ani

def create_profile_and_heatmap_animation(z, time, h_profile, theta_profile, interval=100, save_path=None):
    """
    Create an animated visualization with line profiles and a 2D heatmap.
    
    Parameters:
    - z: depth coordinates [m]
    - time: time points [hours]
    - h_profile: pressure head profiles over time
    - theta_profile: water content profiles over time
    - interval: time between animation frames [ms]
    - save_path: if provided, save animation to this file
    
    Returns:
    - Animation object
    """
    # Use fewer frames for smoother animation
    ani_frames = 100
    frame_idx = np.linspace(0, len(time)-1, ani_frames, dtype=int)
    
    # Create figure with 3 subplots
    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(1, 3)
    
    # Pressure head profile subplot
    ax1 = fig.add_subplot(gs[0, 0])
    line1, = ax1.plot(h_profile[0, :], z)
    ax1.set_xlabel('Pressure Head [m]')
    ax1.set_ylabel('Depth [m]')
    ax1.set_title('Pressure Head Profile')
    h_min = np.min(h_profile)
    h_max = max(0, np.max(h_profile))
    ax1.set_xlim(h_min * 1.1, h_max * 1.1 if h_max > 0 else 0.1)
    ax1.invert_yaxis()
    ax1.grid(True)
    
    # Water content profile subplot
    ax2 = fig.add_subplot(gs[0, 1])
    line2, = ax2.plot(theta_profile[0, :], z)
    ax2.set_xlabel('Water Content [m³/m³]')
    ax2.set_ylabel('Depth [m]')
    ax2.set_title('Water Content Profile')
    ax2.set_xlim(np.min(theta_profile) * 0.9, np.max(theta_profile) * 1.1)
    ax2.invert_yaxis()
    ax2.grid(True)
    
    # 2D heatmap subplot
    ax3 = fig.add_subplot(gs[0, 2])
    
    # Create 2D grid for heatmap
    time_mesh, depth_mesh = np.meshgrid(time[frame_idx], z)
    theta_mesh = np.zeros((len(z), len(frame_idx)))
    
    for i, t_idx in enumerate(frame_idx):
        theta_mesh[:, i] = theta_profile[t_idx, :]
    
    # Initial heatmap
    heatmap = ax3.pcolormesh(
        time_mesh, 
        depth_mesh, 
        theta_mesh, 
        cmap='viridis',
        shading='auto'
    )
    
    # Vertical line for current time
    time_line = ax3.axvline(x=0, color='red', linestyle='-', linewidth=2)
    
    # Add colorbar
    cbar = fig.colorbar(heatmap, ax=ax3, label='Water Content [m³/m³]')
    
    # Configure heatmap axes
    ax3.set_xlabel('Time [hours]')
    ax3.set_ylabel('Depth [m]')
    ax3.set_title('Moisture Dispersion Over Time')
    ax3.invert_yaxis()
    
    time_text = fig.suptitle('Time: 0.0 hr', fontsize=16)
    
    def update(frame):
        # Update profile lines
        line1.set_xdata(h_profile[frame_idx[frame], :])
        line2.set_xdata(theta_profile[frame_idx[frame], :])
        
        # Update time line on heatmap
        time_line.set_xdata(time[frame_idx[frame]])
        
        # Update time text
        time_text.set_text(f'Time: {time[frame_idx[frame]]:.1f} hr')
        
        return line1, line2, time_line, time_text
    
    ani = animation.FuncAnimation(
        fig, update, frames=range(len(frame_idx)), 
        interval=interval, blit=True
    )
    
    if save_path:
        ani.save(save_path, writer='pillow', fps=10)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the title
    return ani

# Example usage with different soil types
def run_simulation(soil_type="sandy_loam", scenario="rainfall"):
    """
    Run a Richards equation simulation with predefined soil types and scenarios.
    
    Parameters:
    - soil_type: Soil type ('sand', 'loamy_sand', 'sandy_loam', 'loam', 'clay_loam', 'clay')
    - scenario: Water flow scenario ('rainfall', 'infiltration', 'drainage', 'evaporation')
    """
    # Soil hydraulic parameters based on typical values
    soil_params = {
        'sand': {
            'Ks': 0.25,        # Saturated hydraulic conductivity [m/hour]
            'theta_r': 0.045,  # Residual water content
            'theta_s': 0.43,   # Saturated water content
            'alpha': 14.5,     # van Genuchten parameter [1/m]
            'n': 2.68          # van Genuchten parameter
        },
        'loamy_sand': {
            'Ks': 0.15,
            'theta_r': 0.057,
            'theta_s': 0.41,
            'alpha': 12.4,
            'n': 2.28
        },
        'sandy_loam': {
            'Ks': 0.075,
            'theta_r': 0.065,
            'theta_s': 0.41,
            'alpha': 7.5,
            'n': 1.89
        },
        'loam': {
            'Ks': 0.025,
            'theta_r': 0.078,
            'theta_s': 0.43,
            'alpha': 3.6,
            'n': 1.56
        },
        'clay_loam': {
            'Ks': 0.008,
            'theta_r': 0.095,
            'theta_s': 0.41,
            'alpha': 1.9,
            'n': 1.31
        },
        'clay': {
            'Ks': 0.002,
            'theta_r': 0.068,
            'theta_s': 0.38,
            'alpha': 0.8,
            'n': 1.09
        }
    }
    
    # Scenario parameters
    scenario_params = {
        'rainfall': {
            'initial_condition': 'dry',
            'top_boundary': 'rainfall',
            'bottom_boundary': 'free_drainage',
            'total_time': 24.0
        },
        'infiltration': {
            'initial_condition': 'dry',
            'top_boundary': 'constant',
            'bottom_boundary': 'free_drainage',
            'total_time': 48.0
        },
        'drainage': {
            'initial_condition': 'wet',
            'top_boundary': 'evaporation',
            'bottom_boundary': 'free_drainage',
            'total_time': 72.0
        },
        'evaporation': {
            'initial_condition': 'wet',
            'top_boundary': 'evaporation',
            'bottom_boundary': 'zero_flux',
            'total_time': 120.0
        }
    }
    
    # Merge parameters
    params = {
        'depth': 1.0,
        'n_cells': 100,
        'dt': 0.1,
        **soil_params[soil_type],
        **scenario_params[scenario]
    }
    
    print(f"Running simulation for {soil_type} soil with {scenario} scenario...")
    
    # Run simulation
    z, time, h_profile, theta_profile = richards_equation_solver(**params)
    
    # Create visualizations
    print("Creating standard profile plots...")
    plot_times = [0, params['total_time']/4, params['total_time']/2, 
                 3*params['total_time']/4, params['total_time']]
    fig_profiles = plot_results(z, time, h_profile, theta_profile, plot_times)
    
    print("Creating animated heatmap...")
    fig_heatmap = create_heatmap_animation(z, time, theta_profile)
    
    print("Creating combined animated visualization...")
    fig_combined = create_profile_and_heatmap_animation(z, time, h_profile, theta_profile)
    
    plt.show()
    
    return z, time, h_profile, theta_profile

# Run the example if this file is executed directly
if __name__ == "__main__":
    # Choose soil type and scenario
    soil_type = "sandy_loam"  # Options: sand, loamy_sand, sandy_loam, loam, clay_loam, clay
    scenario = "rainfall"     # Options: rainfall, infiltration, drainage, evaporation
    
    z, time, h_profile, theta_profile = run_simulation(soil_type, scenario)
