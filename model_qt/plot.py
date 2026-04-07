import numpy as np
import matplotlib.pyplot as plt

# Pressure levels in hPa (from surface to upper atmosphere)
pressure_levels = ['1000', '925', '850', '700', '600', '500', '400',
                   '300', '250', '200', '150', '100', '50']
p_vals = np.array([1000, 925, 850, 700, 600, 500, 400,
                   300, 250, 200, 150, 100, 50], dtype=float)

# Load RMSE arrays (shape: 4 variables x 13 levels)
rmse_baseline = np.load("/share/home/lililei4/lzx/Unet/code/model_qt/results/rmse_baseline.npy")  # (2, 13)
rmse_model    = np.load("/share/home/lililei4/lzx/Unet/code/model_qt/results/rmse_model.npy")     # (2, 13)

# Variable names for titles
var_names = ['q (specific humidity)', 't (temperature)']

# Plot settings
plt.style.use('default')
fig_size = (6, 8)

# Generate one figure per variable
for i, var in enumerate(var_names):
    plt.figure(figsize=fig_size)
    
    baseline_vals = rmse_baseline[i, :]  # RMSE for baseline
    model_vals    = rmse_model[i, :]     # RMSE for model
    
    plt.plot(baseline_vals, p_vals, 'o-', label='Baseline', linewidth=2, markersize=6)
    plt.plot(model_vals,    p_vals, 's-', label='Model',    linewidth=2, markersize=6)
    
    # Invert y-axis so that higher pressure (1000 hPa) is at bottom
    plt.gca().invert_yaxis()
    plt.yticks(p_vals, pressure_levels)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('RMSE', fontsize=12)
    plt.ylabel('Pressure Level (hPa)', fontsize=12)
    plt.title(var, fontsize=14)
    plt.legend(fontsize=11)
    
    plt.tight_layout()
    
    # Save figure with variable name as filename
    filename = f'rmse_{var.split()[0]}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"Saved: {filename}")

print("All figures saved.")