import numpy as np
import pyvista as pv

# --------------------------
# Parámetros del cilindro
# --------------------------
L = 0.30       # longitud [m]
R = 0.15       # radio [m]
Nx, Ny, Nz = 80, 80, 80  # resolución
x = np.linspace(0, L, Nx)
y = np.linspace(-R, R, Ny)
z = np.linspace(-R, R, Nz)
dx, dy, dz = x[1]-x[0], y[1]-y[0], z[1]-z[0]

X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# --------------------------
# Máscara del cilindro
# --------------------------
mask = Y**2 + Z**2 <= R**2

# --------------------------
# Distribución de temperatura/SAR
# --------------------------
T0 = 25.0
T = T0*np.ones_like(X)
T += mask * 30*np.exp(-((Y**2 + Z**2)/(R**2*0.2) + ((X-L/2)**2)/(L**2*0.1)))

# --------------------------
# Crear StructuredGrid
# --------------------------
grid = pv.StructuredGrid(X, Y, Z)
# Para voxels fuera del cilindro ponemos NaN para que sean transparentes
T_masked = np.where(mask, T, np.nan)
grid["SAR"] = T_masked.flatten(order="F")

# --------------------------
# Visualización
# --------------------------
plotter = pv.Plotter()
# Volumen transparente y suave
plotter.add_volume(grid, scalars="SAR", opacity="linear", cmap="hot", shade=True)

# --------------------------
# Sondas embebidas
# --------------------------
angles_deg = [0, 45, 80, 135, 180]
r_s = 0.8*R
for angle in angles_deg:
    theta = np.deg2rad(angle)
    y_s = r_s * np.cos(theta) * np.ones_like(x)
    z_s = r_s * np.sin(theta) * np.ones_like(x)
    plotter.add_mesh(pv.Line([0, y_s[0], z_s[0]], [L, y_s[-1], z_s[-1]]),
                     color="blue", line_width=3)

# --------------------------
# Opciones de visualización
# --------------------------
plotter.add_axes()
plotter.show_grid()
plotter.show()
