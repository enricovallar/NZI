import meep as mp
from meep import mpb
import numpy as np
import matplotlib.pyplot as plt
from photonic_crystal import suppress_output

# Define the lattice: square lattice with lattice constant a=1
lattice = mp.Lattice(
    size=mp.Vector3(1, 1),
    basis1=mp.Vector3(0.5, np.sqrt(3)/2),
    basis2=mp.Vector3(0.5, -np.sqrt(3)/2)
    )

# Define materials
material_A = mp.Medium(epsilon=12.0)  # High dielectric constant
material_B = mp.Medium(epsilon=2.0)   # Low dielectric constant

# Define geometry: two cylinders (rods) of different materials in the unit cell
geometry = [
    mp.Cylinder(radius=0.2, material=material_A, center=mp.Vector3(-1 / 6, 1 / 6)),
    mp.Cylinder(radius=0.2, material=material_B, center=mp.Vector3(1 / 6, -1 / 6))
]


# Define k-points: path in the Brillouin zone
k_points = mp.interpolate(10, [
    mp.Vector3(),           # Gamma
    mp.Vector3(0.5),        # X
    mp.Vector3(0.5, 0.5),   # M
    mp.Vector3()            # Gamma
])

# Set up the mode solver
ms = mpb.ModeSolver(
    geometry_lattice=lattice,
    geometry=geometry,
    resolution=32,
    num_bands=8,
    k_points=k_points
)

with suppress_output():
    # Run the simulation for TE modes
    ms.run_te()

    # Run the simulation for TM modes
    ms.run_tm()

# Retrieve dielectric function (epsilon)
epsilon = ms.get_epsilon()
md = mpb.MPBData(rectify=True, resolution=32, periods=4)
epsilon = md.convert(epsilon)

# Plot dielectric function as heatmap
plt.figure()
plt.imshow(epsilon.T, interpolation='spline36', cmap='binary')
plt.title('Dielectric Function (ε)')
plt.axis('off')
plt.colorbar(label='ε')
plt.show()

# Retrieve and plot the band structure
def plot_band_structure(ms, title):
    bands = ms.all_freqs  # Normalized frequencies
    num_k_points = len(ms.k_points)
    x = np.arange(num_k_points)
    plt.figure()
    for i in range(len(bands[0])):
        plt.plot(x, bands[:, i], 'b-' if 'TE' in title else 'r--')
    plt.xlabel("k-point index")
    plt.ylabel("Frequency (c/a)")
    plt.title(title)
    plt.grid(True)
    plt.show()

# Plot TE band structure
plot_band_structure(ms, 'TE Band Structure')

# Plot TM band structure
plot_band_structure(ms, 'TM Band Structure')
