
#%%
import math
import meep as mp
from meep import mpb

from photonic_crystal import PhotonicCrystal


h = 0.5  # the thickness of the slab
eps = 12.0  # the dielectric constant of the slab
loweps = 1.0  # the dielectric constant of the substrate
r = 0.3  # the radius of the holes
supercell_h = 4  # height of the supercell

geometry = [
    mp.Block(
        material=mp.Medium(epsilon=loweps),
        center=mp.Vector3(z=0.25 * supercell_h),
        size=mp.Vector3(mp.inf, mp.inf, 0.5 * supercell_h),
    ),
    mp.Block(material=mp.Medium(epsilon=eps), size=mp.Vector3(mp.inf, mp.inf, h)),
    mp.Cylinder(r, material=mp.air, height=supercell_h),
]



# Create a PhotonicCrystal object with slab geometry
pc = PhotonicCrystal(lattice_type='triangular', 
                     num_bands=8, geometry=geometry,
                     resolution=mp.Vector3(32,32,64))


M = mp.Vector3(y=0.5)

# Run the simulation
ms = mpb.ModeSolver(
            geometry=pc.geometry,
            geometry_lattice=pc.geometry_lattice,
            k_points=M,
            resolution=pc.resolution,
            num_bands=pc.num_bands
        )
ms.run_tm(mpb.output_at_kpoint(M, mpb.output_hfield_z,))


# %%
