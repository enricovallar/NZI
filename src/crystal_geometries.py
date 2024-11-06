import math
import meep as mp
from meep import mpb
from crystal_materials import Crystal_Materials
from functools import partial

class Crystal_Geometry:

    GEOMETRY_TYPES = ['square', 'circular', 'rectangular', 'elliptical']


    def __init__(self, 
                 material : Crystal_Materials,
                 geometry_type: str = 'circular'
                 ):
        self.required_arguments = {
            "material": material,
            "geometry_type": geometry_type,
        }
        self.kwargs = {}
        self.arguments = {**self.required_arguments, **self.kwargs}
        self.material = material
        self.geometry =[]
        self.base_geometry = self.geometry
        self.geometry_type = geometry_type
        
        self.atomic_function = None


    
    def square_atom(self, a, r):
        raise NotImplementedError("This method is not yet implemented")
        

    def circular_atom(self, r):
        raise NotImplementedError("This method is not yet implemented")
        
    def rectangular_atom(self, a, b):
        raise NotImplementedError("This method is not yet implemented")
    
    def elliptical_atom(self, a, b):
        raise NotImplementedError("This method is not yet implemented")
    
    def build_geometry(self):
        raise NotImplementedError("This method is not yet implemented")
    
    def geometry_function(self, atomic_function,  **kwargs):
        geometry = self.base_geometry
        geometry.append(atomic_function(**kwargs))

    def to_list(self):
        return self.geometry
    
    def to_partial(self, exclude_key): 
        arguments = self.arguments.copy()
        arguments.pop(exclude_key)
        return partial(self.__class__, **self.arguments)
        

        


    

class Crystal2D_Geometry(Crystal_Geometry):

    def __init__(self,
                 material: Crystal_Materials,
                 geometry_type: str ='circular',
                 **kwargs
                 ):
        """
        Initializes a new CrystalGeometry instance.

        Args:
            material (CrystalMaterials): The material of the crystal.
            geometry_type (str, optional): The type of geometry to initialize.
                Defaults to 'circular'. Supported types include 'circular', 'square',
                'rectangular', and 'elliptical'.
            **kwargs: Additional keyword arguments specific to the geometry type.
                If `geometry_type` is 'circular', kwargs may include:
                    r (float): Radius of the circular geometry.
                    center (mp.Vector3): Center of the circular geometry.
                If `geometry_type` is 'square', kwargs may include:
                    a (float): Length of each side of the square.
                    center (mp.Vector3): Center of the square. 
                If `geometry_type` is 'rectangular', kwargs may include:
                    a (float): Width of the rectangle.
                    b (float): Height of the rectangle.
                    center (mp.Vector3): Center of the rectangle.   
                If `geometry_type` is 'elliptical', kwargs may include:
                    a: Length of the major axis.
                    b: Length of the minor axis.
                    center (mp.Vector3): Center of the ellipse. 
        Raises:
            ValueError: If an invalid `geometry_type` is provided.
        """

        super().__init__(material, geometry_type)
        self.required_arguments = {
            "material": material,
            "geometry_type": geometry_type,
        }
        self.kwargs = kwargs
        self.arguments = {**self.required_arguments, **self.kwargs}
        
        self.geometry = []
        #bulk
        self.geometry.append(
            mp.Block(
                size = mp.Vector3(mp.inf, mp.inf, 0),
                material=self.material.bulk),
        )
        self.base_geometry = self.geometry
        
        if self.geometry_type == 'circular':
            self.atomic_function = self.circular_atom  
        elif self.geometry_type == 'square':
            self.atomic_function = self.square_atom
        elif self.geometry_type == 'rectangular':
            self.atomic_function = self.rectangular_atom
        elif self.geometry_type == 'elliptical':
            self.atomic_function = self.elliptical_atom
        else:
            raise ValueError(f"Invalid geometry type: {self.geometry_type}")
        self.atomic_function(**kwargs)
                 

    def circular_atom(self, r: float = 0.2, center: mp.Vector3 = mp.Vector3(0,0)):
        """
        Adds a circular atom to the geometry.

        Args:
            r (float, optional): Radius of the circular atom. Defaults to 0.2.
            center (mp.Vector3, optional): Center of the circular atom. Defaults to mp.Vector3(0,0).

        Returns:
            None
        """
        self.geometry.append(mp.Cylinder(
                                        material=self.material.atom, 
                                        center = center,
                                        radius=r,
                                        ))
        
    def square_atom(self, a: float = 0.5, center: mp.Vector3 = mp.Vector3(0,0)):
        """
        Adds a square atom to the geometry.

        Args:
            a (float, optional): Length of each side of the square atom. Defaults to 0.5.
            center (mp.Vector3, optional): Center of the square atom. Defaults to mp.Vector3(0,0).

        Returns:
            None
        """

        self.geometry.append(mp.Block(size = mp.Vector3(a, a, 0),
                                        material=self.material.atom, 
                                        center = center))
    
    def rectangular_atom(self, a: float = 0.2, b : float = 0.5, center: mp.Vector3 = mp.Vector3(0,0)):
        """
        Adds a rectangular atom to the geometry.

        Args:
            a (float, optional): Width of the rectangular atom. Defaults to 0.2.
            b (float, optional): Height of the rectangular atom. Defaults to 0.5.
            center (mp.Vector3, optional): Center of the rectangular atom. Defaults to mp.Vector3(0,0).

        Returns:
            None
        """
        self.geometry.append(mp.Block(
                            size=mp.Vector3(a, b, 0),
                            material=self.material.atom, 
                            center = center))
    
    def elliptical_atom(self, a : float = 0.2, b : float = 0.5, center: mp.Vector3 = mp.Vector3(0,0)):
        """
        Adds an elliptical atom to the geometry.    

        Args:
            a (float, optional): Length of the major axis. Defaults to 0.2.
            b (float, optional): Length of the minor axis. Defaults to 0.5.
            center (mp.Vector3, optional): Center of the elliptical atom. Defaults to mp.Vector3(0,0).

        Returns:
            None
        """
        self.geometry.append(mp.Ellipsoid(
                                      size = mp.Vector3(a, b, mp.inf),
                                      material=self.material.atom, 
                                      center = center))

        


    

class CrystalSlab_Geometry(Crystal_Geometry):
    """
    Represents a 3D crystal geometry with a slab structure.
    Inherits from Crystal_Geometry and adds additional layers for substrate and background.
    """

    def __init__(self,
                    material: Crystal_Materials,
                    geometry_type: str = 'circular',
                    height_slab: float = 0.5,
                    height_supercell: float = 4.0,
                    **kwargs):
        """
        Initializes a new CrystalSlab_Geometry instance.

        Args:
            material (Crystal_Materials): The material of the crystal.
            geometry_type (str, optional): Type of geometry ('circular', 'square', 'rectangular', 'elliptical').
                Defaults to 'circular'.
            height_slab (float, optional): Height of the slab. Defaults to 0.5.
            height_supercell (float, optional): Height of the supercell. Defaults to 4.0.
            **kwargs: Additional arguments specific to the geometry type.
                If `geometry_type` is 'circular', kwargs may include:
                    r (float): Radius of the circular atom.
                    center (mp.Vector3): Center of the circular atom.   
                If `geometry_type` is 'square', kwargs may include:
                    a (float): Side length of the square atom.
                    center (mp.Vector3): Center of the square atom.
                If `geometry_type` is 'rectangular', kwargs may include:
                    a (float): Width of the rectangular atom.
                    b (float): Height of the rectangular atom.
                    center (mp.Vector3): Center of the rectangular atom.
                If `geometry_type` is 'elliptical', kwargs may include:
                    a (float): Length of the major axis.
                    b (float): Length of the minor axis.
                    center (mp.Vector3): Center of the elliptical atom.
        Raises:
            ValueError: If an invalid `geometry_type` is provided.


        """
        super().__init__(material, geometry_type)
        self.required_arguments = {
            "material": material,
            "geometry_type": geometry_type,
            "height_slab": height_slab,
            "height_supercell": height_supercell,
        }
        self.kwargs = kwargs
        self.arguments = {**self.required_arguments, **self.kwargs}

        self.height_slab = height_slab
        self.height_supercell = height_supercell
        

        # Add background material extending infinitely in x and y, with specified height in z
        self.geometry.append(
            mp.Block(
                size=mp.Vector3(mp.inf, mp.inf, self.height_supercell),
                material=self.material.background),
        )

        # Add substrate material below the origin
        self.geometry.append(mp.Block(
            size=mp.Vector3(mp.inf, mp.inf, self.height_supercell * 0.5),
            center=mp.Vector3(0, 0, -self.height_supercell * 0.25),
            material=self.material.substrate
        ))

        # Add slab material at the origin
        self.geometry.append(
            mp.Block(
                size=mp.Vector3(mp.inf, mp.inf, self.height_slab),
                material=self.material.bulk),
        )
        self.base_geometry = self.geometry

        # Initialize the specific atom geometry based on the geometry type
        if self.geometry_type == 'circular':
            self.atomic_function = self.circular_atom
        elif self.geometry_type == 'square':
            self.atomic_function = self.square_atom
        elif self.geometry_type == 'rectangular':
            self.atomic_function = self.rectangular_atom
        elif self.geometry_type == 'elliptical':
            self.atomic_function = self.elliptical_atom
        else:
            raise ValueError(f"Invalid geometry type: {self.geometry_type}")
        self.atomic_function(**kwargs)

    def circular_atom(self, r: float = 0.2):
        """
        Adds a cylindrical atom to the slab geometry.

        Args:
            r (float, optional): Radius of the cylindrical atom. Defaults to 0.2.
            center (mp.Vector3, optional): Center of the cylindrical atom. Defaults to mp.Vector3(0,0).
        """
        self.geometry.append(mp.Cylinder(
            radius=r,
            material=self.material.atom,
            height=self.height_slab
        ))

    def square_atom(self, a: float = 0.5, center: mp.Vector3 = mp.Vector3(0, 0)):
        """
        Adds a square block atom to the slab geometry.

        Args:
            a (float, optional): Side length of the square atom. Defaults to 0.5.
            center (mp.Vector3, optional): Center of the square atom. Defaults to mp.Vector3(0,0).
        """
        self.geometry.append(mp.Block(
            size=mp.Vector3(a, a, self.height_slab),
            material=self.material.atom, 
            center=center

        ))

    def rectangular_atom(self, a: float = 0.2, b: float = 0.5, center: mp.Vector3 = mp.Vector3(0, 0)):
        """
        Adds a rectangular block atom to the slab geometry.

        Args:
            a (float, optional): Width of the rectangular atom. Defaults to 0.2.
            b (float, optional): Height of the rectangular atom. Defaults to 0.5.
            center (mp.Vector3, optional): Center of the rectangular atom. Defaults to mp.Vector3(0,0).
        """
        self.geometry.append(mp.Block(
            size=mp.Vector3(a, b, self.height_slab),
            material=self.material.atom, 
            center = center
        ))

    def elliptical_atom(self, a: float = 0.2, b: float = 0.5, center: mp.Vector3 = mp.Vector3(0, 0)):
        """
        Adds an elliptical ellipsoid atom to the slab geometry.

        Args:
            a (float, optional): Length of the major axis. Defaults to 0.2.
            b (float, optional): Length of the minor axis. Defaults to 0.5.
            center (mp.Vector3, optional): Center of the elliptical atom. Defaults to mp.Vector3(0,0).
        """
        
    
        background =  mp.Block(
            size=mp.Vector3(mp.inf, mp.inf, self.height_supercell/2 - self.height_slab/2),
            material=self.material.background,
            center = mp.Vector3(0,0, self.height_slab/2 + (self.height_supercell/2 - self.height_slab/2)/2)
        )

        substrate = mp.Block(
            size=mp.Vector3(mp.inf, mp.inf, self.height_supercell/2-self.height_slab/2),
            material=self.material.substrate,
            center = mp.Vector3(0, 0, -self.height_slab/2 -(self.height_supercell/2 - self.height_slab/2)/2)
        )
        slab = self.geometry[2]
        atom = (mp.Ellipsoid(
            size=mp.Vector3(a, b, mp.inf),
            material=self.material.atom,
            center=center
        ))

        self.geometry = [slab, atom, background, substrate]
        

        
        



if __name__ == '__main__':
    # %%
    from crystal_materials import Crystal_Materials
    from crystal_geometries import Crystal2D_Geometry, CrystalSlab_Geometry
    from photonic_crystal import PhotonicCrystal, Crystal2D, CrystalSlab

    # Initialize materials
    material = Crystal_Materials()
    material.background = {'epsilon': 1}
    material.substrate = {'epsilon': 1}
    material.bulk = {'epsilon': 12}
    material.atom = {'epsilon': 5}

    # Simulation parameters
    num_bands = 8
    resolution = 32
    interp = 3
    periods = 1
    lattice_type = 'square'
    pickle_id = 'square'

    # Create geometry
    geometry = Crystal2D_Geometry(
        material,
        geometry_type='elliptical',
        a=0.5, 
        b=0.2,

    )
    
    # Initialize and run photonic crystal simulation
    crystal_2d = Crystal2D(
        lattice_type=lattice_type,
        num_bands=num_bands,
        resolution=resolution,
        interp=interp,
        periods=periods,
        pickle_id=pickle_id,
        geometry=geometry,
        k_point_max=0.5,
        use_XY=True
    )
    crystal_2d.run_dumb_simulation()

    # Plot and display epsilon distribution
    fig = crystal_2d.plot_epsilon()
    fig.show()

    print(geometry.kwargs)
    print(vars(geometry))

    partial_geometry = geometry.to_partial(exclude_key='a')
    print(partial_geometry().kwargs)


# %%
    