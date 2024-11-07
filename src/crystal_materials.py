import math 
import meep as mp 
from meep import mpb

class Crystal_Materials(): 
        """Crystal_Materials class for defining crystal material properties.

        This class manages the material properties of different components in a crystal structure,
        including background, bulk, atom, and substrate materials. Each material is defined by its
        dielectric properties and nonlinear susceptibility tensors.
        
        Usage:

            Use a dictionary to set the material properties for each component. The dictionary should
            contain the valid configuration keys for material properties. The material properties can
            be set using the background, bulk, atom, and substrate properties.
        
        Example:
            ```python
            materials = Crystal_Materials()
            materials.background = {
                'epsilon': 1.0,
            }
            materials.bulk = {
                'epsilon_diag': mp.Vector3(3.5, 4.0, 4.0),
                'epsilon_offdiag': mp.Vector3(0.0, 0.0, 0.0),
                'E_chi2_diag': mp.Vector3(0.0, 0.0, 0.0),
            }

            materials.atom = {
                'epsilon' : 1.0,
            }

            materials.substrate = {
                'epsilon' : 1
            }
            ```



        
            
        Valid configuration parameters for materials:

            epsilon (float): Dielectric constant of the material.
            epsilon_diag (mp.Vector3): Diagonal elements of the dielectric tensor.
            epsilon_offdiag (mp.Vector3): Off-diagonal elements of the dielectric tensor.
            E_chi2_diag (mp.Vector3): Diagonal elements of the second-order nonlinear susceptibility tensor.
            E_chi3_diag (mp.Vector3): Diagonal elements of the third-order nonlinear susceptibility tensor.

        Attributes:
            VALID_KEYS (list): List of valid configuration keys for material properties.
            _background (mp.Medium): Background material properties.
            _bulk (mp.Medium): Bulk material properties.
            _atom (mp.Medium): Atom material properties.
            _substrate (mp.Medium): Substrate material properties.
            background: Gets/sets the background material properties.
            bulk: Gets/sets the bulk material properties.
            atom: Gets/sets the atom material properties.
            substrate: Gets/sets the substrate material properties.

        

        Raises:
            ValueError: If invalid configuration keys are provided when setting material properties.
        """
        VALID_KEYS = ['epsilon', 'epsilon_diag', 'epsilon_offdiag', 'E_chi2_diag', 'E_chi3_diag']
    
        def __init__(self):
            """
            Initializes the Crystal_Materials class.

            """
            self._background = None
            self._bulk = None
            self._atom = None
            self._substrate = None
        
   
        
        @property
        def background(self):
            return self._background
        
        @property
        def bulk(self):
            return self._bulk
        
        @property   
        def atom(self):
            return self._atom
        
        @property
        def substrate(self):
            return self._substrate
        
        @background.setter
        def background(self, configuration: dict):
            if not all(key in self.VALID_KEYS for key in configuration.keys()):
                raise ValueError("Invalid configuration keys")
            self._background = mp.Medium(**configuration)
        
        @bulk.setter
        def bulk(self, configuration: dict): 
            if not all(key in self.VALID_KEYS for key in configuration.keys()):
                raise ValueError("Invalid configuration keys")
            self._bulk = mp.Medium(**configuration)
            
        
        @atom.setter
        def atom(self, configuration: dict): 
            if not all(key in self.VALID_KEYS for key in configuration.keys()):
                raise ValueError("Invalid configuration keys")
            self._atom = mp.Medium(**configuration)

        @substrate.setter  
        def substrate(self, configuration: dict):
            if not all(key in self.VALID_KEYS for key in configuration.keys()):
                raise ValueError("Invalid configuration keys")
            self._substrate = mp.Medium(**configuration)
        
            

