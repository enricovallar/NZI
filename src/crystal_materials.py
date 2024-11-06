import math 
import meep as mp 
from meep import mpb

class Crystal_Materials(): 
        
        VALID_KEYS = ['epsilon', 'epsilon_diag', 'epsilon_offdiag', 'E_chi2_diag', 'E_chi3_diag']
    
        def __init__(self):
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
        
            

