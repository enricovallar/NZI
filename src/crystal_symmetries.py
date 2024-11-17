import numpy as np

def rotation_matrix(theta):
    """Return a 2D rotation matrix for rotation by theta degrees."""
    rad = np.deg2rad(theta)
    return np.array([
        [ np.cos(rad), -np.sin(rad)],
        [ np.sin(rad),  np.cos(rad)]
    ])

def reflection_matrix(angle):
    """Return a 2D reflection matrix for reflection over a line at the given angle in degrees."""
    rad = np.deg2rad(2 * angle)
    return np.array([
        [ np.cos(rad),  np.sin(rad)],
        [ np.sin(rad), -np.cos(rad)]
    ])
# Symmetry operations for C4v point group (square lattice)
C4v_operations = {
    "E": np.identity(2),
    "C4": rotation_matrix(90),
    "C2": rotation_matrix(180),
    "C4^3": rotation_matrix(270),
    "sigma_y": reflection_matrix(0),        # Reflection over x-axis
    "sigma_x": reflection_matrix(90),       # Reflection over y-axis
    "sigma_d'": reflection_matrix(45),       # Reflection over y = x
    "sigma_d''": reflection_matrix(-45) # Reflection over y = -x
}

# Symmetry operations for C6v point group (triangular lattice)
C6v_operations = {
    "E": np.identity(2),              # Identity operation
    "C6": rotation_matrix(60),        # 60-degree rotation
    "C3": rotation_matrix(120),       # 120-degree rotation
    "C2": rotation_matrix(180),       # 180-degree rotation
    "C3^2": rotation_matrix(240),     # 240-degree rotation
    "C6^5": rotation_matrix(300),     # 300-degree rotation
    "sigma_y": reflection_matrix(0),   # Reflection over x-axis
    "sigma_x'": reflection_matrix(30), # Reflection over 30 degrees
    "sigma_y''": reflection_matrix(60),# Reflection over 60 degrees
    "sigma_x": reflection_matrix(90),  # Reflection over y-axis
    "sigma_y'": reflection_matrix(120),# Reflection over 120 degrees
    "sigma_x''": reflection_matrix(150) # Reflection over 150 degrees
}

# Irreducible representations of C4v point group
C4v_irrep = [{
    "A1": {"E": 1, "C4": 1, "C2": 1, "C4^3": 1, "sigma_x": 1, "sigma_y": 1, "sigma_d'": 1, "sigma_d''": 1},
    "A2": {"E": 1, "C4": 1, "C2": 1, "C4^3": 1, "sigma_x": -1, "sigma_y": -1, "sigma_d'": -1, "sigma_d''": -1},
    "B1": {"E": 1, "C4": -1, "C2": 1, "C4^3": -1, "sigma_x": 1, "sigma_y": 1, "sigma_d'": -1, "sigma_d''": -1},
    "B2": {"E": 1, "C4": -1, "C2": 1, "C4^3": -1, "sigma_x": -1, "sigma_y": -1, "sigma_d'": 1, "sigma_d''": 1},
    "E": {"E": 2, "C4": 0, "C2": -2, "C4^3": 0, "sigma_x": 0, "sigma_y": 0, "sigma_d'": 0, "sigma_d''": 0}
}]

# Irreducible representations of C6v point group
C6v_irrep = [{
    "A1": {"E": 1, "C6": 1, "C3": 1, "C2": 1, "C3^2": 1, "C6^5": 1, "sigma_y": 1, "sigma_x'": 1, "sigma_y''": 1, "sigma_x": 1, "sigma_y'": 1, "sigma_x''": 1},
    "A2": {"E": 1, "C6": 1, "C3": 1, "C2": 1, "C3^2": 1, "C6^5": 1, "sigma_y": -1, "sigma_x'": -1, "sigma_y''": -1, "sigma_x": -1, "sigma_y'": -1, "sigma_x''": -1},
    "B1": {"E": 1, "C6": -1, "C3": 1, "C2": -1, "C3^2": 1, "C6^5": -1, "sigma_y": 1, "sigma_x'": -1, "sigma_y''": 1, "sigma_x": -1, "sigma_y'": 1, "sigma_x''": -1},
    "B2": {"E": 1, "C6": -1, "C3": 1, "C2": -1, "C3^2": 1, "C6^5": -1, "sigma_y": -1, "sigma_x'": 1, "sigma_y''": -1, "sigma_x": 1, "sigma_y'": -1, "sigma_x''": 1},
    "E1": {"E": 2, "C6": 1, "C3": -1, "C2": -2, "C3^2": -1, "C6^5": 1, "sigma_y": 0, "sigma_x'": 0, "sigma_y''": 0, "sigma_x": 0, "sigma_y'": 0, "sigma_x''": 0},
    "E2": {"E": 2, "C6": -1, "C3": -1, "C2": 2, "C3^2": -1, "C6^5": -1, "sigma_y": 0, "sigma_x'": 0, "sigma_y''": 0, "sigma_x": 0, "sigma_y'": 0, "sigma_x''": 0}
}]

C2v_irrep =[ {
    "A1": {"E": 1, "C2": 1, "sigma_x": 1, "sigma_y": 1},
    "A2": {"E": 1, "C2": 1, "sigma_x": -1, "sigma_y": -1},
    "B1": {"E": 1, "C2": -1, "sigma_x": 1, "sigma_y": -1},
    "B2": {"E": 1, "C2": -1, "sigma_x": -1, "sigma_y": 1}
}]

C_1h_irrep =[ {
    "A": {"E": 1, "sigma_y":1},
    "B": {"E": -1, "sigma_y": -1}
    },
    {
    "A": {"E": 1, "sigma_x":1},
    "B": {"E": -1, "sigma_x": -1}
    },
    {
    "A": {"E": 1, "sigma_d'":1},
    "B": {"E": -1, "sigma_d'": -1}
    },
]

# Symmetry operations for C3v point group (equilateral triangular lattice)
C3v_operations = {
    "E": np.identity(2),              # Identity operation
    "C3": rotation_matrix(120),       # 120-degree rotation
    "C3^2": rotation_matrix(240),     # 240-degree rotation
    "sigma_v": reflection_matrix(0),   # Reflection over x-axis
    "sigma_v'": reflection_matrix(120),# Reflection over 120 degrees
    "sigma_v''": reflection_matrix(240)# Reflection over 240 degrees
}

# Irreducible representations of C3v point group
C3v_irrep = [{
    "A1": {"E": 1, "C3": 1, "C3^2": 1, "sigma_v": 1, "sigma_v'": 1, "sigma_v''": 1},
    "A2": {"E": 1, "C3": 1, "C3^2": 1, "sigma_v": -1, "sigma_v'": -1, "sigma_v''": -1},
    "E": {"E": 2, "C3": -1, "C3^2": -1, "sigma_v": 0, "sigma_v'": 0, "sigma_v''": 0}
},
{
    "A1": {"E": 1, "C3": 1, "C3^2": 1, "sigma_v": 1, "sigma_v'": 1, "sigma_v''": 1},
    "A2": {"E": 1, "C3": 1, "C3^2": 1, "sigma_v": -1, "sigma_v'": -1, "sigma_v''": -1},
    "E": {"E": 2, "C3": -1, "C3^2": -1, "sigma_v": 0, "sigma_v'": 0, "sigma_v''": 0}
}
]


def test_symmetry_operations():
    # Create a test point
    point = np.array([1.0, 0.5])
    
    import matplotlib.pyplot as plt
    
    # Test C4v operations
    plt.figure(figsize=(10, 10))
    plt.title('C4v Symmetry Operations')
    
    # Plot symmetry axes with different colors
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='σv')
    plt.axvline(x=0, color='blue', linestyle='--', alpha=0.5, label='σv\'')
    plt.plot([-2, 2], [-2, 2], 'green', linestyle='--', alpha=0.5, label='σd')
    plt.plot([-2, 2], [2, -2], 'orange', linestyle='--', alpha=0.5, label='σd\'')
    
    # Plot original point
    plt.plot(point[0], point[1], 'ko', markersize=12, label='Original')
    
    # Apply and plot each C4v operation
    for name, op in C4v_operations.items():
        transformed = op @ point
        plt.plot(transformed[0], transformed[1], 'o', markersize=12, label=name)
    
    plt.grid(True)
    plt.legend()
    plt.axis('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    
    # Test C6v operations
    plt.figure(figsize=(10, 10))
    plt.title('C6v Symmetry Operations')
    
    # Plot symmetry axes with different colors
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    for angle, color in zip([0, 30, 60, 90, 120, 150], colors):
        rad = np.deg2rad(angle)
        plt.plot([-2*np.cos(rad), 2*np.cos(rad)], 
                [-2*np.sin(rad), 2*np.sin(rad)], 
                color=color, linestyle='--', alpha=0.5, label=f'σ_{angle}°')
    
    # Plot original point
    plt.plot(point[0], point[1], 'ko', markersize=12, label='Original')
    
    # Apply and plot each C6v operation
    for name, op in C6v_operations.items():
        transformed = op @ point
        plt.plot(transformed[0], transformed[1], 'o', markersize=12, label=name)
    
    plt.grid(True)
    plt.legend()
    plt.axis('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.show()

if __name__ == '__main__':
    test_symmetry_operations()
