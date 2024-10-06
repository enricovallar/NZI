import numpy as np

# Function to apply a symmetry operation to the field and check if it's symmetric
def check_symmetry(field_data, operation, tolerance=1e-6):
    """
    Apply the symmetry operation (e.g., rotation, reflection) to the field
    and compare it to the original field data.
    
    Args:
        field_data (ndarray): The original field data (2D array).
        operation (function): A function that applies the symmetry operation to the field.
        tolerance (float): Numerical tolerance for comparing fields.
        
    Returns:
        bool: True if the field remains unchanged under the symmetry operation.
    """
    transformed_field = operation(field_data)
    return np.allclose(field_data, transformed_field, atol=tolerance)

# Function to apply a parity operation to the field and check if it's symmetric
def check_parity(field_data, operation, tolerance=1e-6):
    """
    Apply the parity operation (e.g., inversion) to the field
    and compare it to the original field data.
    
    Args:
        field_data (ndarray): The original field data (2D array).
        operation (function): A function that applies the parity operation to the field.
        tolerance (float): Numerical tolerance for comparing fields.
        
    Returns:
        bool: True if the field remains unchanged under the parity operation.
    """
    transformed_field = operation(field_data)
    return np.allclose(field_data, transformed_field, atol=tolerance)

# Example symmetry operations for D4 group
def rotate_90(field_data):
    return np.rot90(field_data, k=1)

def rotate_180(field_data):
    return np.rot90(field_data, k=2)

def rotate_270(field_data):
    return np.rot90(field_data, k=3)

def reflect_vertical(field_data):
    return np.flipud(field_data)

def reflect_horizontal(field_data):
    return np.fliplr(field_data)

def reflect_diagonal(field_data):
    return np.transpose(field_data)

# Example parity operation
def inversion(field_data):
    return np.flipud(np.fliplr(field_data))

# Test if the field has D4 symmetry and create a list of satisfied symmetries
def test_symmetries(field_data):
    """
    Test if a given field has D4 symmetries by applying the symmetry operations
    of the D4 group (rotations and reflections).
    
    Args:
        field_data (ndarray): The field data to be tested.
    
    Returns:
        list: A list of symmetry operations that the field satisfies.
    """
    # Initialize symmetry operations and their names
    symmetry_operations = {
        'Rotation by 90 degrees': rotate_90,
        'Rotation by 180 degrees': rotate_180,
        'Rotation by 270 degrees': rotate_270,
        'Reflection about vertical axis': reflect_vertical,
        'Reflection about horizontal axis': reflect_horizontal,
        'Reflection about diagonal axis': reflect_diagonal
    }

    # Check each symmetry operation
    satisfied_symmetries = []
    for operation_name, operation in symmetry_operations.items():
        if check_symmetry(field_data, operation):
            satisfied_symmetries.append(operation_name)
    
    return satisfied_symmetries

# Assign a symmetry group based on the symmetries satisfied by the field
def assign_symmetry_group(symmetries):
    """
    Assign the symmetry group based on the satisfied symmetries.
    
    Args:
        symmetries (list): List of symmetry operations that the field satisfies.
    
    Returns:
        str: The name of the symmetry group assigned to the field.
    """
    # Simple logic for assigning groups based on symmetries
    if 'Rotation by 90 degrees' in symmetries and 'Rotation by 180 degrees' in symmetries and 'Rotation by 270 degrees' in symmetries:
        if 'Reflection about vertical axis' in symmetries and 'Reflection about horizontal axis' in symmetries and 'Reflection about diagonal axis' in symmetries:
            return "D4 (Full Dihedral Symmetry)"
        else:
            return "C4 (Cyclic Symmetry with 90-degree rotation)"
    elif 'Rotation by 180 degrees' in symmetries:
        if 'Reflection about vertical axis' in symmetries or 'Reflection about horizontal axis' in symmetries:
            return "D2 (Dihedral Symmetry with 180-degree rotation)"
        else:
            return "C2 (Cyclic Symmetry with 180-degree rotation)"
    elif 'Reflection about vertical axis' in symmetries and 'Reflection about horizontal axis' in symmetries:
        return "Mirror Symmetry (No rotational symmetry)"
    else:
        return "No significant symmetry group"

