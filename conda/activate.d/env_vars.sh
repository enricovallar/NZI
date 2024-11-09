# FILE: conda/activate.d/env_vars.sh

#!/bin/sh
# Retrieve the Python major and minor version (e.g., 3.11)
PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")

# Set PYTHONPATH to include only the active environment's site-packages
export PYTHONPATH="${CONDA_PREFIX}/lib/python${PYTHON_VERSION}/site-packages:${PYTHONPATH}"

# Set additional environment variables for MPI
export OMPI_MCA_opal_cuda_support=true
export UCX_MEMTYPE_CACHE=n