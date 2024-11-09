# FILE: conda/deactivate.d/unset_env_vars.sh

#!/bin/sh
# Unset PYTHONPATH
unset PYTHONPATH

# Unset MPI-related environment variables
unset OMPI_MCA_opal_cuda_support
unset UCX_MEMTYPE_CACHE