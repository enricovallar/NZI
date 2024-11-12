```bash
# First build the package 
conda build conda-recipe -c conda-forge
conda build purge


# Then anaconda login 
anaconda login

# Then publish: 
ls ~/miniconda3/conda-bld/linux-64/

anaconda upload ~/miniconda3/conda-bld/linux-64/nzi-phc-finder-0.1.0-py311_0.tar.bz2
```
