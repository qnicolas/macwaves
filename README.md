# Description
Code to solve for forced MAC waves in spherical geometry - Nicolas & Buffett (2023), GJI (accepted).

This repo contains code to solve for forced MAC waves in spherical geometry - as in Nicolas, Q., & Buffett, B. A. (2023). Excitation of high-latitude MAC waves in Earth's core, GJI (accepted).

Preprocessing of output from the geodynamo model (i.e. converting from spherical harmonics & toroidal-poloidal representation to a spherical domain with r,theta,phi components) is illustrated in preprocessing.ipynb, using utilities in preprocessingTools.py.

Code to solve for MAC wave eigenmodes is in macmodes.py, with code to reproduce Figure 3 from the paper in eigenmodes.ipynb.

Finally, code to solve for forced MAC waves is mostly in forced_waves.py and macmodes.py with other components in time_tools.py and fourier_tools.py. A notebook demonstrating the workflow is forcedWaves.ipynb

Current release: [![DOI](https://zenodo.org/badge/296985370.svg)](https://zenodo.org/badge/latestdoi/296985370)

# Getting the data
Data is too large to be available on Github, but is archived on Zenodo at [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7592727.svg)](https://doi.org/10.5281/zenodo.7592727)

# Running the code
A .yml file is included that contains all necessary python packages to run the code and produce the figures. Create a conda environment using conda env create -f environment.yml, then activate with conda activate orogconv, launch a Jupyter notebook and you are hopefully all set!

# Contact
For any questions, contact qnicolas --at-- berkeley --dot-- edu
