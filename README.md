# Description
Code to solve for forced MAC waves in spherical geometry.

This repo demonstrates how to reproduce figures in Nicolas & Buffett (2023), Excitation of high-latitude MAC waves in Earth's core, Geophysical Journal International, Volume 233, Issue 3, 1961–1973.

`preprocessing.ipynb` illustrates how to preprocess the output from the geodynamo model (i.e. converting from spherical harmonics & toroidal-poloidal representation to a spherical domain with r,theta,phi components). It uses functions from `preprocessingTools.py`.

The code to solve for MAC wave eigenmodes is in `macmodes.py`. `eigenmodes.ipynb` shows how to reproduce Figure 3 from the paper.

Finally, the code to solve for forced MAC waves is in `forced_waves.py` and `macmodes.py`, with additional utils in `time_tools.py` and `fourier_tools.py`. `forcedWaves.ipynb` demonstrates the workflow and shows how to reproduce figure 3.

Current release: [![DOI](https://zenodo.org/badge/296985370.svg)](https://zenodo.org/badge/latestdoi/296985370)

# Getting the data
Data is too large to be available on Github, but is archived on Zenodo at [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7592727.svg)](https://doi.org/10.5281/zenodo.7592727)

# Running the code
A .yml file is included that contains all necessary python packages to run the code and produce the figures. Create a conda environment using `conda env create -f environment.yml`, then activate with `conda activate macwaves`, launch a Jupyter notebook and you are hopefully all set!

# Contact
For any questions, contact qnicolas --at-- berkeley --dot-- edu
