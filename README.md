# OpenSim Fitter
A Python library for fitting OpenSim model geometry and kinematics to motion capture and video-based data sources.

## Create the Python environment

conda create -n opensim_fitter python=3.13
conda activate opensim_fitter
pip install -r dependencies/requirements.txt

## OpenSim installation

TODO: Python wheels

    python dependencies/wheels/install_wheel.py

If you would like to install OpenSim manually for development purposes, see section "Manual OpenSim Installation" below.

## Install OpenSim Fitter

To install the OpenSim Fitter Python package,

    pip install .

Or, if you're a developer, install in "editable" mode,

    pip install -e .

## Manual OpenSim Installation

### Dependencies

Install the following dependencies with your favorite package manager on your platform (Homebrew, apt-get, etc.):
- cmake
- autoconf
- automake
- pkg-config
- libtool
- openblas
- lapack
- freeglut
- doxygen
- pcre
- pcre2
- openssl
- gcc

Consult [the OpenSim build scripts](https://github.com/opensim-org/opensim-core/tree/main/scripts/build) for platform-specific package installation commands.

### Create the `config.yaml` file

Create a file named `config.yaml` in the root directory of the repository with the field `python_root_dir`, which is a full path to a Python installation directory, e.g.,

    python_root_dir: '/Users/nbianco/miniconda3/envs/opensim_dev'

Note: make sure that the Python version in `python_root_dir` matches the Python version in your installation environment.

### Build OpenSim

Run the following command from the root directory to build OpenSim and install it into your conda environment.

    python dependencies/install_opensim.py
