# OpenSim Fitter
A Python library for fitting OpenSim model geometry and kinematics to motion capture and video-based data sources.

## Create the Python environment

    conda create -n opensim_fitter python=3.13
    conda activate opensim_fitter
    pip install -r dependencies/requirements.txt

## OpenSim installation

For now, OpenSim Fitter requires installing a custom branch of OpenSim that disables the installation of the CasADi dependency that would conflict with the CasADi installed in the Python package we just created. Run the following command from the root directory to install a custom OpenSim Python wheel that has CasADi disabled (plus some other in-developement changes):

    python dependencies/install_opensim.py

Wheels are currently available for the following platforms and Python versions:

| Platform | Architecture | Python versions |
| --- | --- | --- |
| Windows | x86_64 | 3.11, 3.12, 3.13 |
| macOS | arm64 (Apple Silicon) | 3.11, 3.12, 3.13 |
| Linux (manylinux_2_28) | x86_64 | 3.11, 3.12, 3.13 |

If no wheel matches your interpreter and platform, build OpenSim from source instead (see below).

## Install OpenSim Fitter

To install the OpenSim Fitter Python package,

    pip install .

Or, if you're a developer, install in "editable" mode,

    pip install -e .

### Building OpenSim from source

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
