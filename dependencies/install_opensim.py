import os
import pathlib
import shutil
import subprocess
import sys

import yaml


with open('config.yaml') as f:
    config = yaml.safe_load(f)

# Normalize to a forward-slash path so CMake accepts it verbatim on every
# platform (including Windows, where backslashes can be misread as escapes
# by downstream tooling).
python_root_dir = pathlib.Path(config['python_root_dir']).as_posix()
cwd = os.path.dirname(os.path.abspath(__file__))

# On Windows, dispatch to the PowerShell port. This avoids depending on
# bash being on PATH, since C:\Windows\System32\bash.exe is the WSL
# launcher and fails immediately on machines with no WSL distro
# installed (the GitHub-hosted windows-latest runner included).
if sys.platform == 'win32':
    pwsh = shutil.which('pwsh') or shutil.which('powershell.exe') or 'pwsh'
    cmd = [pwsh, '-NoProfile', '-ExecutionPolicy', 'Bypass',
           '-File', 'install_opensim.ps1', python_root_dir]
else:
    cmd = ['bash', 'install_opensim.sh', python_root_dir]

subprocess.run(cmd, check=True, cwd=cwd)

# Install the OpenSim Python package in the current environment.
package = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'opensim',
                       'opensim_core_install', 'sdk', 'Python', '.')
subprocess.check_call([sys.executable, "-m", "pip", "install", package])
