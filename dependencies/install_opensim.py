import os
import pathlib
import shutil
import subprocess
import sys

import yaml
with open('config.yaml') as f:
    config = yaml.safe_load(f)

python_root_dir = pathlib.Path(config['python_root_dir']).as_posix()
cwd = os.path.dirname(os.path.abspath(__file__))

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
