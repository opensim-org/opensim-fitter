import os
import sys
import subprocess
import yaml
with open('config.yaml') as f:
    config = yaml.safe_load(f)

# Build OpenSim
python_root_dir = config['python_root_dir']
cwd = os.path.dirname(os.path.abspath(__file__))
subprocess.run(['bash', 'install_opensim.sh', python_root_dir], check=True, cwd=cwd)

# Install the OpenSim Python package in the current environment.
package = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'opensim',
                       'opensim_core_install', 'sdk', 'Python', '.')
subprocess.check_call([sys.executable, "-m", "pip", "install", package])
