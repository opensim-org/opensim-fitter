"""Install the opensim wheel for the current platform."""

import platform
import subprocess
import sys
from pathlib import Path

WHEELS_DIR = Path(__file__).parent

WHEEL_MAP = {
    ("linux",  "x86_64"): "opensim-4.6-cp313-cp313-linux_x86_64.whl",
    ("darwin", "x86_64"): "opensim-4.6-cp313-cp313-macosx_11_0_universal2.whl",
    ("darwin", "arm64"):  "opensim-4.6-cp313-cp313-macosx_11_0_universal2.whl",
    ("win32",  "AMD64"):  "opensim-4.6-cp313-cp313-win_amd64.whl",
}

key = (sys.platform, platform.machine())
wheel = WHEEL_MAP.get(key)

if wheel is None:
    print(f"No opensim wheel available for platform: "
          f"{sys.platform} / {platform.machine()}")
    sys.exit(1)

wheel_path = WHEELS_DIR / wheel
print(f"Installing {wheel_path.name} ...")
subprocess.run([sys.executable, "-m", "pip", "install", str(wheel_path)], check=True)
