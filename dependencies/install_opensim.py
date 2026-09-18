"""
Install the OpenSim Python package that osimfit depends on.

By default this downloads a prebuilt wheel from the opensim-fitter release matching
this checkout's version and installs it with pip. Pass --from-source to build OpenSim
and Simbody from the pinned submodules instead.
"""

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import urllib.error
import urllib.request

REPO = 'opensim-org/opensim-fitter'
CWD = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(CWD)


def read_release_tag() -> str:
    """
    Return the release tag matching this checkout, derived from the osimfit version
    in pyproject.toml the same way the build-wheels workflow derives it.
    """
    import tomllib
    pyproject = pathlib.Path(REPO_DIR, 'pyproject.toml')
    with open(pyproject, 'rb') as f:
        return 'v' + tomllib.load(f)['project']['version']


def fetch_release_assets(tag: str) -> list[dict]:
    """
    Return the asset list for the given release tag from the GitHub API.
    """
    url = f'https://api.github.com/repos/{REPO}/releases/tags/{tag}'
    request = urllib.request.Request(
        url, headers={'Accept': 'application/vnd.github+json'})
    try:
        with urllib.request.urlopen(request) as response:
            return json.load(response)['assets']
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise RuntimeError(
                f"No release tagged '{tag}' was found in {REPO}. This checkout's "
                f"pyproject.toml version has no published wheels yet; build from "
                f"the submodules instead with:\n"
                f'    python install_opensim.py --from-source') from e
        raise RuntimeError(
            f'Could not read release {tag} from the GitHub API ({e.code} '
            f'{e.reason}).') from e


def select_wheel(assets: list[dict]) -> dict:
    """
    Return the asset whose wheel tags are compatible with the running interpreter and
    platform, preferring the most specific match.

    Raises
    ------
    RuntimeError
        If no asset is a wheel compatible with this interpreter and platform.
    """
    from packaging.tags import sys_tags
    from packaging.utils import parse_wheel_filename

    # sys_tags() is ordered most- to least-preferred, so the lowest index wins.
    tags_for_this_platform = list(sys_tags())
    supported = {tag: i for i, tag in enumerate(tags_for_this_platform)}

    best = None
    wheels = [a for a in assets if a['name'].endswith('.whl')]
    for asset in wheels:
        _, _, _, tags = parse_wheel_filename(asset['name'])
        ranks = [supported[t] for t in tags if t in supported]
        if ranks and (best is None or min(ranks) < best[0]):
            best = (min(ranks), asset)

    if best is None:
        # The first entry is the most specific tag this interpreter accepts, which
        # is exactly what a compatible wheel would have to be built for.
        wanted = tags_for_this_platform[0]
        available = '\n'.join(f'    {a["name"]}' for a in wheels) or '    (none)'
        raise RuntimeError(
            f'None of the wheels published with this release are compatible with '
            f'this platform ({wanted}).\n'
            f'Available wheels:\n{available}\n'
            f'Build from the submodules instead with:\n'
            f'    python install_opensim.py --from-source')

    return best[1]


def install_from_release():
    """
    Download and install the release wheel matching this interpreter and platform.
    """
    tag = read_release_tag()
    print(f'Looking for an OpenSim wheel in {REPO} release {tag}')
    asset = select_wheel(fetch_release_assets(tag))
    print(f'Installing {asset["name"]}')
    subprocess.check_call([sys.executable, '-m', 'pip', 'install',
                           asset['browser_download_url']])


def build_from_source():
    """
    Build Simbody and opensim-core from the pinned submodules and install the
    resulting Python package.
    """
    with open(os.path.join(CWD, 'config.yaml')) as f:
        import yaml
        config = yaml.safe_load(f)

    python_root_dir = pathlib.Path(config['python_root_dir']).as_posix()

    source_dirs = {}
    for name in ('opensim-core', 'simbody'):
        subprocess.run(['git', 'submodule', 'update', '--init', name],
                       check=True, cwd=REPO_DIR)
        source_dirs[name] = pathlib.Path(REPO_DIR, name).as_posix()
        version = subprocess.run(['git', 'describe', '--tags', '--always'],
                                 check=True, cwd=source_dirs[name],
                                 capture_output=True, text=True).stdout.strip()
        print(f'Building {name} at {version}')

    env = os.environ.copy()
    env['OPENSIM_CORE_SOURCE_DIR'] = source_dirs['opensim-core']
    env['SIMBODY_SOURCE_DIR'] = source_dirs['simbody']

    if sys.platform == 'win32':
        pwsh = shutil.which('pwsh') or shutil.which('powershell.exe') or 'pwsh'
        cmd = [pwsh, '-NoProfile', '-ExecutionPolicy', 'Bypass',
               '-File', 'install_opensim.ps1', python_root_dir]
    else:
        cmd = ['bash', 'install_opensim.sh', python_root_dir]

    subprocess.run(cmd, check=True, cwd=CWD, env=env)

    # Install the OpenSim Python package in the current environment.
    package = os.path.join(CWD, 'opensim', 'opensim_core_install', 'sdk',
                           'Python', '.')
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

    # On Windows, Python 3.8+ no longer resolves .pyd DLL dependencies via PATH;
    # only directories registered with os.add_dll_directory() are searched. The
    # opensim wheel's __init__.py already calls os.add_dll_directory() on its own
    # package directory, so copy the runtime DLLs next to the .pyd files there.
    if sys.platform == 'win32':
        import importlib.util
        spec = importlib.util.find_spec('opensim')
        pkg_dir = pathlib.Path(spec.submodule_search_locations[0])
        install_bin = (pathlib.Path(CWD) / 'opensim'
                       / 'opensim_core_install' / 'bin')
        for dll in install_bin.glob('*.dll'):
            shutil.copy2(dll, pkg_dir / dll.name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument(
        '--from-source', action='store_true',
        help='build OpenSim and Simbody from the pinned submodules instead of '
             'installing a prebuilt wheel from the matching release')
    args = parser.parse_args()

    if args.from_source:
        build_from_source()
    else:
        try:
            install_from_release()
        except RuntimeError as e:
            print(f'\nERROR: {e}', file=sys.stderr)
            return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
