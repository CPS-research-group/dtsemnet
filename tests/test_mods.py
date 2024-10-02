'''Test imports of necessary packages for installation'''

import pytest

def test_mods():
    required_modules = ['numpy', 'torch', 'stable_baselines3', 'gym', 'matplotlib', 'pandas', 'seaborn', 'sklearn', 'scipy', 'agents']
    missing_modules = []

    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)

    if missing_modules:
        print("The following modules are missing:")
        for module in missing_modules:
            print(module)
        if 'agents' in missing_modules:
            print("use: python -m pip install -e")
    else:
        print("All required modules are installed.")