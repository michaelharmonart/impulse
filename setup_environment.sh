#!/usr/bin/env bash

# Activate the virtual environment
source .venv/bin/activate


# Add python packages, including my own scripts
venv_site_packages=$(python -c "import site; print(site.getsitepackages()[0])")
export PYTHONPATH="${PYTHONPATH}:${venv_site_packages}:${HOME}/impulse/"


# Launch Maya
exec /usr/autodesk/maya2025/bin/maya &
