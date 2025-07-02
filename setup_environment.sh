#!/usr/bin/env bash

# Activate the virtual environment
source .venv/bin/activate


# Add python packages, including my own scripts
venv_site_packages=$(python -c "import site; print(site.getsitepackages()[0])")
export PYTHONPATH="${PYTHONPATH}:${venv_site_packages}:${PWD}"



#export QT_FONT_DPI=$MAYA_FONT_DPI
# Launch Maya
exec /usr/autodesk/maya2025/bin/maya &

# Launch NXT in the background
#nxt ui &
