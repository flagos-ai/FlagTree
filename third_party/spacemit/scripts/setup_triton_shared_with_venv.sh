# sets up a python virtual environment, clones triton shared, and does some preliminary pip installs
# in preparation for build_llvm_for_sanitizers and build_triton_shared_for_sanitizers
# user needs to activate their venv after this

#!/bin/bash
set -e
set -x

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <desired path to venv> <desired path to triton shared>"
  exit 1
fi

VENV_PATH="$(realpath "$1")"
TRITON_SHARED_PATH="$(realpath "$2")"

# check if the path exists
if [ ! -e "$VENV_PATH" ]; then
  echo "Error: Path '$VENV_PATH' does not exist."
  exit 1
fi

if [ ! -e "$TRITON_SHARED_PATH" ]; then
  echo "Error: Path '$TRITON_SHARED_PATH' does not exist."
  exit 1
fi

# set up venv
python3 -m venv ${VENV_PATH}/venv
source ${VENV_PATH}/venv/bin/activate

# clone triton-shared
git clone --recurse-submodules https://github.com/microsoft/triton-shared.git "${TRITON_SHARED_PATH}/triton_shared"

python3 -m pip install --upgrade pip
python3 -m pip install cmake==3.24 ninja pytest-xdist pybind11 setuptools torch==2.7 # need torch 2.7 for triton 3.3.0
sudo apt-get install -y ccache clang lld

# echo to user to start virtual environment
echo "Please start the virtual environment: source ${VENV_PATH}/venv/bin/activate"
echo "This export is recommended for subsequent triton-shared builds: export TRITON_PLUGIN_DIRS="${TRITON_SHARED_PATH}/triton_shared"
