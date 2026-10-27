# Shared paths and toolchain settings for every build/bench script. Source it, don't run it.
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export WS
export INSTALL="$WS/install"
export BUILD="$WS/build"
export JOBS="${JOBS:-10}"
export CC=gcc CXX=g++ FC=gfortran
