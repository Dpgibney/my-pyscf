#!/usr/bin/env bash
# OpenMPI 4.1.8 in user space on node2 (~/Chemistry/openmpi-4.1.8), for ORCA 6.1.1's shared build
# (orca_6_1_1_linux_x86-64_shared_openmpi418_nodmrg), whose *_mpi programs need libmpi.so.40. With the Fortran
# bindings (gfortran): ORCA calls MPI_Type_match_size, which looks up OpenMPI's Fortran datatypes; a C-only build fails
# it with MPI_ERR_ARG in orca_startup_mpi. -std=gnu17 because GCC 15 defaults to C23.
# Remove with: rm -rf ~/Chemistry/openmpi-4.1.8*
set -euo pipefail
SHA256=466f68e3132a1dc02710cc2011fafced8336d98359fa2dae4dddcfd5719f12a9  # www.open-mpi.org/software/ompi/v4.1/
SRC=~/Chemistry/openmpi-4.1.8-src
PREFIX=~/Chemistry/openmpi-4.1.8
mkdir -p $SRC && cd $SRC
[[ -f openmpi-4.1.8.tar.bz2 ]] || wget -q https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.8.tar.bz2
echo "$SHA256  openmpi-4.1.8.tar.bz2" | sha256sum -c -
rm -rf openmpi-4.1.8 && tar xjf openmpi-4.1.8.tar.bz2 && cd openmpi-4.1.8
echo "configure $(date +%H:%M:%S)"
./configure --prefix=$PREFIX --disable-oshmem --enable-shared --disable-static \
    CFLAGS="-O2 -std=gnu17" CXXFLAGS="-O2" > ../configure.log 2>&1
echo "make $(date +%H:%M:%S)"
make -j6 > ../make.log 2>&1
rm -rf $PREFIX
make install > ../install.log 2>&1
$PREFIX/bin/mpirun --version | head -1
echo "NODE2 OPENMPI DONE $(date +%H:%M:%S)"
