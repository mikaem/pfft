source .fftw_cray
./bootstrap.sh
#./configure --prefix=/scratch/x_mortenm/Software --exec-prefix=/scratch/x_mortenm/Software CPPFLAGS="-I/scratch/x_mortenm/Software/include" LDFLAGS="-Wl,-rpath,/scratch/x_mortenm/Software/lib -L/scratch/x_mortenm/Software/lib" FC=ftn CC=cc MPICC=cc MPIFC=ftn
./configure --disable-fortran --prefix=$SCRATCH/Software/cray --enable-static=no LDFLAGS="-static -Wl,--no-export-dynamic" FC=ftn CC=cc
make
make install
