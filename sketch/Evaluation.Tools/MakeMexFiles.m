delete('*.mexa64');
mex mex_CalcHammDist.cc CXXFLAGS="\$CXXFLAGS -mpopcnt -O3 -Wall -fopenmp"  LDFLAGS="\$LDFLAGS -fopenmp"