#!/bin/sh



# Small 1
# perl geoconv.pl mael/mael_small_001.pdb mael/mael_small_001.geo mael_test1.param
#nvprof ./program mael_test1.param mael/mael_small_001.pdb mael/ml_001_2mFo-DFc.map

# All
# perl geoconv.pl mael/ml_001.pdb mael/ml_001.geo mael.param 
nvprof ./program mael.param mael/mael_test1.pdb mael/ml_001_2mFo-DFc.map

#TRP
# perl geoconv.pl mael/mael_trp_001.pdb  mael/mael_trp_001.geo mael_trp1.param
#nvprof ./program mael_trp1.param mael/mael_trp_001.pdb mael/ml_001_2mFo-DFc.map

#PHE
# perl geoconv.pl mael/mael_phe_001.pdb mael/mael_phe_001.geo mael_phe1.param
#nvprof ./program mael_phe1.param mael/mael_phe_001.pdb mael/ml_001_2mFo-DFc.map
