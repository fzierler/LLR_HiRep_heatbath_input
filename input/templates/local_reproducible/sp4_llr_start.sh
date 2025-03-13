#!/bin/bash 

bash ../setup_replicas.sh -r 19 -A pre.dat
find . -name "input_file*" | xargs sed -i '/rlx_seed =/c\rlx_seed = 1'
mpirun -n 76 ../../llr_hb
