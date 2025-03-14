#!/bin/bash 

bash ../setup_replicas_start_cont.sh -r 19
find . -name "input_file*" | xargs sed -i '/rlx_seed =/c\rlx_seed = 1'
mpirun -n 76 ../../llr_hb
