#!/bin/bash 

bash ../setup_replicas_fxa.sh -r 19
find . -name "input_file*" | xargs sed -i '/rlx_seed =/c\rlx_seed = 1'
mpirun -n 76 ../../llr_hb -i input_file_fxa
