#!/bin/bash 

bash ../update_replicas.sh -r 19 -i input_file_fxa
find . -name "input_file*" | xargs sed -i '/rlx_seed =/c\rlx_seed = 1'
mpirun -n 76 ../../llr_hb -i input_file_fxa
