#!/bin/bash 

bash ../setup_replicas_cont.sh -r 19 -A pre.dat
mpirun -n 76 ../../llr_hb
