n_RM=50
n_NR=10
run_name=sp4_4x20_8

bname=$(basename "$PWD")
for i in {0..0}; do
    # create repeat dircetory
    cp base $i/ -r
    cd $i
    N_NRm1=$((n_NR-1))

    # submit all jobs
    bash ../sp4_llr_start.sh
    for i in $(seq 1 $N_NRm1); do
        bash ../sp4_llr_start_cont.sh
    done
    for i in $(seq 1 $n_RM); do
        bash ../sp4_llr_cont.sh
    done
    #bash ../sp4_llr_fxa.sh

    # move on to next repeat
    cd ..
done
