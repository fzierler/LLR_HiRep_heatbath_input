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
    bash ../sp4_llr_start_cont.sh
    bash ../sp4_llr_cont.sh
    bash ../sp4_llr_fxa.sh

    # move on to next repeat
    cd ..
done
