n_RM=10
n_NR=10
run_name="sp4_5x80_95"

bname=$(basename "$PWD")
for i in {0..0}; do
    # create repeat dircetory
    cp base $i/ -r
    cd $i
    N_NRm1=$((n_NR-1))

    # submit all jobs
    id=$(sbatch --parsable -J "${bname}_repeat${i}_start" ../sp4_llr_start.sh)
    id=$(sbatch --array=1-${N_NRm1}%1 --parsable -J "${bname}_repeat${i}_start_cont" --dependency=afterok:$id ../sp4_llr_start_cont.sh)
    id=$(sbatch --array=1-${n_RM}%1   --parsable -J "${bname}_repeat${i}_cont"       --dependency=afterok:$id ../sp4_llr_cont.sh)
    sbatch --dependency=afterok:$id -J "${bname}_repeat${i}_fxa" ../sp4_llr_fxa.sh

    # move on to next repeat
    cd ..
done
