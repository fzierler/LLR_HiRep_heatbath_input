n_RM=10
n_NR=10
run_name="sp4_5x80_95"

bname=$(basename "$PWD")
for i in {0..0}; do
    # create repeat dircetory
    cp base $i/ -r
    cd $i

    # submit all jobs
    id=$(sbatch --parsable -J "${bname}_repeat${i}_therm" ../sp4_llr_therm.sh)
    id=$(sbatch --array=1-${n_NR}%1 --parsable -J "${bname}_repeat${i}_newton_raphson" --dependency=afterok:$id ../sp4_llr_newton_raphson.sh)
    id=$(sbatch --array=1-${n_RM}%1 --parsable -J "${bname}_repeat${i}_robbins_monro"  --dependency=afterok:$id ../sp4_llr_robbins_monro.sh)
    #sbatch --dependency=afterok:$id -J "${bname}_repeat${i}_fxa" ../sp4_llr_fxa.sh

    # move on to next repeat
    cd ..
done
