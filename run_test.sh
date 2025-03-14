mkdir -p output
cd output
git clone https://github.com/fzierler/Hirep_LLR_SP
cd Hirep_LLR_SP/LLR_HB/
make 
cp ./llr_hb ../../
cd ../../../

python setup_run.py
cd output/LLR_4x4_8/

# now we have a test run in output/LLR_4x4_8/0/
bash setup_llr_repeat.sh
mv 0/ 0_reference/

# Now repeat the run again, to test whether it is fully reproducible
bash setup_llr_repeat.sh
mv 0/ 0_new/

# We need to remove the output files beore comparing since they differ in the timings of the operations
find . -name out_0 | xargs rm

diff -r -q 0_new/ 0_reference/
meld 0_new/ 0_reference/