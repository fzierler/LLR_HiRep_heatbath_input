mkdir -p output

cd output
git clone https://github.com/fzierler/Hirep_LLR_SP
cd Hirep_LLR_SP/LLR_HB/
git checkout llr_heatbath_new_spn

make 
cp ./llr_hb ../../
cd ../../../

cd output
rm -rf ./LLR_4x4_8/
cd -
python setup_run.py
cd output/LLR_4x4_8/

# now we have a test run in output/LLR_4x4_8/0/
rm -rf ../0_reference/ 
bash setup_llr_repeat.sh
mv 0/ ../0_reference/
cd -

cd output
git clone https://github.com/fzierler/Hirep_LLR_SP
cd Hirep_LLR_SP/LLR_HB/
git checkout cb25283d87269a684a25f6049290595cb0ba2d11

make 
cp ./llr_hb ../../
cd ../../../

cd output
rm -rf ./LLR_4x4_8/
cd -
python setup_run.py
cd output/LLR_4x4_8/

# Now repeat the run again, to test whether it is fully reproducible
rm -rf ../0_new/
bash setup_llr_repeat.sh
mv 0/ ../0_new/

# We need to remove the output files beore comparing since they differ in the timings of the operations
#find .. -name out_0 | xargs rm

#diff -rqwB ../0_new/ ../0_reference/
meld ../0_new/ ../0_reference/