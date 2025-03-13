mkdir -p output
cd output
git clone https://github.com/fzierler/Hirep_LLR_SP
cd Hirep_LLR_SP/LLR_HB/
make 
cp ./llr_hb ../../
cd ../../../

python setup_run.py
cd output/LLR_4x4_8/

bash setup_llr_repeat.sh
# now we have a test run in output/LLR_4x4_8/0/
# remove everything but the input files
find . -name rand_state* | xargs rm
find . -name run1* | xargs rm
find . -name Cnfg | xargs rm -r
find . -name CSV | xargs rm -r
find . -name out_0 | xargs rm
find . -name err_0 | xargs rm
tar cf - ./0/ | sha1sum  # obtain checksum of the resulting directory 
mv 0/ 0_reference/       # The input files can now be easily compared with a tool like meld

# Now repeat the run again, to test whether it is fully reproducible
bash setup_llr_repeat.sh
find . -name rand_state* | xargs rm
find . -name run1* | xargs rm
find . -name Cnfg | xargs rm -r
find . -name CSV | xargs rm -r
find . -name out_0 | xargs rm
find . -name err_0 | xargs rm
tar cf - ./0/ | sha1sum  # obtain checksum of the resulting directory 
mv 0/ 0_new/

meld 0_new/ 0_reference/