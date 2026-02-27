mkdir -p output

cd output
git clone https://github.com/fzierler/Hirep_LLR_SP
cd Hirep_LLR_SP/LLR_HB/
git checkout llr_heatbath_new_spn
cd ../../../  

cd output/Hirep_LLR_SP/LLR_HB/
make 
cp ./llr_hb ../../
cd ../../../

cd output
rm -rf ./LLR_sp4_4x4_8_Run_1/
cd -

python3 main.py --input_params_csv input/local_tests.csv --machine local --mpi_runner mpirun --path_llr_exec "../.."
cd output/LLR_sp4_4x4_8_Run_1/

bash setup_llr_repeat.sh