#!/usr/bin/env python
# coding: utf-8
import llranalysis.inputfiles as ifiles
import pandas as pd 
import os.path as op
import os
from shutil import copyfile

outdir     = "./output/"
input_dir  = "./input/templates/"
infofile   = "./input/local_tests.csv"

bash_files  = ["sp4_llr_start.sh","sp4_llr_start_cont.sh","sp4_llr_cont.sh","sp4_llr_fxa.sh"]
input_files = ["input_file_start", "input_file_start_cont", "input_file_cont", "input_file_fxa"]

## create a suitable name for the run:
def get_run_name(input_data):
    Lt  = input_data["Lt"].values[0]
    Ls  = input_data["Ls"].values[0]
    Rep = input_data["n_replicas"].values[0]
    return f"LLR_{Lt}x{Ls}_{Rep}"

input_data     = pd.read_csv(infofile)
template_dir   = input_data["machine"].values[0]
cores_per_node = input_data["cores_per_node"].values[0]
run_name       = get_run_name(input_data)
folder         = op.join(outdir,run_name)

os.makedirs(os.path.join(folder,"base"), exist_ok=True)
newinfofile = os.path.join(folder,"base","info.csv")
copyfile(infofile,newinfofile)

Eks, aks, dE = ifiles.predat_from_csv(folder,newinfofile)
ifiles.copy_identical_files(folder,input_dir)

for infile in input_files:
    ifiles.setup_input_files(op.join(input_dir,"base","input_file"),     op.join(folder,"base",infile)         ,newinfofile)
    ifiles.setup_input_files(op.join(input_dir,"base","input_file_rep"), op.join(folder,"base",infile+"_tmp")  ,newinfofile)
    ifiles.setup_energy_range(op.join(folder,"base",infile+"_tmp")     , op.join(folder,"base",infile+"_rep")  ,min(Eks),max(Eks))
    os.remove(op.join(folder,"base",infile+"_tmp"))

ifiles.setup_bash_files(op.join(input_dir,template_dir,"setup_llr_repeat.sh"),op.join(folder,"setup_llr_repeat.sh"),newinfofile)
for name in bash_files:
    ifiles.setup_batch_files(op.join(input_dir,template_dir,name),op.join(folder,name),newinfofile,cores_per_node)
