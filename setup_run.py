#!/usr/bin/env python
# coding: utf-8

import llranalysis.inputfiles as ifiles
import pandas as pd 
import os.path as op

outdir    = "./output/"
input_dir = "./input/templates/"
infofile  = "./input/local_tests.csv"

# Either "dial3", "csd3", otherwise defaults to "local_reproducible"
machine = "local"

if machine == "dial3":
    cores_per_node = 128
    template_dir = "dial3"
elif machine == "csd3":
    cores_per_node = 76
    template_dir = "csd3"
else:
    cores_per_node = 8
    template_dir = "local_reproducible"

bash_files = ["sp4_llr_start.sh","sp4_llr_start_cont.sh","sp4_llr_cont.sh","sp4_llr_fxa.sh"]

## create a suitable name for the run:
input_data = pd.read_csv(infofile)
Lt  = input_data["Lt"].values[0]
Ls  = input_data["Ls"].values[0]
Rep = input_data["n_replicas"].values[0]
run_name = "LLR_"+str(Lt)+"x"+str(Ls)+"_"+str(Rep)
folder = op.join(outdir,run_name)

ifiles.predat_from_csv(folder,infofile)
ifiles.copy_identical_files(folder,input_dir)
ifiles.input_files_from_csv(op.join(input_dir,"base","input_file"),     op.join(folder,"base","input_file"),infofile)
ifiles.input_files_from_csv(op.join(input_dir,"base","input_file_rep"), op.join(folder,"base","input_file_rep"),infofile)

if template_dir == "local_reproducible":
    infile  = op.join(input_dir,"setup_llr_repeat_local.sh")
else:
    infile  = op.join(input_dir,"setup_llr_repeat.sh")
outfile = op.join(folder,"setup_llr_repeat.sh")
ifiles.input_files_from_csv(infile,outfile,infofile)

for name in bash_files:
    infile  = op.join(input_dir,template_dir,name)
    outfile = op.join(folder,name)
    ifiles.edit_batch_files(infile,outfile,infofile,cores_per_node)
